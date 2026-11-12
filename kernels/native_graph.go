package kernels

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// NativeGraph is a CUDA driver graph containing kernel, copy, and memset nodes.
// Referenced buffers are retained until Close synchronizes graph launches.
type NativeGraph struct {
	mu         sync.Mutex
	handle     unsafe.Pointer
	exec       unsafe.Pointer
	deviceID   int
	modules    map[*NativeModule]struct{}
	buffers    []*memory.Memory
	retentions []*memory.Retention
	streams    map[*streams.NativeStream]struct{}
	invalid    bool
}

// NativeGraphNode identifies a node owned by a NativeGraph.
type NativeGraphNode struct {
	graph  *NativeGraph
	handle unsafe.Pointer
}

// DependsOn reports whether dependency is a direct dependency of this node.
func (node *NativeGraphNode) DependsOn(dependency *NativeGraphNode) (bool, error) {
	if node == nil || dependency == nil || node.graph == nil || node.graph != dependency.graph {
		return false, fmt.Errorf("nodes must belong to the same graph")
	}
	graph := node.graph
	graph.mu.Lock()
	defer graph.mu.Unlock()
	if graph.handle == nil {
		return false, fmt.Errorf("native graph is closed")
	}
	if graph.invalid {
		return false, fmt.Errorf("native graph is invalid and must be closed")
	}
	var found bool
	err := internal.RunOnDevice(graph.deviceID, func() error {
		var err error
		found, err = nativeGraphHasDependency(node.handle, dependency.handle)
		return err
	})
	return found, err
}

// NewNativeGraph creates a CUDA graph on deviceID; it does not simulate one.
func NewNativeGraph(deviceID int) (*NativeGraph, error) {
	if !internal.ShouldUseCuda() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= internal.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var handle unsafe.Pointer
	err := internal.RunOnDevice(deviceID, func() error {
		var err error
		handle, err = createNativeGraph()
		return err
	})
	if err != nil {
		if handle != nil {
			cleanupErr := internal.RunOnDevice(deviceID, func() error { return destroyNativeGraph(handle) })
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return &NativeGraph{
		handle: handle, deviceID: deviceID,
		modules: make(map[*NativeModule]struct{}),
		streams: make(map[*streams.NativeStream]struct{}),
	}, nil
}

func (g *NativeGraph) prepareNode(dependencies []*NativeGraphNode) ([]unsafe.Pointer, error) {
	if g.handle == nil {
		return nil, fmt.Errorf("native graph is closed")
	}
	if g.invalid {
		return nil, fmt.Errorf("native graph is invalid and must be closed")
	}
	if g.exec != nil {
		return nil, fmt.Errorf("cannot add nodes after graph instantiation")
	}
	deps := make([]unsafe.Pointer, len(dependencies))
	seen := make(map[*NativeGraphNode]struct{}, len(dependencies))
	for i, dependency := range dependencies {
		if dependency == nil || dependency.graph != g || dependency.handle == nil {
			return nil, fmt.Errorf("dependency %d does not belong to this graph", i)
		}
		if _, exists := seen[dependency]; exists {
			return nil, fmt.Errorf("duplicate graph dependency")
		}
		seen[dependency] = struct{}{}
		deps[i] = dependency.handle
	}
	return deps, nil
}

func retainGraphBuffers(buffers []*memory.Memory) ([]*memory.Retention, error) {
	retentions := make([]*memory.Retention, 0, len(buffers))
	for _, buffer := range buffers {
		retention, err := buffer.Retain()
		if err != nil {
			for _, acquired := range retentions {
				acquired.Release()
			}
			return nil, err
		}
		retentions = append(retentions, retention)
	}
	return retentions, nil
}

func (g *NativeGraph) finishNode(node unsafe.Pointer, err error, buffers []*memory.Memory, retentions []*memory.Retention) (*NativeGraphNode, error) {
	if err != nil && node == nil {
		for _, retention := range retentions {
			retention.Release()
		}
		return nil, err
	}
	g.buffers = append(g.buffers, buffers...)
	g.retentions = append(g.retentions, retentions...)
	if err != nil || node == nil {
		g.invalid = true
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("CUDA graph node creation returned no handle")
	}
	return &NativeGraphNode{graph: g, handle: node}, nil
}

// AddKernelNode adds a kernel and dependencies already belonging to this graph.
// Its argument values are captured when the node is added.
func (g *NativeGraph) AddKernelNode(function *NativeFunction, dependencies []*NativeGraphNode, grid, block Dim3, sharedBytes int, args ...NativeArg) (*NativeGraphNode, error) {
	if g == nil || function == nil || function.module == nil {
		return nil, fmt.Errorf("graph and native function are required")
	}
	if function.module.deviceID != g.deviceID {
		return nil, fmt.Errorf("function device %d does not match graph device %d", function.module.deviceID, g.deviceID)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	deps, err := g.prepareNode(dependencies)
	if err != nil {
		return nil, err
	}
	function.module.mu.Lock()
	defer function.module.mu.Unlock()
	if function.module.handle == nil {
		return nil, fmt.Errorf("native module is closed")
	}
	var buffers []*memory.Memory
	for _, arg := range args {
		if arg.isBuffer {
			buffers = append(buffers, arg.buffer)
		}
	}
	retentions, err := retainGraphBuffers(buffers)
	if err != nil {
		return nil, err
	}
	values, err := function.prepareArguments(grid, block, sharedBytes, args)
	if err != nil {
		for _, retention := range retentions {
			retention.Release()
		}
		return nil, err
	}
	var node unsafe.Pointer
	err = internal.RunOnDevice(g.deviceID, func() error {
		var err error
		node, err = addNativeGraphKernel(g.handle, function.handle, deps, grid, block, sharedBytes, values)
		return err
	})
	if err != nil && node == nil {
		for _, retention := range retentions {
			retention.Release()
		}
		return nil, err
	}
	if _, exists := g.modules[function.module]; !exists {
		g.modules[function.module] = struct{}{}
		function.module.graphRefs++
	}
	return g.finishNode(node, err, buffers, retentions)
}

func (g *NativeGraph) validateBuffer(buffer *memory.Memory, size int64, pinnedAllowed bool) error {
	if buffer == nil || buffer.Ptr() == nil || size <= 0 || size > buffer.Size() {
		return fmt.Errorf("graph buffer is nil, freed, or too small")
	}
	switch buffer.GetType() {
	case memory.TypePinned:
		if !pinnedAllowed {
			return fmt.Errorf("graph operation requires device or managed memory")
		}
	case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
		if buffer.GetDeviceID() != g.deviceID {
			return fmt.Errorf("graph buffer device %d does not match graph device %d", buffer.GetDeviceID(), g.deviceID)
		}
	default:
		return fmt.Errorf("graph operation requires device, managed, or pinned memory")
	}
	return nil
}

// AddMemcpyNode adds a one-dimensional CUDA copy between device/managed and
// pinned buffers. At least one buffer must be device or managed memory.
func (g *NativeGraph) AddMemcpyNode(dst, src *memory.Memory, size int64, dependencies []*NativeGraphNode) (*NativeGraphNode, error) {
	if g == nil {
		return nil, fmt.Errorf("nil native graph")
	}
	if err := g.validateBuffer(dst, size, true); err != nil {
		return nil, err
	}
	if err := g.validateBuffer(src, size, true); err != nil {
		return nil, err
	}
	if dst.Ptr() == src.Ptr() || (dst.GetType() == memory.TypePinned && src.GetType() == memory.TypePinned) {
		return nil, fmt.Errorf("graph copy requires distinct buffers and at least one device buffer")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	deps, err := g.prepareNode(dependencies)
	if err != nil {
		return nil, err
	}
	buffers := []*memory.Memory{dst, src}
	retentions, err := retainGraphBuffers(buffers)
	if err != nil {
		return nil, err
	}
	var node unsafe.Pointer
	err = internal.RunOnDevice(g.deviceID, func() error {
		var err error
		node, err = addNativeGraphMemcpy(g.handle, deps, dst.Ptr(), src.Ptr(), size, dst.GetType() == memory.TypePinned, src.GetType() == memory.TypePinned)
		return err
	})
	return g.finishNode(node, err, buffers, retentions)
}

// AddMemsetNode adds a CUDA memset of count elements of size 1, 2, or 4 bytes.
func (g *NativeGraph) AddMemsetNode(dst *memory.Memory, value uint32, elementSize, count int, dependencies []*NativeGraphNode) (*NativeGraphNode, error) {
	if g == nil {
		return nil, fmt.Errorf("nil native graph")
	}
	if (elementSize != 1 && elementSize != 2 && elementSize != 4) || count <= 0 {
		return nil, fmt.Errorf("invalid graph memset element size or count")
	}
	if dst == nil || int64(count) > dst.Size()/int64(elementSize) {
		return nil, fmt.Errorf("graph memset buffer is nil or too small")
	}
	if err := g.validateBuffer(dst, int64(count)*int64(elementSize), false); err != nil {
		return nil, err
	}
	if uintptr(dst.Ptr())%uintptr(elementSize) != 0 {
		return nil, fmt.Errorf("graph memset buffer is not aligned")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	deps, err := g.prepareNode(dependencies)
	if err != nil {
		return nil, err
	}
	buffers := []*memory.Memory{dst}
	retentions, err := retainGraphBuffers(buffers)
	if err != nil {
		return nil, err
	}
	var node unsafe.Pointer
	err = internal.RunOnDevice(g.deviceID, func() error {
		var err error
		node, err = addNativeGraphMemset(g.handle, deps, dst.Ptr(), value, elementSize, count)
		return err
	})
	return g.finishNode(node, err, buffers, retentions)
}

// Instantiate creates the executable graph. Nodes cannot be added afterward.
func (g *NativeGraph) Instantiate() error {
	if g == nil {
		return fmt.Errorf("nil native graph")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.handle == nil {
		return fmt.Errorf("native graph is closed")
	}
	if g.invalid {
		return fmt.Errorf("native graph is invalid and must be closed")
	}
	if g.exec != nil {
		return fmt.Errorf("native graph is already instantiated")
	}
	err := internal.RunOnDevice(g.deviceID, func() error {
		var err error
		g.exec, err = instantiateNativeGraph(g.handle)
		return err
	})
	if err != nil && g.exec != nil {
		g.invalid = true
	}
	return err
}

// Launch enqueues the instantiated graph on stream.
func (g *NativeGraph) Launch(stream *streams.NativeStream) error {
	if g == nil || stream == nil {
		return fmt.Errorf("graph and native stream are required")
	}
	if stream.DeviceID() != g.deviceID {
		return fmt.Errorf("stream device %d does not match graph device %d", stream.DeviceID(), g.deviceID)
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.handle == nil || g.exec == nil {
		return fmt.Errorf("native graph is closed or not instantiated")
	}
	if g.invalid {
		return fmt.Errorf("native graph is invalid and must be closed")
	}
	for _, buffer := range g.buffers {
		if buffer.Ptr() == nil {
			return fmt.Errorf("a graph buffer has been freed")
		}
	}
	return stream.WithNativeHandle(func(handle unsafe.Pointer) error {
		g.streams[stream] = struct{}{}
		return launchNativeGraph(g.exec, handle)
	})
}

// Close synchronizes streams used by the graph and destroys its resources.
func (g *NativeGraph) Close() error {
	if g == nil {
		return nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.handle == nil {
		return nil
	}
	var syncErrors []error
	for stream := range g.streams {
		if err := stream.Synchronize(); err != nil && !errors.Is(err, streams.ErrNativeStreamClosed) {
			syncErrors = append(syncErrors, err)
		}
	}
	if err := errors.Join(syncErrors...); err != nil {
		return err
	}
	err := internal.RunOnDevice(g.deviceID, func() error {
		if g.exec != nil {
			if err := destroyNativeGraphExec(g.exec); err != nil {
				return err
			}
			g.exec = nil
		}
		if err := destroyNativeGraph(g.handle); err != nil {
			return err
		}
		g.handle = nil
		return nil
	})
	if g.handle == nil {
		for module := range g.modules {
			module.mu.Lock()
			module.graphRefs--
			module.mu.Unlock()
		}
		g.modules = nil
		g.buffers = nil
		for _, retention := range g.retentions {
			retention.Release()
		}
		g.retentions = nil
		g.streams = nil
	}
	return err
}
