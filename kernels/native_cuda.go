//go:build cuda

package kernels

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcuda -lcudart
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcuda -lcudart

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

static CUresult gocudaLoadPTX(void **module, const char *ptx) {
	return cuModuleLoadData((CUmodule*)module, ptx);
}
static CUresult gocudaModuleFunction(void **function, void *module, const char *name) {
	return cuModuleGetFunction((CUfunction*)function, (CUmodule)module, name);
}
typedef CUresult (CUDAAPI *gocudaParamInfoFn)(CUfunction, size_t, size_t*, size_t*);
static CUresult gocudaFindParamInfo(void **function) {
#if CUDA_VERSION >= 12000
	return cuGetProcAddress("cuFuncGetParamInfo", function, CUDA_VERSION, 0, NULL);
#else
	return cuGetProcAddress("cuFuncGetParamInfo", function, CUDA_VERSION, 0);
#endif
}
static CUresult gocudaParamInfo(void *query, void *function, size_t index, size_t *size) {
	size_t offset = 0;
	return ((gocudaParamInfoFn)query)((CUfunction)function, index, &offset, size);
}
static CUresult gocudaLaunch(void *function, void *stream,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	unsigned int shared, void **args) {
	return cuLaunchKernel((CUfunction)function, gx, gy, gz, bx, by, bz,
		shared, (CUstream)stream, args, NULL);
}
static CUresult gocudaUnload(void *module) {
	return cuModuleUnload((CUmodule)module);
}
static CUresult gocudaGraphCreate(void **graph) {
	return cuGraphCreate((CUgraph*)graph, 0);
}
static CUresult gocudaGraphAddKernel(void **node, void *graph, void **dependencies,
	size_t count, void *function,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	unsigned int shared, void **args) {
	CUDA_KERNEL_NODE_PARAMS params = {0};
	params.func = (CUfunction)function;
	params.gridDimX = gx; params.gridDimY = gy; params.gridDimZ = gz;
	params.blockDimX = bx; params.blockDimY = by; params.blockDimZ = bz;
	params.sharedMemBytes = shared;
	params.kernelParams = args;
	return cuGraphAddKernelNode((CUgraphNode*)node, (CUgraph)graph,
		(const CUgraphNode*)dependencies, count, &params);
}
static CUresult gocudaGraphAddMemcpy(void **node, void *graph, void **dependencies,
	size_t count, void *dst, void *src, size_t size, int dstHost, int srcHost) {
	CUcontext context = NULL;
	CUresult status = cuCtxGetCurrent(&context);
	if (status != CUDA_SUCCESS) return status;
	if (context == NULL) return CUDA_ERROR_INVALID_CONTEXT;
	CUDA_MEMCPY3D params = {0};
	params.srcMemoryType = srcHost ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
	params.dstMemoryType = dstHost ? CU_MEMORYTYPE_HOST : CU_MEMORYTYPE_DEVICE;
	if (srcHost) params.srcHost = src; else params.srcDevice = (CUdeviceptr)(uintptr_t)src;
	if (dstHost) params.dstHost = dst; else params.dstDevice = (CUdeviceptr)(uintptr_t)dst;
	params.srcPitch = size; params.dstPitch = size;
	params.srcHeight = 1; params.dstHeight = 1;
	params.WidthInBytes = size; params.Height = 1; params.Depth = 1;
	return cuGraphAddMemcpyNode((CUgraphNode*)node, (CUgraph)graph,
		(const CUgraphNode*)dependencies, count, &params, context);
}
static CUresult gocudaGraphAddMemset(void **node, void *graph, void **dependencies,
	size_t count, void *dst, unsigned int value, unsigned int elementSize, size_t width) {
	CUcontext context = NULL;
	CUresult status = cuCtxGetCurrent(&context);
	if (status != CUDA_SUCCESS) return status;
	if (context == NULL) return CUDA_ERROR_INVALID_CONTEXT;
	CUDA_MEMSET_NODE_PARAMS params = {0};
	params.dst = (CUdeviceptr)(uintptr_t)dst;
	params.value = value; params.elementSize = elementSize;
	params.width = width; params.height = 1;
	return cuGraphAddMemsetNode((CUgraphNode*)node, (CUgraph)graph,
		(const CUgraphNode*)dependencies, count, &params, context);
}
static CUresult gocudaGraphInstantiate(void **exec, void *graph) {
	return cuGraphInstantiate((CUgraphExec*)exec, (CUgraph)graph, 0);
}
static CUresult gocudaGraphLaunch(void *exec, void *stream) {
	return cuGraphLaunch((CUgraphExec)exec, (CUstream)stream);
}
static CUresult gocudaGraphExecDestroy(void *exec) {
	return cuGraphExecDestroy((CUgraphExec)exec);
}
static CUresult gocudaGraphDestroy(void *graph) {
	return cuGraphDestroy((CUgraph)graph);
}
static CUresult gocudaGraphDependencies(void *node, void **dependencies, size_t *count) {
#if CUDA_VERSION >= 12030
	return cuGraphNodeGetDependencies((CUgraphNode)node, (CUgraphNode*)dependencies, NULL, count);
#else
	return cuGraphNodeGetDependencies((CUgraphNode)node, (CUgraphNode*)dependencies, count);
#endif
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func nativeDriverError(operation string, status C.CUresult) error {
	if status == C.CUDA_SUCCESS {
		return nil
	}
	var message *C.char
	if C.cuGetErrorString(status, &message) == C.CUDA_SUCCESS && message != nil {
		return fmt.Errorf("%s failed: %s (%d)", operation, C.GoString(message), int(status))
	}
	return fmt.Errorf("%s failed with CUDA driver status %d", operation, int(status))
}

func loadNativePTX(ptx string) (unsafe.Pointer, error) {
	if err := ensureNativeContext(); err != nil {
		return nil, err
	}
	code := C.CString(ptx)
	defer C.free(unsafe.Pointer(code))
	var handle unsafe.Pointer
	status := C.gocudaLoadPTX(&handle, code)
	return handle, nativeDriverError("cuModuleLoadData", status)
}

func ensureNativeContext() error {
	if status := C.cudaFree(nil); status != C.cudaSuccess {
		return fmt.Errorf("CUDA context initialization failed: %s (%d)", C.GoString(C.cudaGetErrorString(status)), int(status))
	}
	if status := C.cuInit(0); status != C.CUDA_SUCCESS {
		return nativeDriverError("cuInit", status)
	}
	return nil
}

func nativeModuleFunction(module unsafe.Pointer, name string) (unsafe.Pointer, []int, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var function unsafe.Pointer
	status := C.gocudaModuleFunction(&function, module, cName)
	if err := nativeDriverError("cuModuleGetFunction", status); err != nil {
		return nil, nil, err
	}
	var query unsafe.Pointer
	status = C.gocudaFindParamInfo(&query)
	if err := nativeDriverError("cuGetProcAddress(cuFuncGetParamInfo)", status); err != nil {
		return nil, nil, err
	}
	if query == nil {
		return nil, nil, fmt.Errorf("CUDA driver does not expose kernel parameter metadata")
	}
	var sizes []int
	for index := 0; index < 65536; index++ {
		var size C.size_t
		status = C.gocudaParamInfo(query, function, C.size_t(index), &size)
		if status == C.CUDA_ERROR_INVALID_VALUE {
			return function, sizes, nil
		}
		if err := nativeDriverError("cuFuncGetParamInfo", status); err != nil {
			return nil, nil, err
		}
		if uint64(size) > uint64(^uint(0)>>1) {
			return nil, nil, fmt.Errorf("kernel parameter size exceeds supported range")
		}
		sizes = append(sizes, int(size))
	}
	return nil, nil, fmt.Errorf("kernel has too many parameters")
}

func withNativeArguments(values [][8]byte, fn func(*unsafe.Pointer) error) error {
	var parameters unsafe.Pointer
	var storage unsafe.Pointer
	if len(values) > 0 {
		parameters = C.malloc(C.size_t(len(values)) * C.size_t(unsafe.Sizeof(uintptr(0))))
		if parameters == nil {
			return fmt.Errorf("allocating kernel argument pointers failed")
		}
		defer C.free(parameters)
		storage = C.malloc(C.size_t(len(values)) * 8)
		if storage == nil {
			return fmt.Errorf("allocating kernel argument storage failed")
		}
		defer C.free(storage)
		pointers := unsafe.Slice((*unsafe.Pointer)(parameters), len(values))
		for i, value := range values {
			pointers[i] = unsafe.Add(storage, i*8)
			copy(unsafe.Slice((*byte)(pointers[i]), 8), value[:])
		}
	}
	return fn((*unsafe.Pointer)(parameters))
}

func launchNativeFunction(function, stream unsafe.Pointer, grid, block Dim3, sharedBytes int, values [][8]byte) error {
	return withNativeArguments(values, func(parameters *unsafe.Pointer) error {
		status := C.gocudaLaunch(function, stream,
			C.uint(grid.X), C.uint(grid.Y), C.uint(grid.Z),
			C.uint(block.X), C.uint(block.Y), C.uint(block.Z),
			C.uint(sharedBytes), parameters)
		return nativeDriverError("cuLaunchKernel", status)
	})
}

func unloadNativeModule(module unsafe.Pointer) error {
	return nativeDriverError("cuModuleUnload", C.gocudaUnload(module))
}

func createNativeGraph() (unsafe.Pointer, error) {
	if err := ensureNativeContext(); err != nil {
		return nil, err
	}
	var graph unsafe.Pointer
	status := C.gocudaGraphCreate(&graph)
	return graph, nativeDriverError("cuGraphCreate", status)
}

func withNativeGraphDependencies(deps []unsafe.Pointer, fn func(*unsafe.Pointer) error) error {
	var dependencyStorage unsafe.Pointer
	if len(deps) > 0 {
		dependencyStorage = C.malloc(C.size_t(len(deps)) * C.size_t(unsafe.Sizeof(uintptr(0))))
		if dependencyStorage == nil {
			return fmt.Errorf("allocating graph dependencies failed")
		}
		defer C.free(dependencyStorage)
		copy(unsafe.Slice((*unsafe.Pointer)(dependencyStorage), len(deps)), deps)
	}
	return fn((*unsafe.Pointer)(dependencyStorage))
}

func addNativeGraphKernel(graph, function unsafe.Pointer, deps []unsafe.Pointer, grid, block Dim3, sharedBytes int, values [][8]byte) (unsafe.Pointer, error) {
	var node unsafe.Pointer
	err := withNativeGraphDependencies(deps, func(dependencies *unsafe.Pointer) error {
		return withNativeArguments(values, func(parameters *unsafe.Pointer) error {
			status := C.gocudaGraphAddKernel(&node, graph, dependencies, C.size_t(len(deps)), function,
				C.uint(grid.X), C.uint(grid.Y), C.uint(grid.Z),
				C.uint(block.X), C.uint(block.Y), C.uint(block.Z),
				C.uint(sharedBytes), parameters)
			return nativeDriverError("cuGraphAddKernelNode", status)
		})
	})
	return node, err
}

func addNativeGraphMemcpy(graph unsafe.Pointer, deps []unsafe.Pointer, dst, src unsafe.Pointer, size int64, dstHost, srcHost bool) (unsafe.Pointer, error) {
	var node unsafe.Pointer
	var dstFlag, srcFlag C.int
	if dstHost {
		dstFlag = 1
	}
	if srcHost {
		srcFlag = 1
	}
	err := withNativeGraphDependencies(deps, func(dependencies *unsafe.Pointer) error {
		status := C.gocudaGraphAddMemcpy(&node, graph, dependencies, C.size_t(len(deps)), dst, src, C.size_t(size), dstFlag, srcFlag)
		return nativeDriverError("cuGraphAddMemcpyNode", status)
	})
	return node, err
}

func addNativeGraphMemset(graph unsafe.Pointer, deps []unsafe.Pointer, dst unsafe.Pointer, value uint32, elementSize, count int) (unsafe.Pointer, error) {
	var node unsafe.Pointer
	err := withNativeGraphDependencies(deps, func(dependencies *unsafe.Pointer) error {
		status := C.gocudaGraphAddMemset(&node, graph, dependencies, C.size_t(len(deps)), dst, C.uint(value), C.uint(elementSize), C.size_t(count))
		return nativeDriverError("cuGraphAddMemsetNode", status)
	})
	return node, err
}

func instantiateNativeGraph(graph unsafe.Pointer) (unsafe.Pointer, error) {
	var exec unsafe.Pointer
	status := C.gocudaGraphInstantiate(&exec, graph)
	return exec, nativeDriverError("cuGraphInstantiate", status)
}

func launchNativeGraph(exec, stream unsafe.Pointer) error {
	return nativeDriverError("cuGraphLaunch", C.gocudaGraphLaunch(exec, stream))
}

func destroyNativeGraphExec(exec unsafe.Pointer) error {
	return nativeDriverError("cuGraphExecDestroy", C.gocudaGraphExecDestroy(exec))
}

func destroyNativeGraph(graph unsafe.Pointer) error {
	return nativeDriverError("cuGraphDestroy", C.gocudaGraphDestroy(graph))
}

func nativeGraphHasDependency(node, dependency unsafe.Pointer) (bool, error) {
	var count C.size_t
	status := C.gocudaGraphDependencies(node, nil, &count)
	if err := nativeDriverError("cuGraphNodeGetDependencies", status); err != nil {
		return false, err
	}
	if count == 0 {
		return false, nil
	}
	if uint64(count) > uint64(^uint(0)>>1)/uint64(unsafe.Sizeof(uintptr(0))) {
		return false, fmt.Errorf("graph dependency count exceeds supported range")
	}
	data := C.malloc(count * C.size_t(unsafe.Sizeof(uintptr(0))))
	if data == nil {
		return false, fmt.Errorf("allocating graph dependency list failed")
	}
	defer C.free(data)
	status = C.gocudaGraphDependencies(node, (*unsafe.Pointer)(data), &count)
	if err := nativeDriverError("cuGraphNodeGetDependencies", status); err != nil {
		return false, err
	}
	for _, current := range unsafe.Slice((*unsafe.Pointer)(data), int(count)) {
		if current == dependency {
			return true, nil
		}
	}
	return false, nil
}
