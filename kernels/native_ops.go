package kernels

import (
	"errors"
	"fmt"
	"math"

	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

const nativeOpsPTX = `
.version 7.0
.target sm_70
.address_size 64

.visible .entry gocuda_vector_add(
	.param .u64 a, .param .u64 b, .param .u64 output, .param .u32 n
) {
	.reg .pred %p;
	.reg .b32 %r<5>;
	.reg .b64 %rd<8>;
	.reg .f32 %f<4>;
	ld.param.u64 %rd1, [a];
	ld.param.u64 %rd2, [b];
	ld.param.u64 %rd3, [output];
	ld.param.u32 %r1, [n];
	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mad.lo.u32 %r2, %r3, %r4, %r2;
	setp.ge.u32 %p, %r2, %r1;
	@%p bra VADD_DONE;
	mul.wide.u32 %rd4, %r2, 4;
	add.u64 %rd5, %rd1, %rd4;
	add.u64 %rd6, %rd2, %rd4;
	add.u64 %rd7, %rd3, %rd4;
	ld.global.f32 %f1, [%rd5];
	ld.global.f32 %f2, [%rd6];
	add.f32 %f3, %f1, %f2;
	st.global.f32 [%rd7], %f3;
VADD_DONE:
	ret;
}

.visible .entry gocuda_saxpy(
	.param .f32 alpha, .param .u64 x, .param .u64 y, .param .u32 n
) {
	.reg .pred %p;
	.reg .b32 %r<5>;
	.reg .b64 %rd<6>;
	.reg .f32 %f<5>;
	ld.param.f32 %f1, [alpha];
	ld.param.u64 %rd1, [x];
	ld.param.u64 %rd2, [y];
	ld.param.u32 %r1, [n];
	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mad.lo.u32 %r2, %r3, %r4, %r2;
	setp.ge.u32 %p, %r2, %r1;
	@%p bra SAXPY_DONE;
	mul.wide.u32 %rd3, %r2, 4;
	add.u64 %rd4, %rd1, %rd3;
	add.u64 %rd5, %rd2, %rd3;
	ld.global.f32 %f2, [%rd4];
	ld.global.f32 %f3, [%rd5];
	fma.rn.f32 %f4, %f1, %f2, %f3;
	st.global.f32 [%rd5], %f4;
SAXPY_DONE:
	ret;
}

.visible .entry gocuda_matmul(
	.param .u64 a, .param .u64 b, .param .u64 output,
	.param .u32 m, .param .u32 n, .param .u32 k
) {
	.reg .pred %p<4>;
	.reg .b32 %r<16>;
	.reg .b64 %rd<9>;
	.reg .f32 %f<5>;
	ld.param.u64 %rd1, [a];
	ld.param.u64 %rd2, [b];
	ld.param.u64 %rd3, [output];
	ld.param.u32 %r1, [m];
	ld.param.u32 %r2, [n];
	ld.param.u32 %r3, [k];
	mov.u32 %r4, %tid.x;
	mov.u32 %r5, %ctaid.x;
	mov.u32 %r6, %ntid.x;
	mad.lo.u32 %r4, %r5, %r6, %r4;
	mov.u32 %r5, %tid.y;
	mov.u32 %r6, %ctaid.y;
	mov.u32 %r7, %ntid.y;
	mad.lo.u32 %r5, %r6, %r7, %r5;
	setp.ge.u32 %p1, %r5, %r1;
	setp.ge.u32 %p2, %r4, %r2;
	or.pred %p3, %p1, %p2;
	@%p3 bra MATMUL_DONE;
	mov.f32 %f1, 0f00000000;
	mov.u32 %r8, 0;
MATMUL_LOOP:
	setp.ge.u32 %p1, %r8, %r3;
	@%p1 bra MATMUL_STORE;
	mad.lo.u32 %r9, %r5, %r3, %r8;
	mad.lo.u32 %r10, %r8, %r2, %r4;
	mul.wide.u32 %rd4, %r9, 4;
	mul.wide.u32 %rd5, %r10, 4;
	add.u64 %rd6, %rd1, %rd4;
	add.u64 %rd7, %rd2, %rd5;
	ld.global.f32 %f2, [%rd6];
	ld.global.f32 %f3, [%rd7];
	fma.rn.f32 %f1, %f2, %f3, %f1;
	add.u32 %r8, %r8, 1;
	bra MATMUL_LOOP;
MATMUL_STORE:
	mad.lo.u32 %r11, %r5, %r2, %r4;
	mul.wide.u32 %rd4, %r11, 4;
	add.u64 %rd8, %rd3, %rd4;
	st.global.f32 [%rd8], %f1;
MATMUL_DONE:
	ret;
}

.visible .entry gocuda_zero(.param .u64 output) {
	.reg .b64 %rd;
	ld.param.u64 %rd, [output];
	st.global.u32 [%rd], 0;
	ret;
}

.visible .entry gocuda_sum(
	.param .u64 input, .param .u64 output, .param .u32 n
) {
	.reg .pred %p;
	.reg .b32 %r<5>;
	.reg .b64 %rd<5>;
	.reg .f32 %f<3>;
	ld.param.u64 %rd1, [input];
	ld.param.u64 %rd2, [output];
	ld.param.u32 %r1, [n];
	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mad.lo.u32 %r2, %r3, %r4, %r2;
	setp.ge.u32 %p, %r2, %r1;
	@%p bra SUM_DONE;
	mul.wide.u32 %rd3, %r2, 4;
	add.u64 %rd4, %rd1, %rd3;
	ld.global.f32 %f1, [%rd4];
	atom.global.add.f32 %f2, [%rd2], %f1;
SUM_DONE:
	ret;
}

.visible .entry gocuda_conv2d(
	.param .u64 input, .param .u64 filter, .param .u64 output,
	.param .u32 iw, .param .u32 ih, .param .u32 fw, .param .u32 fh,
	.param .u32 ow, .param .u32 oh
) {
	.reg .pred %p<3>;
	.reg .b32 %r<24>;
	.reg .b64 %rd<10>;
	.reg .f32 %f<4>;
	ld.param.u64 %rd1, [input];
	ld.param.u64 %rd2, [filter];
	ld.param.u64 %rd3, [output];
	ld.param.u32 %r1, [iw];
	ld.param.u32 %r2, [ih];
	ld.param.u32 %r3, [fw];
	ld.param.u32 %r4, [fh];
	ld.param.u32 %r5, [ow];
	ld.param.u32 %r6, [oh];
	mov.u32 %r7, %tid.x;
	mov.u32 %r8, %ctaid.x;
	mov.u32 %r9, %ntid.x;
	mad.lo.u32 %r7, %r8, %r9, %r7;
	mul.lo.u32 %r8, %r5, %r6;
	setp.ge.u32 %p1, %r7, %r8;
	@%p1 bra CONV_DONE;
	rem.u32 %r9, %r7, %r5;
	div.u32 %r10, %r7, %r5;
	mov.f32 %f1, 0f00000000;
	mov.u32 %r11, 0;
CONV_Y:
	setp.ge.u32 %p1, %r11, %r4;
	@%p1 bra CONV_STORE;
	add.u32 %r13, %r10, %r11;
	setp.ge.u32 %p1, %r13, %r2;
	@%p1 bra CONV_NEXT_Y;
	mov.u32 %r12, 0;
CONV_X:
	setp.ge.u32 %p1, %r12, %r3;
	@%p1 bra CONV_NEXT_Y;
	add.u32 %r14, %r9, %r12;
	setp.ge.u32 %p2, %r14, %r1;
	@%p2 bra CONV_NEXT_X;
	mad.lo.u32 %r15, %r13, %r1, %r14;
	mad.lo.u32 %r16, %r11, %r3, %r12;
	mul.wide.u32 %rd4, %r15, 4;
	mul.wide.u32 %rd5, %r16, 4;
	add.u64 %rd6, %rd1, %rd4;
	add.u64 %rd7, %rd2, %rd5;
	ld.global.f32 %f2, [%rd6];
	ld.global.f32 %f3, [%rd7];
	fma.rn.f32 %f1, %f2, %f3, %f1;
CONV_NEXT_X:
	add.u32 %r12, %r12, 1;
	bra CONV_X;
CONV_NEXT_Y:
	add.u32 %r11, %r11, 1;
	bra CONV_Y;
CONV_STORE:
	mul.wide.u32 %rd8, %r7, 4;
	add.u64 %rd9, %rd3, %rd8;
	st.global.f32 [%rd9], %f1;
CONV_DONE:
	ret;
}
`

// NativeOps provides GPU implementations of selected float32 operations.
// The PTX requires a GPU with compute capability 7.0 or newer.
type NativeOps struct {
	module         *NativeModule
	vectorAdd      *NativeFunction
	saxpy          *NativeFunction
	matrixMultiply *NativeFunction
	zero           *NativeFunction
	sum            *NativeFunction
	convolution    *NativeFunction
}

// NewNativeOps loads the built-in GPU kernels on deviceID.
func NewNativeOps(deviceID int) (*NativeOps, error) {
	module, err := LoadPTX(deviceID, nativeOpsPTX)
	if err != nil {
		return nil, err
	}
	ops := &NativeOps{module: module}
	for _, entry := range []struct {
		name string
		dest **NativeFunction
	}{
		{"gocuda_vector_add", &ops.vectorAdd},
		{"gocuda_saxpy", &ops.saxpy},
		{"gocuda_matmul", &ops.matrixMultiply},
		{"gocuda_zero", &ops.zero},
		{"gocuda_sum", &ops.sum},
		{"gocuda_conv2d", &ops.convolution},
	} {
		*entry.dest, err = module.Function(entry.name)
		if err != nil {
			return nil, errors.Join(err, module.Close())
		}
	}
	return ops, nil
}

func (ops *NativeOps) validate(stream *streams.NativeStream, count uint64, buffers ...*memory.Memory) error {
	if ops == nil || ops.module == nil || stream == nil {
		return fmt.Errorf("native operations and stream are required")
	}
	if stream.DeviceID() != ops.module.deviceID {
		return fmt.Errorf("stream device does not match native operations device")
	}
	if count == 0 || count > math.MaxUint32 {
		return fmt.Errorf("element count must be between 1 and %d", uint64(math.MaxUint32))
	}
	for _, buffer := range buffers {
		if buffer == nil || buffer.Ptr() == nil || uint64(buffer.Size())/4 < count {
			return fmt.Errorf("buffer is nil, freed, or too small for %d float32 elements", count)
		}
		switch buffer.GetType() {
		case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
		default:
			return fmt.Errorf("native operation requires device or managed memory")
		}
		if buffer.GetDeviceID() != ops.module.deviceID {
			return fmt.Errorf("buffer device does not match native operations device")
		}
	}
	return nil
}

// VectorAdd enqueues output[i] = a[i] + b[i] on the GPU.
func (ops *NativeOps) VectorAdd(stream *streams.NativeStream, a, b, output *memory.Memory, n int) error {
	if n <= 0 {
		return fmt.Errorf("vector length must be positive")
	}
	if err := ops.validate(stream, uint64(n), a, b, output); err != nil {
		return err
	}
	grid := Dim3{X: int((uint64(n) + 255) / 256), Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	return ops.vectorAdd.Launch(stream, grid, block, 0, BufferArg(a, 0), BufferArg(b, 0), BufferArg(output, 0), Uint32Arg(uint32(n)))
}

// SAXPY enqueues y[i] = alpha*x[i] + y[i] on the GPU.
func (ops *NativeOps) SAXPY(stream *streams.NativeStream, alpha float32, x, y *memory.Memory, n int) error {
	if n <= 0 {
		return fmt.Errorf("vector length must be positive")
	}
	if err := ops.validate(stream, uint64(n), x, y); err != nil {
		return err
	}
	grid := Dim3{X: int((uint64(n) + 255) / 256), Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	return ops.saxpy.Launch(stream, grid, block, 0, Float32Arg(alpha), BufferArg(x, 0), BufferArg(y, 0), Uint32Arg(uint32(n)))
}

// MatrixMultiply enqueues row-major output = a*b for m-by-k and k-by-n inputs.
// This baseline kernel is not an optimized replacement for cuBLAS GEMM.
func (ops *NativeOps) MatrixMultiply(stream *streams.NativeStream, a, b, output *memory.Memory, m, n, k int) error {
	if m <= 0 || n <= 0 || k <= 0 || uint64(m) > math.MaxUint32 || uint64(n) > math.MaxUint32 || uint64(k) > math.MaxUint32 {
		return fmt.Errorf("matrix dimensions must be positive 32-bit values")
	}
	if err := ops.validate(stream, uint64(m)*uint64(k), a); err != nil {
		return err
	}
	if err := ops.validate(stream, uint64(k)*uint64(n), b); err != nil {
		return err
	}
	if err := ops.validate(stream, uint64(m)*uint64(n), output); err != nil {
		return err
	}
	if output.Ptr() == a.Ptr() || output.Ptr() == b.Ptr() {
		return fmt.Errorf("matrix output cannot alias an input")
	}
	grid := Dim3{X: int((uint64(n) + 15) / 16), Y: int((uint64(m) + 15) / 16), Z: 1}
	block := Dim3{X: 16, Y: 16, Z: 1}
	return ops.matrixMultiply.Launch(stream, grid, block, 0,
		BufferArg(a, 0), BufferArg(b, 0), BufferArg(output, 0), Uint32Arg(uint32(m)), Uint32Arg(uint32(n)), Uint32Arg(uint32(k)))
}

// Sum enqueues a float32 reduction into output[0]. Atomic accumulation order
// is nondeterministic, so rounding can differ from a serial sum.
func (ops *NativeOps) Sum(stream *streams.NativeStream, input, output *memory.Memory, n int) error {
	if n <= 0 {
		return fmt.Errorf("vector length must be positive")
	}
	if err := ops.validate(stream, uint64(n), input); err != nil {
		return err
	}
	if err := ops.validate(stream, 1, output); err != nil {
		return err
	}
	if input.Ptr() == output.Ptr() {
		return fmt.Errorf("reduction output cannot alias input")
	}
	one := Dim3{X: 1, Y: 1, Z: 1}
	if err := ops.zero.Launch(stream, one, one, 0, BufferArg(output, 0)); err != nil {
		return err
	}
	grid := Dim3{X: int((uint64(n) + 255) / 256), Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	return ops.sum.Launch(stream, grid, block, 0, BufferArg(input, 0), BufferArg(output, 0), Uint32Arg(uint32(n)))
}

// Convolution2D enqueues valid-position cross-correlation with zero padding
// outside the right and bottom input boundaries.
func (ops *NativeOps) Convolution2D(stream *streams.NativeStream, input, filter, output *memory.Memory, inputWidth, inputHeight, filterWidth, filterHeight, outputWidth, outputHeight int) error {
	for _, dim := range []int{inputWidth, inputHeight, filterWidth, filterHeight, outputWidth, outputHeight} {
		if dim <= 0 || uint64(dim) > math.MaxUint32 {
			return fmt.Errorf("convolution dimensions must be positive 32-bit values")
		}
	}
	if uint64(outputWidth)+uint64(filterWidth)-2 > math.MaxUint32 || uint64(outputHeight)+uint64(filterHeight)-2 > math.MaxUint32 {
		return fmt.Errorf("convolution coordinates exceed the supported range")
	}
	if err := ops.validate(stream, uint64(inputWidth)*uint64(inputHeight), input); err != nil {
		return err
	}
	if err := ops.validate(stream, uint64(filterWidth)*uint64(filterHeight), filter); err != nil {
		return err
	}
	count := uint64(outputWidth) * uint64(outputHeight)
	if err := ops.validate(stream, count, output); err != nil {
		return err
	}
	if output.Ptr() == input.Ptr() || output.Ptr() == filter.Ptr() {
		return fmt.Errorf("convolution output cannot alias an input")
	}
	grid := Dim3{X: int((count + 255) / 256), Y: 1, Z: 1}
	block := Dim3{X: 256, Y: 1, Z: 1}
	return ops.convolution.Launch(stream, grid, block, 0,
		BufferArg(input, 0), BufferArg(filter, 0), BufferArg(output, 0),
		Uint32Arg(uint32(inputWidth)), Uint32Arg(uint32(inputHeight)),
		Uint32Arg(uint32(filterWidth)), Uint32Arg(uint32(filterHeight)),
		Uint32Arg(uint32(outputWidth)), Uint32Arg(uint32(outputHeight)))
}

// Close synchronizes outstanding operations and unloads the GPU kernels.
func (ops *NativeOps) Close() error {
	if ops == nil || ops.module == nil {
		return nil
	}
	return ops.module.Close()
}
