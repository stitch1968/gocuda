//go:build cuda

package tests

import (
	"encoding/binary"
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

const vectorAddPTX = `
.version 7.0
.target sm_70
.address_size 64
.visible .entry vector_add(
	.param .u64 a,
	.param .u64 b,
	.param .u64 c,
	.param .u32 n
) {
	.reg .pred %p;
	.reg .b32 %r<5>;
	.reg .b64 %rd<8>;
	.reg .f32 %f<4>;
	ld.param.u64 %rd1, [a];
	ld.param.u64 %rd2, [b];
	ld.param.u64 %rd3, [c];
	ld.param.u32 %r1, [n];
	mov.u32 %r2, %tid.x;
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mad.lo.u32 %r2, %r3, %r4, %r2;
	setp.ge.u32 %p, %r2, %r1;
	@%p bra DONE;
	mul.wide.u32 %rd4, %r2, 4;
	add.u64 %rd5, %rd1, %rd4;
	add.u64 %rd6, %rd2, %rd4;
	add.u64 %rd7, %rd3, %rd4;
	ld.global.f32 %f1, [%rd5];
	ld.global.f32 %f2, [%rd6];
	add.f32 %f3, %f1, %f2;
	st.global.f32 [%rd7], %f3;
DONE:
	ret;
}
.visible .entry no_args() {
	ret;
}
.visible .entry scalar_args(
	.param .u64 output,
	.param .s32 int32_value,
	.param .s64 int64_value,
	.param .f32 float32_value,
	.param .f64 float64_value
) {
	.reg .b32 %r;
	.reg .b64 %rd<3>;
	.reg .f32 %f;
	.reg .f64 %d;
	ld.param.u64 %rd1, [output];
	ld.param.s32 %r, [int32_value];
	ld.param.s64 %rd2, [int64_value];
	ld.param.f32 %f, [float32_value];
	ld.param.f64 %d, [float64_value];
	st.global.u32 [%rd1], %r;
	st.global.u64 [%rd1+8], %rd2;
	st.global.f32 [%rd1+16], %f;
	st.global.f64 [%rd1+24], %d;
	ret;
}
`

func TestNativeKernelLaunch(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	module, err := kernels.LoadPTX(0, vectorAddPTX)
	if err != nil {
		t.Fatal(err)
	}
	defer module.Close()
	if _, err := module.Function("missing"); err == nil {
		t.Fatal("expected unknown kernel name to fail")
	}
	function, err := module.Function("vector_add")
	if err != nil {
		t.Fatal(err)
	}
	noArgs, err := module.Function("no_args")
	if err != nil {
		t.Fatal(err)
	}
	hostA, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostA.Free()
	hostB, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostB.Free()
	hostResult, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostResult.Free()
	a, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	result, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer result.Free()
	defer func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	}()
	if err := noArgs.Launch(stream, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0, kernels.Uint32Arg(1)); err == nil {
		t.Fatal("expected extra argument for parameterless kernel to fail")
	}
	if err := noArgs.Launch(stream, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0); err != nil {
		t.Fatal(err)
	}

	aValues, err := memory.View[float32](hostA, 8)
	if err != nil {
		t.Fatal(err)
	}
	bValues, err := memory.View[float32](hostB, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 8 {
		aValues[i] = float32(i)
		bValues[i] = float32(2 * i)
	}
	if err := stream.CopyAsync(a, hostA, 32); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(b, hostB, 32); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	grid, block := kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 32, Y: 1, Z: 1}
	if err := function.Launch(stream, grid, block, 0); err == nil {
		t.Fatal("expected missing kernel arguments to fail")
	}
	if err := function.Launch(stream, grid, block, 0,
		kernels.BufferArg(a, 0), kernels.BufferArg(b, 0), kernels.BufferArg(result, 0), kernels.Uint64Arg(7)); err == nil {
		t.Fatal("expected wrong kernel argument size to fail")
	}
	if err := function.Launch(stream, grid, block, 0,
		kernels.BufferArg(nil, 0), kernels.BufferArg(b, 0), kernels.BufferArg(result, 0), kernels.Uint32Arg(7)); err == nil {
		t.Fatal("expected nil device buffer to fail")
	}
	if err := function.Launch(stream, grid, block, 0,
		kernels.BufferArg(hostA, 0), kernels.BufferArg(b, 0), kernels.BufferArg(result, 0), kernels.Uint32Arg(7)); err == nil {
		t.Fatal("expected host buffer to fail")
	}
	if err := function.Launch(stream, kernels.Dim3{}, block, 0); err == nil {
		t.Fatal("expected invalid launch dimensions to fail")
	}
	if err := function.Launch(stream, grid, block, 0,
		kernels.BufferArg(a, 4), kernels.BufferArg(b, 4), kernels.BufferArg(result, 0), kernels.Uint32Arg(7)); err != nil {
		t.Fatal(err)
	}
	if err := a.Free(); err == nil {
		t.Fatal("expected kernel input to remain retained")
	}
	if err := stream.CopyAsync(hostResult, result, 28); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got, err := memory.View[float32](hostResult, 7)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range got {
		if want := float32(3 * (i + 1)); value != want {
			t.Fatalf("result[%d] = %v, want %v", i, value, want)
		}
	}
	scalars, err := module.Function("scalar_args")
	if err != nil {
		t.Fatal(err)
	}
	if err := scalars.Launch(stream, grid, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0,
		kernels.BufferArg(result, 0), kernels.Int32Arg(-123), kernels.Int64Arg(-456),
		kernels.Float32Arg(1.5), kernels.Float64Arg(-3.25)); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(hostResult, result, 32); err != nil {
		t.Fatal(err)
	}
	if err := module.Close(); err != nil {
		t.Fatal(err)
	}
	bytes, err := memory.View[byte](hostResult, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got := int32(binary.NativeEndian.Uint32(bytes[:4])); got != -123 {
		t.Fatalf("int32 argument = %d", got)
	}
	if got := int64(binary.NativeEndian.Uint64(bytes[8:16])); got != -456 {
		t.Fatalf("int64 argument = %d", got)
	}
	if got := math.Float32frombits(binary.NativeEndian.Uint32(bytes[16:20])); got != 1.5 {
		t.Fatalf("float32 argument = %v", got)
	}
	if got := math.Float64frombits(binary.NativeEndian.Uint64(bytes[24:32])); got != -3.25 {
		t.Fatalf("float64 argument = %v", got)
	}
	if err := function.Launch(stream, grid, block, 0,
		kernels.BufferArg(a, 0), kernels.BufferArg(b, 0), kernels.BufferArg(result, 0), kernels.Uint32Arg(7)); err == nil {
		t.Fatal("expected launch from a closed module to fail")
	}
}
