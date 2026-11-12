//go:build cuda

package streams

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcudart
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcudart

#include <cuda_runtime.h>

static cudaError_t gocudaStreamCreate(void **stream, unsigned int flags) {
	return cudaStreamCreateWithFlags((cudaStream_t*)stream, flags);
}
static cudaError_t gocudaEventCreate(void **event, unsigned int flags) {
	return cudaEventCreateWithFlags((cudaEvent_t*)event, flags);
}
static cudaError_t gocudaStreamSynchronize(void *stream) {
	return cudaStreamSynchronize((cudaStream_t)stream);
}
static cudaError_t gocudaStreamDestroy(void *stream) {
	return cudaStreamDestroy((cudaStream_t)stream);
}
static cudaError_t gocudaCopyAsync(void *stream, void *dst, const void *src, size_t size) {
	return cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, (cudaStream_t)stream);
}
static cudaError_t gocudaEventRecord(void *event, void *stream) {
	return cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
}
static cudaError_t gocudaStreamWaitEvent(void *stream, void *event) {
	return cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0);
}
static cudaError_t gocudaEventQuery(void *event) {
	return cudaEventQuery((cudaEvent_t)event);
}
static cudaError_t gocudaEventSynchronize(void *event) {
	return cudaEventSynchronize((cudaEvent_t)event);
}
static cudaError_t gocudaEventElapsedTime(float *ms, void *start, void *end) {
	return cudaEventElapsedTime(ms, (cudaEvent_t)start, (cudaEvent_t)end);
}
static cudaError_t gocudaEventDestroy(void *event) {
	return cudaEventDestroy((cudaEvent_t)event);
}
*/
import "C"

import (
	"fmt"
	"time"
	"unsafe"
)

func nativeCUDAError(operation string, status C.cudaError_t) error {
	if status == C.cudaSuccess {
		return nil
	}
	return fmt.Errorf("%s failed: %s (%d)", operation, C.GoString(C.cudaGetErrorString(status)), int(status))
}

func createNativeStream(flags StreamFlags) (unsafe.Pointer, error) {
	var handle unsafe.Pointer
	status := C.gocudaStreamCreate(&handle, C.uint(flags))
	return handle, nativeCUDAError("cudaStreamCreateWithFlags", status)
}

func synchronizeNativeStream(handle unsafe.Pointer) error {
	return nativeCUDAError("cudaStreamSynchronize", C.gocudaStreamSynchronize(handle))
}

func destroyNativeStream(handle unsafe.Pointer) error {
	return nativeCUDAError("cudaStreamDestroy", C.gocudaStreamDestroy(handle))
}

func copyNativeAsync(handle, dst, src unsafe.Pointer, size int64) error {
	return nativeCUDAError("cudaMemcpyAsync", C.gocudaCopyAsync(handle, dst, src, C.size_t(size)))
}

func createNativeEvent(flags NativeEventFlags) (unsafe.Pointer, error) {
	var handle unsafe.Pointer
	status := C.gocudaEventCreate(&handle, C.uint(flags))
	return handle, nativeCUDAError("cudaEventCreateWithFlags", status)
}

func recordNativeEvent(event, stream unsafe.Pointer) error {
	return nativeCUDAError("cudaEventRecord", C.gocudaEventRecord(event, stream))
}

func waitNativeEvent(stream, event unsafe.Pointer) error {
	return nativeCUDAError("cudaStreamWaitEvent", C.gocudaStreamWaitEvent(stream, event))
}

func queryNativeEvent(event unsafe.Pointer) (bool, error) {
	status := C.gocudaEventQuery(event)
	if status == C.cudaErrorNotReady {
		return false, nil
	}
	return status == C.cudaSuccess, nativeCUDAError("cudaEventQuery", status)
}

func synchronizeNativeEvent(event unsafe.Pointer) error {
	return nativeCUDAError("cudaEventSynchronize", C.gocudaEventSynchronize(event))
}

func elapsedNativeEvents(start, end unsafe.Pointer) (time.Duration, error) {
	var milliseconds C.float
	status := C.gocudaEventElapsedTime(&milliseconds, start, end)
	if err := nativeCUDAError("cudaEventElapsedTime", status); err != nil {
		return 0, err
	}
	return time.Duration(float64(milliseconds) * float64(time.Millisecond)), nil
}

func destroyNativeEvent(event unsafe.Pointer) error {
	return nativeCUDAError("cudaEventDestroy", C.gocudaEventDestroy(event))
}
