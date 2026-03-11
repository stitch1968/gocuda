//go:build cuda
// +build cuda

package main

/*
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcudart -lcuda -Wl,--no-as-needed

#include <cuda_runtime.h>
#include <stdio.h>

int test_cuda_init() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("CUDA Device Count: %d, Error: %d\n", count, (int)err);
    return count;
}
*/
import "C"

import "fmt"

func main() {
	fmt.Println("=== Minimal CGO + CUDA Test ===")
	count := C.test_cuda_init()
	fmt.Printf("Found %d CUDA devices\n", int(count))
}
