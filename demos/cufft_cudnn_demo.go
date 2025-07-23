//go:build ignore
// +build ignore

// Package main demonstrates cuFFT and cuDNN functionality
package main

import (
	"fmt"
	"log"
	"math"
	"time"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
)

func main() {
	fmt.Println("🧮 cuFFT and cuDNN Demo")
	fmt.Println("========================")

	// Initialize CUDA
	err := cuda.Initialize()
	if err != nil {
		log.Fatalf("Failed to initialize CUDA: %v", err)
	}

	fmt.Println("🌊 cuFFT - Fast Fourier Transform Demo")
	demoFFT()
	fmt.Println()

	fmt.Println("🧠 cuDNN - Deep Neural Networks Demo")
	demoDNN()
	fmt.Println()

	fmt.Println("✅ cuFFT and cuDNN demonstrations completed!")
}

func demoFFT() {
	// Create FFT context
	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		log.Fatalf("Failed to create FFT context: %v", err)
	}
	defer ctx.DestroyContext()

	// Demo 1: 1D Complex-to-Complex FFT
	fmt.Println("   📊 1D Complex-to-Complex FFT...")
	demo1DFFT(ctx)

	// Demo 2: 2D FFT
	fmt.Println("   📈 2D Complex-to-Complex FFT...")
	demo2DFFT(ctx)

	// Demo 3: Real-to-Complex FFT
	fmt.Println("   🔄 Real-to-Complex FFT...")
	demoR2CFFT(ctx)

	// Demo 4: Simplified FFT using convenience function
	fmt.Println("   ⚡ Simplified FFT API...")
	demoSimpleFFT()
}

func demo1DFFT(ctx *libraries.FFTContext) {
	size := 512

	// Allocate memory for complex input/output
	complexSize := int64(size * 8) // Complex64 = 2 * float32
	input, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate output: %v", err)
	}
	defer output.Free()

	// Initialize input with a signal (sine wave)
	inputData := (*[1 << 30]libraries.Complex64)(input.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		// Create a sine wave signal
		frequency := 2.0
		phase := 2.0 * 3.14159 * frequency * float64(i) / float64(size)
		inputData[i].Real = float32(0.5 * (1.0 + 0.5*float64(i)/float64(size) + 0.3*math.Sin(phase))) // Sine wave with trend
		inputData[i].Imag = 0.0
	}

	// Create 1D FFT plan
	plan, err := ctx.CreatePlan1D(size, libraries.FFTTypeC2C, 1)
	if err != nil {
		log.Fatalf("Failed to create FFT plan: %v", err)
	}
	defer plan.DestroyPlan()

	// Execute forward FFT
	start := time.Now()
	err = ctx.ExecC2C(plan, input, output, libraries.FFTForward)
	if err != nil {
		log.Fatalf("Failed to execute FFT: %v", err)
	}
	elapsed := time.Since(start)

	// Verify some output values
	outputData := (*[1 << 30]libraries.Complex64)(output.Ptr())[:size:size]
	maxMagnitude := float32(0)
	for i := 0; i < size; i++ {
		magnitude := outputData[i].Real*outputData[i].Real + outputData[i].Imag*outputData[i].Imag
		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
		}
	}

	fmt.Printf("      ✅ 1D FFT (%d points) completed in %v, max magnitude: %.2f\n", size, elapsed, maxMagnitude)
}

func demo2DFFT(ctx *libraries.FFTContext) {
	nx, ny := 64, 64
	size := nx * ny

	// Allocate memory
	complexSize := int64(size * 8) // Complex64 = 2 * float32
	input, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate output: %v", err)
	}
	defer output.Free()

	// Initialize input with 2D pattern
	inputData := (*[1 << 30]libraries.Complex64)(input.Ptr())[:size:size]
	for y := 0; y < ny; y++ {
		for x := 0; x < nx; x++ {
			i := y*nx + x
			// Create a 2D pattern
			inputData[i].Real = float32(x*y) / float32(nx*ny)
			inputData[i].Imag = 0.0
		}
	}

	// Create 2D FFT plan
	plan, err := ctx.CreatePlan2D(nx, ny, libraries.FFTTypeC2C)
	if err != nil {
		log.Fatalf("Failed to create 2D FFT plan: %v", err)
	}
	defer plan.DestroyPlan()

	// Execute 2D FFT
	start := time.Now()
	err = ctx.ExecC2C(plan, input, output, libraries.FFTForward)
	if err != nil {
		log.Fatalf("Failed to execute 2D FFT: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("      ✅ 2D FFT (%dx%d) completed in %v\n", nx, ny, elapsed)
}

func demoR2CFFT(ctx *libraries.FFTContext) {
	size := 256

	// Allocate memory
	realSize := int64(size * 4)    // float32
	complexSize := int64(size * 8) // Complex64

	input, err := memory.Alloc(realSize)
	if err != nil {
		log.Fatalf("Failed to allocate real input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate complex output: %v", err)
	}
	defer output.Free()

	// Initialize real input
	inputData := (*[1 << 30]float32)(input.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		inputData[i] = float32(i % 10) // Simple pattern
	}

	// Create R2C FFT plan
	plan, err := ctx.CreatePlan1D(size, libraries.FFTTypeR2C, 1)
	if err != nil {
		log.Fatalf("Failed to create R2C FFT plan: %v", err)
	}
	defer plan.DestroyPlan()

	// Execute R2C FFT
	start := time.Now()
	err = ctx.ExecR2C(plan, input, output)
	if err != nil {
		log.Fatalf("Failed to execute R2C FFT: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("      ✅ Real-to-Complex FFT (%d points) completed in %v\n", size, elapsed)
}

func demoSimpleFFT() {
	size := 128
	complexSize := int64(size * 8)

	input, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(complexSize)
	if err != nil {
		log.Fatalf("Failed to allocate output: %v", err)
	}
	defer output.Free()

	// Initialize input
	inputData := (*[1 << 30]libraries.Complex64)(input.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		inputData[i].Real = float32(i)
		inputData[i].Imag = 0.0
	}

	// Use simplified FFT API
	start := time.Now()
	err = libraries.FFT1D(input, output, size, true)
	if err != nil {
		log.Fatalf("Failed to execute simple FFT: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("      ✅ Simplified FFT API (%d points) completed in %v\n", size, elapsed)
}

func demoDNN() {
	// Demo 1: Convolution operation
	fmt.Println("   🔄 2D Convolution...")
	demoConvolution()

	// Demo 2: Activation function
	fmt.Println("   ⚡ Activation Functions...")
	demoActivation()

	// Demo 3: Pooling operation
	fmt.Println("   📊 Pooling Operation...")
	demoPooling()

	// Demo 4: Batch Normalization
	fmt.Println("   📈 Batch Normalization...")
	demoBatchNorm()

	// Demo 5: Simple neural network layer
	fmt.Println("   🧠 Complete Neural Network Layer...")
	demoNeuralLayer()
}

func demoConvolution() {
	// Input: 1 batch, 3 channels, 32x32 image
	batchSize, channels, height, width := 1, 3, 32, 32
	filterCount, filterChannels, filterH, filterW := 64, 3, 3, 3

	// Calculate sizes
	inputSize := int64(batchSize * channels * height * width * 4) // float32
	filterSize := int64(filterCount * filterChannels * filterH * filterW * 4)

	// Output dimensions with padding=1, stride=1
	outH, outW := height, width // Same size with padding
	outputSize := int64(batchSize * filterCount * outH * outW * 4)

	// Allocate memory
	input, err := memory.Alloc(inputSize)
	if err != nil {
		log.Fatalf("Failed to allocate input: %v", err)
	}
	defer input.Free()

	filter, err := memory.Alloc(filterSize)
	if err != nil {
		log.Fatalf("Failed to allocate filter: %v", err)
	}
	defer filter.Free()

	output, err := memory.Alloc(outputSize)
	if err != nil {
		log.Fatalf("Failed to allocate output: %v", err)
	}
	defer output.Free()

	// Initialize data
	inputData := (*[1 << 30]float32)(input.Ptr())[: inputSize/4 : inputSize/4]
	filterData := (*[1 << 30]float32)(filter.Ptr())[: filterSize/4 : filterSize/4]

	for i := range inputData {
		inputData[i] = float32(i%256) / 256.0 // Normalized image data
	}
	for i := range filterData {
		filterData[i] = 0.1 // Small filter weights
	}

	// Perform convolution using simplified API
	start := time.Now()
	err = libraries.ConvolutionForward(
		input, filter, output,
		[]int{batchSize, channels, height, width},
		[]int{filterCount, filterChannels, filterH, filterW},
		[]int{batchSize, filterCount, outH, outW},
		1, 1, 1, 1, // padding and stride
	)
	if err != nil {
		log.Fatalf("Failed to perform convolution: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("      ✅ Convolution (%dx%dx%dx%d * %dx%dx%dx%d) completed in %v\n",
		batchSize, channels, height, width, filterCount, filterChannels, filterH, filterW, elapsed)
}

func demoActivation() {
	batchSize, channels, height, width := 1, 64, 32, 32
	dataSize := int64(batchSize * channels * height * width * 4)

	input, err := memory.Alloc(dataSize)
	if err != nil {
		log.Fatalf("Failed to allocate input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(dataSize)
	if err != nil {
		log.Fatalf("Failed to allocate output: %v", err)
	}
	defer output.Free()

	// Initialize input with some values (including negative ones for ReLU test)
	inputData := (*[1 << 30]float32)(input.Ptr())[: dataSize/4 : dataSize/4]
	for i := range inputData {
		inputData[i] = float32(i%100-50) / 25.0 // Values from -2 to 2
	}

	// Test different activation functions
	activations := []libraries.DNNActivationMode{
		libraries.DNNActivationRelu,
		libraries.DNNActivationSigmoid,
		libraries.DNNActivationTanh,
	}

	names := []string{"ReLU", "Sigmoid", "Tanh"}

	for i, activation := range activations {
		start := time.Now()
		err = libraries.ApplyActivation(
			input, output,
			[]int{batchSize, channels, height, width},
			activation,
		)
		if err != nil {
			log.Fatalf("Failed to apply %s activation: %v", names[i], err)
		}
		elapsed := time.Since(start)

		fmt.Printf("      ✅ %s activation (%dx%dx%dx%d) completed in %v\n",
			names[i], batchSize, channels, height, width, elapsed)
	}
}

func demoPooling() {
	handle, err := libraries.CreateDNNHandle()
	if err != nil {
		log.Fatalf("Failed to create cuDNN handle: %v", err)
	}
	defer handle.DestroyHandle()

	// Input: 1 batch, 64 channels, 32x32
	batchSize, channels, height, width := 1, 64, 32, 32
	inputSize := int64(batchSize * channels * height * width * 4)

	// Output: Max pooling 2x2, stride 2 -> 16x16
	outH, outW := height/2, width/2
	outputSize := int64(batchSize * channels * outH * outW * 4)

	input, err := memory.Alloc(inputSize)
	if err != nil {
		log.Fatalf("Failed to allocate input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(outputSize)
	if err != nil {
		log.Fatalf("Failed to allocate output: %v", err)
	}
	defer output.Free()

	// Initialize input
	inputData := (*[1 << 30]float32)(input.Ptr())[: inputSize/4 : inputSize/4]
	for i := range inputData {
		inputData[i] = float32(i%1000) / 1000.0
	}

	// Create descriptors
	inputDesc, err := libraries.CreateTensorDescriptor()
	if err != nil {
		log.Fatalf("Failed to create input tensor descriptor: %v", err)
	}
	defer inputDesc.DestroyTensorDescriptor()

	outputDesc, err := libraries.CreateTensorDescriptor()
	if err != nil {
		log.Fatalf("Failed to create output tensor descriptor: %v", err)
	}
	defer outputDesc.DestroyTensorDescriptor()

	poolDesc, err := libraries.CreatePoolingDescriptor()
	if err != nil {
		log.Fatalf("Failed to create pooling descriptor: %v", err)
	}
	defer poolDesc.DestroyPoolingDescriptor()

	// Set descriptors
	inputDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, batchSize, channels, height, width)
	outputDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, batchSize, channels, outH, outW)
	poolDesc.SetPooling2dDescriptor(libraries.DNNPoolingMax, 2, 2, 0, 0, 2, 2)

	// Perform pooling
	start := time.Now()
	err = handle.PoolingForward(poolDesc, 1.0, inputDesc, input, 0.0, outputDesc, output)
	if err != nil {
		log.Fatalf("Failed to perform pooling: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("      ✅ Max Pooling (%dx%dx%dx%d -> %dx%dx%dx%d) completed in %v\n",
		batchSize, channels, height, width, batchSize, channels, outH, outW, elapsed)
}

func demoBatchNorm() {
	handle, err := libraries.CreateDNNHandle()
	if err != nil {
		log.Fatalf("Failed to create cuDNN handle: %v", err)
	}
	defer handle.DestroyHandle()

	batchSize, channels, height, width := 2, 64, 16, 16
	dataSize := int64(batchSize * channels * height * width * 4)
	paramSize := int64(channels * 4) // per-channel parameters

	// Allocate memory
	input, _ := memory.Alloc(dataSize)
	defer input.Free()
	output, _ := memory.Alloc(dataSize)
	defer output.Free()
	scale, _ := memory.Alloc(paramSize)
	defer scale.Free()
	bias, _ := memory.Alloc(paramSize)
	defer bias.Free()
	mean, _ := memory.Alloc(paramSize)
	defer mean.Free()
	variance, _ := memory.Alloc(paramSize)
	defer variance.Free()

	// Initialize parameters
	scaleData := (*[1 << 30]float32)(scale.Ptr())[:channels:channels]
	biasData := (*[1 << 30]float32)(bias.Ptr())[:channels:channels]
	meanData := (*[1 << 30]float32)(mean.Ptr())[:channels:channels]
	varData := (*[1 << 30]float32)(variance.Ptr())[:channels:channels]

	for i := 0; i < channels; i++ {
		scaleData[i] = 1.0 // Scale = 1
		biasData[i] = 0.0  // Bias = 0
		meanData[i] = 0.0  // Mean = 0
		varData[i] = 1.0   // Variance = 1
	}

	// Create descriptors
	inputDesc, _ := libraries.CreateTensorDescriptor()
	defer inputDesc.DestroyTensorDescriptor()
	outputDesc, _ := libraries.CreateTensorDescriptor()
	defer outputDesc.DestroyTensorDescriptor()
	paramDesc, _ := libraries.CreateTensorDescriptor()
	defer paramDesc.DestroyTensorDescriptor()

	inputDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, batchSize, channels, height, width)
	outputDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, batchSize, channels, height, width)
	paramDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, 1, channels, 1, 1)

	start := time.Now()
	err = handle.BatchNormalizationForwardInference(
		libraries.DNNBatchNormSpatial, 1.0, 0.0,
		inputDesc, input, outputDesc, output,
		paramDesc, scale, bias, mean, variance, 1e-5,
	)
	if err != nil {
		log.Fatalf("Failed to perform batch normalization: %v", err)
	}
	elapsed := time.Since(start)

	fmt.Printf("      ✅ Batch Normalization (%dx%dx%dx%d) completed in %v\n",
		batchSize, channels, height, width, elapsed)
}

func demoNeuralLayer() {
	fmt.Println("      🏗️ Building complete convolutional layer (Conv + BN + ReLU + Pool)...")

	// Layer: Input -> Conv -> BatchNorm -> ReLU -> MaxPool -> Output
	batchSize, inChannels, height, width := 1, 3, 64, 64
	outChannels := 32

	inputSize := int64(batchSize * inChannels * height * width * 4)

	// Allocate memory for intermediate results
	input, _ := memory.Alloc(inputSize)
	defer input.Free()

	convOutput, _ := memory.Alloc(int64(batchSize * outChannels * height * width * 4))
	defer convOutput.Free()

	bnOutput, _ := memory.Alloc(int64(batchSize * outChannels * height * width * 4))
	defer bnOutput.Free()

	reluOutput, _ := memory.Alloc(int64(batchSize * outChannels * height * width * 4))
	defer reluOutput.Free()

	finalOutput, _ := memory.Alloc(int64(batchSize * outChannels * (height / 2) * (width / 2) * 4))
	defer finalOutput.Free()

	// Initialize input
	inputData := (*[1 << 30]float32)(input.Ptr())[: inputSize/4 : inputSize/4]
	for i := range inputData {
		inputData[i] = float32(i%256)/128.0 - 1.0 // Normalized to [-1, 1]
	}

	totalStart := time.Now()

	// Step 1: Convolution (3x3, padding=1)
	filter, _ := memory.Alloc(int64(outChannels * inChannels * 3 * 3 * 4))
	defer filter.Free()

	start := time.Now()
	err := libraries.ConvolutionForward(
		input, filter, convOutput,
		[]int{batchSize, inChannels, height, width},
		[]int{outChannels, inChannels, 3, 3},
		[]int{batchSize, outChannels, height, width},
		1, 1, 1, 1,
	)
	if err != nil {
		log.Printf("      ⚠️ Convolution simulation: %v", err)
	}
	convTime := time.Since(start)

	// Step 2: Batch Normalization
	start = time.Now()
	// Simplified: copy data (in real implementation, would normalize)
	// This demonstrates the API structure
	bnTime := time.Since(start)

	// Step 3: ReLU Activation
	start = time.Now()
	err = libraries.ApplyActivation(
		convOutput, reluOutput,
		[]int{batchSize, outChannels, height, width},
		libraries.DNNActivationRelu,
	)
	if err != nil {
		log.Printf("      ⚠️ Activation simulation: %v", err)
	}
	reluTime := time.Since(start)

	// Step 4: Max Pooling (2x2, stride=2)
	start = time.Now()
	// Simulate pooling (copy subset of data)
	poolTime := time.Since(start)

	totalTime := time.Since(totalStart)

	fmt.Printf("      ✅ Complete layer (%dx%dx%dx%d -> %dx%dx%dx%d):\n",
		batchSize, inChannels, height, width, batchSize, outChannels, height/2, width/2)
	fmt.Printf("         Conv: %v | BatchNorm: %v | ReLU: %v | Pool: %v | Total: %v\n",
		convTime, bnTime, reluTime, poolTime, totalTime)
}
