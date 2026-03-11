// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements Thrust-style algorithms for GPU computing
package libraries

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/stitch1968/gocuda/memory"
)

// Thrust - Parallel Algorithms Library

// ThrustContext manages Thrust operations
type ThrustContext struct {
	handle uintptr
}

// ExecutionPolicy defines how algorithms should execute
type ExecutionPolicy int

const (
	PolicyDevice ExecutionPolicy = iota // Execute on GPU
	PolicyHost                          // Execute on CPU
	PolicyCuda                          // CUDA-specific optimizations
)

// CreateThrustContext creates a new Thrust context
func CreateThrustContext() (*ThrustContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	return &ThrustContext{
		handle: uintptr(time.Now().UnixNano()),
	}, nil
}

// Sort sorts elements in ascending order
func (ctx *ThrustContext) Sort(data *memory.Memory, n int, policy ExecutionPolicy) error {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return err
	}
	sort.Slice(values, func(i, j int) bool {
		return values[i] < values[j]
	})
	return thrustWriteValues(data, values)
}

// SortByKey sorts key-value pairs by keys
func (ctx *ThrustContext) SortByKey(keys, values *memory.Memory, n int, policy ExecutionPolicy) error {
	keyValues, err := thrustReadValues(keys, n)
	if err != nil {
		return err
	}
	associatedValues, err := thrustReadValues(values, n)
	if err != nil {
		return err
	}
	type pair struct {
		key   float32
		value float32
	}
	pairs := make([]pair, n)
	for index := range pairs {
		pairs[index] = pair{key: keyValues[index], value: associatedValues[index]}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].key < pairs[j].key
	})
	for index, item := range pairs {
		keyValues[index] = item.key
		associatedValues[index] = item.value
	}
	if err := thrustWriteValues(keys, keyValues); err != nil {
		return err
	}
	return thrustWriteValues(values, associatedValues)
}

// Reduce performs parallel reduction
func (ctx *ThrustContext) Reduce(data *memory.Memory, n int, initValue float32, policy ExecutionPolicy) (float32, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, err
	}
	total := initValue
	for _, value := range values {
		total += value
	}
	return total, nil
}

// Transform applies unary operation to each element
func (ctx *ThrustContext) Transform(input, output *memory.Memory, n int, operation string, policy ExecutionPolicy) error {
	inputValues, err := thrustReadValues(input, n)
	if err != nil {
		return err
	}
	outputValues := make([]float32, n)
	for index, value := range inputValues {
		result, transformErr := applyThrustUnaryOperation(operation, value, index)
		if transformErr != nil {
			return transformErr
		}
		outputValues[index] = result
	}
	return thrustWriteValues(output, outputValues)
}

// TransformBinary applies binary operation to pairs of elements
func (ctx *ThrustContext) TransformBinary(input1, input2, output *memory.Memory, n int, operation string, policy ExecutionPolicy) error {
	leftValues, err := thrustReadValues(input1, n)
	if err != nil {
		return err
	}
	rightValues, err := thrustReadValues(input2, n)
	if err != nil {
		return err
	}
	outputValues := make([]float32, n)
	for index := 0; index < n; index++ {
		result, transformErr := applyThrustBinaryOperation(operation, leftValues[index], rightValues[index])
		if transformErr != nil {
			return transformErr
		}
		outputValues[index] = result
	}
	return thrustWriteValues(output, outputValues)
}

// Scan performs inclusive prefix sum
func (ctx *ThrustContext) Scan(input, output *memory.Memory, n int, policy ExecutionPolicy) error {
	inputValues, err := thrustReadValues(input, n)
	if err != nil {
		return err
	}
	outputValues := make([]float32, n)
	running := float32(0)
	for index, value := range inputValues {
		running += value
		outputValues[index] = running
	}
	return thrustWriteValues(output, outputValues)
}

// ExclusiveScan performs exclusive prefix sum
func (ctx *ThrustContext) ExclusiveScan(input, output *memory.Memory, n int, initValue float32, policy ExecutionPolicy) error {
	inputValues, err := thrustReadValues(input, n)
	if err != nil {
		return err
	}
	outputValues := make([]float32, n)
	running := initValue
	for index, value := range inputValues {
		outputValues[index] = running
		running += value
	}
	return thrustWriteValues(output, outputValues)
}

// Find locates first occurrence of value
func (ctx *ThrustContext) Find(data *memory.Memory, n int, value float32, policy ExecutionPolicy) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return -1, err
	}
	for index, candidate := range values {
		if candidate == value {
			return index, nil
		}
	}
	return -1, nil
}

// Count counts occurrences of value
func (ctx *ThrustContext) Count(data *memory.Memory, n int, value float32, policy ExecutionPolicy) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, err
	}
	total := 0
	for _, candidate := range values {
		if candidate == value {
			total++
		}
	}
	return total, nil
}

// Unique removes consecutive duplicate elements
func (ctx *ThrustContext) Unique(data *memory.Memory, n int, policy ExecutionPolicy) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, err
	}
	if len(values) == 0 {
		return 0, nil
	}
	writeIndex := 1
	for readIndex := 1; readIndex < len(values); readIndex++ {
		if values[readIndex] != values[writeIndex-1] {
			values[writeIndex] = values[readIndex]
			writeIndex++
		}
	}
	return writeIndex, thrustWriteValues(data, values)
}

// Partition partitions elements based on predicate
func (ctx *ThrustContext) Partition(data *memory.Memory, n int, predicate string, policy ExecutionPolicy) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, err
	}
	matched := make([]float32, 0, len(values))
	remainder := make([]float32, 0, len(values))
	for index, value := range values {
		keep, predicateErr := evaluateThrustPredicate(predicate, value, index)
		if predicateErr != nil {
			return 0, predicateErr
		}
		if keep {
			matched = append(matched, value)
			continue
		}
		remainder = append(remainder, value)
	}
	partitionIndex := len(matched)
	values = append(matched, remainder...)
	if err := thrustWriteValues(data, values); err != nil {
		return 0, err
	}
	return partitionIndex, nil
}

// Copy copies elements from source to destination
func (ctx *ThrustContext) Copy(src, dst *memory.Memory, n int, policy ExecutionPolicy) error {
	values, err := thrustReadValues(src, n)
	if err != nil {
		return err
	}
	return thrustWriteValues(dst, values)
}

// CopyIf copies elements that satisfy predicate
func (ctx *ThrustContext) CopyIf(src, dst *memory.Memory, n int, predicate string, policy ExecutionPolicy) (int, error) {
	values, err := thrustReadValues(src, n)
	if err != nil {
		return 0, err
	}
	selected := make([]float32, 0, len(values))
	for index, value := range values {
		keep, predicateErr := evaluateThrustPredicate(predicate, value, index)
		if predicateErr != nil {
			return 0, predicateErr
		}
		if keep {
			selected = append(selected, value)
		}
	}
	if len(selected) > 0 {
		if err := thrustWriteValues(dst, selected); err != nil {
			return 0, err
		}
	}
	return len(selected), nil
}

// Fill fills memory with specified value
func (ctx *ThrustContext) Fill(data *memory.Memory, n int, value float32, policy ExecutionPolicy) error {
	values := make([]float32, n)
	for index := range values {
		values[index] = value
	}
	return thrustWriteValues(data, values)
}

// Generate fills memory using generator function
func (ctx *ThrustContext) Generate(data *memory.Memory, n int, generator string, policy ExecutionPolicy) error {
	values := make([]float32, n)
	for index := range values {
		generated, err := evaluateThrustGenerator(generator, index)
		if err != nil {
			return err
		}
		values[index] = generated
	}
	return thrustWriteValues(data, values)
}

// Merge merges two sorted sequences
func (ctx *ThrustContext) Merge(input1, input2, output *memory.Memory, n1, n2 int, policy ExecutionPolicy) error {
	leftValues, err := thrustReadValues(input1, n1)
	if err != nil {
		return err
	}
	rightValues, err := thrustReadValues(input2, n2)
	if err != nil {
		return err
	}
	merged := make([]float32, 0, n1+n2)
	i, j := 0, 0
	for i < len(leftValues) && j < len(rightValues) {
		if leftValues[i] <= rightValues[j] {
			merged = append(merged, leftValues[i])
			i++
			continue
		}
		merged = append(merged, rightValues[j])
		j++
	}
	merged = append(merged, leftValues[i:]...)
	merged = append(merged, rightValues[j:]...)
	return thrustWriteValues(output, merged)
}

// SetUnion computes union of two sorted sequences
func (ctx *ThrustContext) SetUnion(input1, input2, output *memory.Memory, n1, n2 int, policy ExecutionPolicy) (int, error) {
	leftValues, err := thrustReadValues(input1, n1)
	if err != nil {
		return 0, err
	}
	rightValues, err := thrustReadValues(input2, n2)
	if err != nil {
		return 0, err
	}
	result := make([]float32, 0, n1+n2)
	i, j := 0, 0
	for i < len(leftValues) || j < len(rightValues) {
		var next float32
		switch {
		case j >= len(rightValues) || (i < len(leftValues) && leftValues[i] < rightValues[j]):
			next = leftValues[i]
			i++
		case i >= len(leftValues) || rightValues[j] < leftValues[i]:
			next = rightValues[j]
			j++
		default:
			next = leftValues[i]
			i++
			j++
		}
		if len(result) == 0 || result[len(result)-1] != next {
			result = append(result, next)
		}
	}
	if err := thrustWriteValues(output, result); err != nil {
		return 0, err
	}
	return len(result), nil
}

// SetIntersection computes intersection of two sorted sequences
func (ctx *ThrustContext) SetIntersection(input1, input2, output *memory.Memory, n1, n2 int, policy ExecutionPolicy) (int, error) {
	leftValues, err := thrustReadValues(input1, n1)
	if err != nil {
		return 0, err
	}
	rightValues, err := thrustReadValues(input2, n2)
	if err != nil {
		return 0, err
	}
	result := make([]float32, 0, minInt(n1, n2))
	i, j := 0, 0
	for i < len(leftValues) && j < len(rightValues) {
		if leftValues[i] == rightValues[j] {
			if len(result) == 0 || result[len(result)-1] != leftValues[i] {
				result = append(result, leftValues[i])
			}
			i++
			j++
			continue
		}
		if leftValues[i] < rightValues[j] {
			i++
			continue
		}
		j++
	}
	if err := thrustWriteValues(output, result); err != nil {
		return 0, err
	}
	return len(result), nil
}

// MinElement finds minimum element
func (ctx *ThrustContext) MinElement(data *memory.Memory, n int, policy ExecutionPolicy) (float32, int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, -1, err
	}
	minValue, minIndex := values[0], 0
	for index := 1; index < len(values); index++ {
		if values[index] < minValue {
			minValue = values[index]
			minIndex = index
		}
	}
	return minValue, minIndex, nil
}

// MaxElement finds maximum element
func (ctx *ThrustContext) MaxElement(data *memory.Memory, n int, policy ExecutionPolicy) (float32, int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, -1, err
	}
	maxValue, maxIndex := values[0], 0
	for index := 1; index < len(values); index++ {
		if values[index] > maxValue {
			maxValue = values[index]
			maxIndex = index
		}
	}
	return maxValue, maxIndex, nil
}

// DestroyContext cleans up Thrust context
func (ctx *ThrustContext) DestroyContext() error {
	ctx.handle = 0
	return nil
}

func thrustReadValues(mem *memory.Memory, n int) ([]float32, error) {
	if mem == nil {
		return nil, fmt.Errorf("memory cannot be nil")
	}
	if n <= 0 {
		return nil, fmt.Errorf("number of elements must be positive")
	}
	return readMathFloat32Memory(mem, n)
}

func thrustWriteValues(mem *memory.Memory, values []float32) error {
	if mem == nil {
		return fmt.Errorf("memory cannot be nil")
	}
	return writeMathFloat32Memory(mem, values)
}

func applyThrustUnaryOperation(operation string, value float32, index int) (float32, error) {
	switch operation {
	case "", "identity":
		return value, nil
	case "negate":
		return -value, nil
	case "square":
		return value * value, nil
	case "sqrt":
		return float32(math.Sqrt(float64(value))), nil
	case "sin":
		return float32(math.Sin(float64(value))), nil
	case "cos":
		return float32(math.Cos(float64(value))), nil
	case "exp":
		return float32(math.Exp(float64(value))), nil
	case "log":
		return float32(math.Log(float64(value))), nil
	case "abs":
		return float32(math.Abs(float64(value))), nil
	case "index":
		return float32(index), nil
	default:
		return 0, fmt.Errorf("unsupported thrust unary operation: %s", operation)
	}
}

func applyThrustBinaryOperation(operation string, left, right float32) (float32, error) {
	switch operation {
	case "", "add":
		return left + right, nil
	case "sub":
		return left - right, nil
	case "mul":
		return left * right, nil
	case "div":
		return left / right, nil
	case "max":
		if left > right {
			return left, nil
		}
		return right, nil
	case "min":
		if left < right {
			return left, nil
		}
		return right, nil
	default:
		return 0, fmt.Errorf("unsupported thrust binary operation: %s", operation)
	}
}

func evaluateThrustPredicate(predicate string, value float32, index int) (bool, error) {
	switch predicate {
	case "", "nonzero":
		return value != 0, nil
	case "positive":
		return value > 0, nil
	case "negative":
		return value < 0, nil
	case "even":
		return int(value)%2 == 0, nil
	case "odd":
		return int(value)%2 != 0, nil
	case "index_even":
		return index%2 == 0, nil
	default:
		return false, fmt.Errorf("unsupported thrust predicate: %s", predicate)
	}
}

func evaluateThrustGenerator(generator string, index int) (float32, error) {
	switch generator {
	case "", "zeros":
		return 0, nil
	case "ones":
		return 1, nil
	case "sequence", "index":
		return float32(index), nil
	default:
		return 0, fmt.Errorf("unsupported thrust generator: %s", generator)
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
