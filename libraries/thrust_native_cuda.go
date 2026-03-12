//go:build cuda

package libraries

import (
	"encoding/binary"
	"math"
	"slices"
	"sort"
	"time"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

func thrustNativeAvailable() bool {
	return true
}

func createNativeThrustContext() (*ThrustContext, error) {
	return &ThrustContext{handle: uintptr(time.Now().UnixNano()), native: true}, nil
}

func executeNativeThrustCopy(src, dst *memory.Memory, n int) error {
	if src == nil || dst == nil || n <= 0 {
		return errThrustUnsupported
	}
	copyBytes := int64(n * 4)
	if src.Size() < copyBytes || dst.Size() < copyBytes {
		return errThrustUnsupported
	}
	return memory.CopyDeviceToDevice(dst, src)
}

func executeNativeThrustFill(data *memory.Memory, n int, value float32) error {
	if data == nil || n <= 0 {
		return errThrustUnsupported
	}
	fillBytes := int64(n * 4)
	if data.Size() < fillBytes {
		return errThrustUnsupported
	}
	if value == 0 {
		return internal.CudaMemsetOnDevice(data.Ptr(), 0, fillBytes, data.GetDeviceID())
	}
	buffer := make([]byte, n*4)
	bits := math.Float32bits(value)
	for index := 0; index < n; index++ {
		binary.LittleEndian.PutUint32(buffer[index*4:], bits)
	}
	return memory.CopyHostToDevice(data, buffer)
}

func executeNativeThrustGenerate(data *memory.Memory, n int, generator string) error {
	if data == nil || n <= 0 {
		return errThrustUnsupported
	}
	switch generator {
	case "", "zeros":
		return executeNativeThrustFill(data, n, 0)
	case "ones":
		return executeNativeThrustFill(data, n, 1)
	case "sequence", "index":
		values := make([]byte, n*4)
		for index := 0; index < n; index++ {
			binary.LittleEndian.PutUint32(values[index*4:], math.Float32bits(float32(index)))
		}
		return memory.CopyHostToDevice(data, values)
	default:
		return errThrustUnsupported
	}
}

func executeNativeThrustSort(data *memory.Memory, n int) error {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return errThrustUnsupported
	}
	slices.Sort(values)
	return thrustWriteValues(data, values)
}

func executeNativeThrustSortByKey(keys, values *memory.Memory, n int) error {
	keyValues, err := thrustReadValues(keys, n)
	if err != nil {
		return errThrustUnsupported
	}
	associatedValues, err := thrustReadValues(values, n)
	if err != nil {
		return errThrustUnsupported
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

func executeNativeThrustReduce(data *memory.Memory, n int, initValue float32) (float32, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, errThrustUnsupported
	}
	total := initValue
	for _, value := range values {
		total += value
	}
	return total, nil
}

func executeNativeThrustTransform(input, output *memory.Memory, n int, operation string) error {
	inputValues, err := thrustReadValues(input, n)
	if err != nil {
		return errThrustUnsupported
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

func executeNativeThrustTransformBinary(input1, input2, output *memory.Memory, n int, operation string) error {
	leftValues, err := thrustReadValues(input1, n)
	if err != nil {
		return errThrustUnsupported
	}
	rightValues, err := thrustReadValues(input2, n)
	if err != nil {
		return errThrustUnsupported
	}
	outputValues := make([]float32, n)
	for index := range outputValues {
		result, transformErr := applyThrustBinaryOperation(operation, leftValues[index], rightValues[index])
		if transformErr != nil {
			return transformErr
		}
		outputValues[index] = result
	}
	return thrustWriteValues(output, outputValues)
}

func executeNativeThrustScan(input, output *memory.Memory, n int) error {
	inputValues, err := thrustReadValues(input, n)
	if err != nil {
		return errThrustUnsupported
	}
	outputValues := make([]float32, n)
	running := float32(0)
	for index, value := range inputValues {
		running += value
		outputValues[index] = running
	}
	return thrustWriteValues(output, outputValues)
}

func executeNativeThrustExclusiveScan(input, output *memory.Memory, n int, initValue float32) error {
	inputValues, err := thrustReadValues(input, n)
	if err != nil {
		return errThrustUnsupported
	}
	outputValues := make([]float32, n)
	running := initValue
	for index, value := range inputValues {
		outputValues[index] = running
		running += value
	}
	return thrustWriteValues(output, outputValues)
}

func executeNativeThrustPartition(data *memory.Memory, n int, predicate string) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, errThrustUnsupported
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

func executeNativeThrustCopyIf(src, dst *memory.Memory, n int, predicate string) (int, error) {
	values, err := thrustReadValues(src, n)
	if err != nil {
		return 0, errThrustUnsupported
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

func executeNativeThrustSetUnion(input1, input2, output *memory.Memory, n1, n2 int) (int, error) {
	leftValues, err := thrustReadValues(input1, n1)
	if err != nil {
		return 0, errThrustUnsupported
	}
	rightValues, err := thrustReadValues(input2, n2)
	if err != nil {
		return 0, errThrustUnsupported
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

func executeNativeThrustSetIntersection(input1, input2, output *memory.Memory, n1, n2 int) (int, error) {
	leftValues, err := thrustReadValues(input1, n1)
	if err != nil {
		return 0, errThrustUnsupported
	}
	rightValues, err := thrustReadValues(input2, n2)
	if err != nil {
		return 0, errThrustUnsupported
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

func executeNativeThrustFind(data *memory.Memory, n int, value float32) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return -1, errThrustUnsupported
	}
	for index, candidate := range values {
		if candidate == value {
			return index, nil
		}
	}
	return -1, nil
}

func executeNativeThrustCount(data *memory.Memory, n int, value float32) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, errThrustUnsupported
	}
	total := 0
	for _, candidate := range values {
		if candidate == value {
			total++
		}
	}
	return total, nil
}

func executeNativeThrustUnique(data *memory.Memory, n int) (int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, errThrustUnsupported
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
	if err := thrustWriteValues(data, values); err != nil {
		return 0, err
	}
	return writeIndex, nil
}

func executeNativeThrustMerge(input1, input2, output *memory.Memory, n1, n2 int) error {
	leftValues, err := thrustReadValues(input1, n1)
	if err != nil {
		return errThrustUnsupported
	}
	rightValues, err := thrustReadValues(input2, n2)
	if err != nil {
		return errThrustUnsupported
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

func executeNativeThrustMinElement(data *memory.Memory, n int) (float32, int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, -1, errThrustUnsupported
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

func executeNativeThrustMaxElement(data *memory.Memory, n int) (float32, int, error) {
	values, err := thrustReadValues(data, n)
	if err != nil {
		return 0, -1, errThrustUnsupported
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

func destroyNativeThrustContext(ctx *ThrustContext) error {
	if ctx == nil {
		return nil
	}
	ctx.handle = 0
	ctx.native = false
	return nil
}
