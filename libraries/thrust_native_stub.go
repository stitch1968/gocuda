//go:build !cuda

package libraries

import "github.com/stitch1968/gocuda/memory"

func thrustNativeAvailable() bool {
	return false
}

func createNativeThrustContext() (*ThrustContext, error) {
	return nil, errThrustUnsupported
}

func executeNativeThrustCopy(src, dst *memory.Memory, n int) error {
	return errThrustUnsupported
}

func executeNativeThrustFill(data *memory.Memory, n int, value float32) error {
	return errThrustUnsupported
}

func executeNativeThrustGenerate(data *memory.Memory, n int, generator string) error {
	return errThrustUnsupported
}

func executeNativeThrustSort(data *memory.Memory, n int) error {
	return errThrustUnsupported
}

func executeNativeThrustSortByKey(keys, values *memory.Memory, n int) error {
	return errThrustUnsupported
}

func executeNativeThrustReduce(data *memory.Memory, n int, initValue float32) (float32, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustTransform(input, output *memory.Memory, n int, operation string) error {
	return errThrustUnsupported
}

func executeNativeThrustTransformBinary(input1, input2, output *memory.Memory, n int, operation string) error {
	return errThrustUnsupported
}

func executeNativeThrustScan(input, output *memory.Memory, n int) error {
	return errThrustUnsupported
}

func executeNativeThrustExclusiveScan(input, output *memory.Memory, n int, initValue float32) error {
	return errThrustUnsupported
}

func executeNativeThrustPartition(data *memory.Memory, n int, predicate string) (int, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustCopyIf(src, dst *memory.Memory, n int, predicate string) (int, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustSetUnion(input1, input2, output *memory.Memory, n1, n2 int) (int, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustSetIntersection(input1, input2, output *memory.Memory, n1, n2 int) (int, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustFind(data *memory.Memory, n int, value float32) (int, error) {
	return -1, errThrustUnsupported
}

func executeNativeThrustCount(data *memory.Memory, n int, value float32) (int, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustUnique(data *memory.Memory, n int) (int, error) {
	return 0, errThrustUnsupported
}

func executeNativeThrustMerge(input1, input2, output *memory.Memory, n1, n2 int) error {
	return errThrustUnsupported
}

func executeNativeThrustMinElement(data *memory.Memory, n int) (float32, int, error) {
	return 0, -1, errThrustUnsupported
}

func executeNativeThrustMaxElement(data *memory.Memory, n int) (float32, int, error) {
	return 0, -1, errThrustUnsupported
}

func destroyNativeThrustContext(ctx *ThrustContext) error {
	return nil
}
