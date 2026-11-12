package tests

import (
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSynchronousTransfersConcurrentWithFree(t *testing.T) {
	for range 25 {
		allocations := make([]*memory.Memory, 4)
		for i := range allocations {
			mem, err := memory.Alloc(256)
			if err != nil {
				t.Fatal(err)
			}
			allocations[i] = mem
		}
		start := make(chan struct{})
		var workers sync.WaitGroup
		for i := range 3 {
			workers.Add(1)
			go func(operation int) {
				defer workers.Done()
				<-start
				host := make([]byte, 256)
				for range 50 {
					switch operation {
					case 0:
						_ = memory.CopyHostToDevice(allocations[0], host)
					case 1:
						_ = memory.CopyDeviceToHost(host, allocations[1])
					case 2:
						_ = memory.CopyDeviceToDevice(allocations[3], allocations[2])
					}
				}
			}(i)
		}
		close(start)
		for _, mem := range allocations {
			_ = mem.Free()
		}
		workers.Wait()
		for _, mem := range allocations {
			if err := mem.Free(); err != nil {
				t.Fatal(err)
			}
		}
	}
}
