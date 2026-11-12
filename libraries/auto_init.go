package libraries

import (
	"sync"

	cuda "github.com/stitch1968/gocuda"
)

var (
	libraryInitOnce sync.Once
	libraryInitErr  error
)

func ensureCudaReady() error {
	libraryInitOnce.Do(func() {
		libraryInitErr = cuda.Initialize()
	})

	return libraryInitErr
}
