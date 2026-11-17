package libraries

import (
	"errors"
	"fmt"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

type dssPendingOperation struct {
	cleanup    func() error
	retentions []*memory.Retention
}

// Synchronize waits for the native default stream and retries pending cleanup.
func (handle *DSSHandle) Synchronize() error {
	if handle == nil {
		return fmt.Errorf("nil cuDSS handle")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.destroyed {
		return fmt.Errorf("cuDSS handle has been destroyed")
	}
	return handle.synchronizeLocked()
}

func (handle *DSSHandle) synchronizeLocked() error {
	if handle.native {
		if err := internal.CudaDefaultStreamSynchronizeOnDevice(handle.deviceID); err != nil {
			return err
		}
	}
	for _, matrix := range handle.pendingMatrices {
		matrix.release(handle)
	}
	handle.pendingMatrices = nil
	var errs []error
	remaining := handle.pendingOperations[:0]
	for _, pending := range handle.pendingOperations {
		err := internal.RunOnDevice(handle.deviceID, pending.cleanup)
		if err != nil {
			errs = append(errs, err)
			remaining = append(remaining, pending)
			continue
		}
		releaseLibraryBuffers(pending.retentions)
	}
	handle.pendingOperations = remaining
	return errors.Join(errs...)
}

func (handle *DSSHandle) hasPendingLocked() bool {
	return len(handle.pendingMatrices) != 0 || len(handle.pendingOperations) != 0
}

func (handle *DSSHandle) runNativePhaseLocked(run func() error) (bool, error) {
	invoked, synchronized := false, false
	err := internal.RunOnDevice(handle.deviceID, func() error {
		invoked = true
		runErr := run()
		syncErr := internal.CudaDefaultStreamSynchronizeOnDevice(handle.deviceID)
		synchronized = syncErr == nil
		return errors.Join(runErr, syncErr)
	})
	return invoked && !synchronized, err
}
