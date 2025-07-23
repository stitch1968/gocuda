package cuda

import (
	"fmt"
)

// Enhanced error handling - addressing the "error handling could be more user-friendly" issue

// EnhancedError provides rich error context with suggestions
type EnhancedError struct {
	Operation   string
	Cause       string
	Suggestion  string
	Details     map[string]interface{}
	Recoverable bool
}

func (e *EnhancedError) Error() string {
	if e.Suggestion != "" {
		return fmt.Sprintf("%s failed: %s. Suggestion: %s",
			e.Operation, e.Cause, e.Suggestion)
	}
	return fmt.Sprintf("%s failed: %s", e.Operation, e.Cause)
}

// IsRecoverable returns true if the error can be handled gracefully
func (e *EnhancedError) IsRecoverable() bool {
	return e.Recoverable
}

// GetDetails returns additional error context
func (e *EnhancedError) GetDetails() map[string]interface{} {
	return e.Details
}

// Common enhanced error creators

// NewMemoryError creates a memory-related error with helpful suggestions
func NewMemoryError(operation, cause string, requestedSize, availableSize int64) *EnhancedError {
	suggestion := ""
	recoverable := true

	if requestedSize > availableSize {
		suggestion = fmt.Sprintf("Reduce allocation size from %d to max %d bytes, or use memory pools for efficiency",
			requestedSize, availableSize)
	} else {
		suggestion = "Try freeing unused allocations or restart the application"
	}

	return &EnhancedError{
		Operation:   operation,
		Cause:       cause,
		Suggestion:  suggestion,
		Recoverable: recoverable,
		Details: map[string]interface{}{
			"requested_bytes": requestedSize,
			"available_bytes": availableSize,
			"usage_percent":   float64(requestedSize) / float64(availableSize) * 100,
		},
	}
}

// NewKernelError creates a kernel execution error with helpful suggestions
func NewKernelError(operation, cause string, gridDim, blockDim Dim3) *EnhancedError {
	suggestion := "Check kernel parameters and reduce grid/block dimensions if necessary"

	totalThreads := gridDim.X * gridDim.Y * gridDim.Z * blockDim.X * blockDim.Y * blockDim.Z
	if totalThreads > 1024*1024 { // Arbitrary large number
		suggestion = fmt.Sprintf("Reduce total threads from %d - try smaller grid/block dimensions", totalThreads)
	}

	return &EnhancedError{
		Operation:   operation,
		Cause:       cause,
		Suggestion:  suggestion,
		Recoverable: true,
		Details: map[string]interface{}{
			"grid_dim":      fmt.Sprintf("%dx%dx%d", gridDim.X, gridDim.Y, gridDim.Z),
			"block_dim":     fmt.Sprintf("%dx%dx%d", blockDim.X, blockDim.Y, blockDim.Z),
			"total_threads": totalThreads,
		},
	}
}

// NewDeviceError creates a device-related error with helpful suggestions
func NewDeviceError(operation, cause string, deviceID int) *EnhancedError {
	suggestion := ""
	if !IsCudaAvailable() {
		suggestion = "Install CUDA drivers and toolkit, or use CPU simulation mode"
	} else if deviceID >= GetCudaDeviceCount() {
		suggestion = fmt.Sprintf("Use device ID 0-%d (you requested %d)", GetCudaDeviceCount()-1, deviceID)
	} else {
		suggestion = "Check device status and restart application if necessary"
	}

	return &EnhancedError{
		Operation:   operation,
		Cause:       cause,
		Suggestion:  suggestion,
		Recoverable: true,
		Details: map[string]interface{}{
			"requested_device":  deviceID,
			"available_devices": GetCudaDeviceCount(),
			"cuda_available":    IsCudaAvailable(),
		},
	}
}

// WrapError wraps an existing error with enhanced context
func WrapError(operation string, originalError error, suggestion string) *EnhancedError {
	return &EnhancedError{
		Operation:   operation,
		Cause:       originalError.Error(),
		Suggestion:  suggestion,
		Recoverable: true,
		Details: map[string]interface{}{
			"original_error": originalError.Error(),
		},
	}
}

// FallbackToCPU provides a fallback mechanism for recoverable errors
func FallbackToCPU() error {
	ForceFallbackToSimulation(true)
	return nil
}

// Enhanced error checking helpers

// CheckMemoryAllocation validates memory allocation parameters
func CheckMemoryAllocation(size int64) error {
	if size <= 0 {
		return &EnhancedError{
			Operation:   "Memory Allocation Validation",
			Cause:       fmt.Sprintf("Invalid size: %d", size),
			Suggestion:  "Use a positive size value",
			Recoverable: false,
			Details: map[string]interface{}{
				"requested_size": size,
			},
		}
	}

	free, _ := GetMemoryInfo()
	if size > free {
		return NewMemoryError("Memory Allocation Validation",
			fmt.Sprintf("Insufficient memory (requested %d, available %d)", size, free),
			size, free)
	}

	return nil
}

// CheckKernelParameters validates kernel launch parameters
func CheckKernelParameters(gridDim, blockDim Dim3) error {
	if blockDim.X*blockDim.Y*blockDim.Z > 1024 {
		return NewKernelError("Kernel Parameter Validation",
			fmt.Sprintf("Block size too large: %dx%dx%d > 1024", blockDim.X, blockDim.Y, blockDim.Z),
			gridDim, blockDim)
	}

	if gridDim.X*gridDim.Y*gridDim.Z > 65535*65535 {
		return NewKernelError("Kernel Parameter Validation",
			fmt.Sprintf("Grid size too large: %dx%dx%d", gridDim.X, gridDim.Y, gridDim.Z),
			gridDim, blockDim)
	}

	return nil
}

// Helper functions for better error messages

// FormatMemorySize formats memory size in human-readable format
func FormatMemorySize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// GetErrorSuggestion provides context-aware error suggestions
func GetErrorSuggestion(err error) string {
	if enhancedErr, ok := err.(*EnhancedError); ok {
		return enhancedErr.Suggestion
	}

	// Fallback suggestions for standard errors
	errStr := err.Error()
	switch {
	case contains(errStr, "memory"):
		return "Try reducing memory usage or freeing unused allocations"
	case contains(errStr, "device"):
		return "Check CUDA installation and device availability"
	case contains(errStr, "kernel"):
		return "Verify kernel parameters and reduce complexity if needed"
	default:
		return "Check CUDA installation and try CPU simulation mode"
	}
}

// Helper function to check if string contains substring (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr || len(s) > len(substr) &&
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
				indexOf(s, substr) >= 0))
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
