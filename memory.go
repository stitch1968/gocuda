package cuda

// Re-exports from memory package for backward compatibility

import (
	"github.com/stitch1968/gocuda/memory"
)

// Type aliases for backward compatibility
type Type = memory.Type
type Pool = memory.Pool

// Memory type constants
const (
	TypeDevice   = memory.TypeDevice
	TypeHost     = memory.TypeHost
	TypePinned   = memory.TypePinned
	TypeUnified  = memory.TypeUnified
	TypeShared   = memory.TypeShared
	TypeConstant = memory.TypeConstant
	TypeTexture  = memory.TypeTexture
)

// Memory allocation functions (excluding Alloc which conflicts with simple_api.go)
var (
	Malloc                       = memory.Alloc
	MallocWithTypeAndStream      = memory.AllocWithTypeAndStream
	AllocWithStream              = memory.AllocWithStream
	AllocWithTypeAndStream       = memory.AllocWithTypeAndStream
	CopyHostToDevice             = memory.CopyHostToDevice
	CopyDeviceToHost             = memory.CopyDeviceToHost
	CopyDeviceToDevice           = memory.CopyDeviceToDevice
	CopyHostToDeviceWithStream   = memory.CopyHostToDeviceWithStream
	CopyDeviceToHostWithStream   = memory.CopyDeviceToHostWithStream
	CopyDeviceToDeviceWithStream = memory.CopyDeviceToDeviceWithStream
	NewPool                      = memory.NewPool
	GetInfo                      = memory.GetInfo
	GetMemoryInfo                = memory.GetInfo // Alias for backward compatibility
)
