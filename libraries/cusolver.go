// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuSOLVER functionality for linear algebra solvers
package libraries

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// ErrSolverSimulationUnsupported is retained for compatibility with earlier
// versions that did not implement simulation decompositions.
var ErrSolverSimulationUnsupported = errors.New("solver operation is not supported in simulation mode")

func validateSolverMatrix(matrix *memory.Memory, rows, cols int) error {
	if rows <= 0 || cols <= 0 || int64(rows) > math.MaxInt32 || int64(cols) > math.MaxInt32 {
		return fmt.Errorf("invalid matrix dimensions: %d x %d", rows, cols)
	}
	if matrix == nil || matrix.Ptr() == nil || uint64(rows)*uint64(cols) > uint64(matrix.Size())/4 {
		return fmt.Errorf("matrix is nil, freed, or too small for %d x %d float32 elements", rows, cols)
	}
	return nil
}

func (ctx *SolverContext) lock() error {
	if ctx == nil {
		return fmt.Errorf("nil solver context")
	}
	ctx.mu.Lock()
	if ctx.destroyed {
		ctx.mu.Unlock()
		return fmt.Errorf("solver context has been destroyed")
	}
	return nil
}

func (ctx *SolverContext) validateMatrix(matrix *memory.Memory, rows, cols int) error {
	if ctx == nil {
		return fmt.Errorf("nil solver context")
	}
	if err := validateSolverMatrix(matrix, rows, cols); err != nil {
		return err
	}
	if ctx.native {
		switch matrix.GetType() {
		case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
		default:
			return fmt.Errorf("native solver matrix must use device or managed memory")
		}
		if matrix.GetDeviceID() != ctx.deviceID {
			return fmt.Errorf("solver matrix device %d does not match context device %d", matrix.GetDeviceID(), ctx.deviceID)
		}
	}
	return nil
}

func (ctx *SolverContext) lockMatrix(matrix *memory.Memory, rows, cols int) (*memory.Retention, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	retention, err := matrix.Retain()
	if err != nil {
		ctx.mu.Unlock()
		return nil, err
	}
	if err := ctx.validateMatrix(matrix, rows, cols); err != nil {
		retention.Release()
		ctx.mu.Unlock()
		return nil, err
	}
	return retention, nil
}

func runSolverOnDevice[T any](ctx *SolverContext, run func() (T, error), cleanup func(T) error) (T, error) {
	var result T
	err := internal.RunOnDevice(ctx.deviceID, func() error {
		var runErr error
		result, runErr = run()
		return runErr
	})
	if err != nil {
		if cleanupErr := cleanup(result); cleanupErr != nil {
			err = errors.Join(err, cleanupErr)
		}
		var zero T
		return zero, err
	}
	return result, nil
}

// cuSOLVER - Linear Algebra Solvers Library

// SolverContext manages cuSOLVER operations
type SolverContext struct {
	handle    unsafe.Pointer // Simulated or native handle
	native    bool
	deviceID  int
	mu        sync.Mutex
	destroyed bool
}

// QRInfo contains information about QR decomposition
type QRInfo struct {
	tau       *memory.Memory // Householder scalars
	workspace *memory.Memory // Workspace memory
	info      int            // Information flag
}

// SVDInfo contains information about SVD decomposition
type SVDInfo struct {
	s         *memory.Memory // Singular values
	u         *memory.Memory // Left singular vectors
	vt        *memory.Memory // Right singular vectors (transposed)
	workspace *memory.Memory // Workspace memory
	info      int            // Information flag
}

// LUInfo contains information about LU decomposition
type LUInfo struct {
	ipiv      *memory.Memory // Pivot indices
	workspace *memory.Memory // Workspace memory
	info      int            // Information flag
}

// Tau returns the Householder scalars owned by the QR result.
// The allocation remains valid until Destroy and must not be freed separately.
func (qr *QRInfo) Tau() *memory.Memory {
	if qr == nil {
		return nil
	}
	return qr.tau
}

// SingularValues returns the allocation owned by the SVD result. It remains
// valid until Destroy and must not be freed separately.
func (svd *SVDInfo) SingularValues() *memory.Memory {
	if svd == nil {
		return nil
	}
	return svd.s
}

// U returns the result-owned column-major left singular vectors, or nil if not
// requested. It remains valid until Destroy and must not be freed separately.
func (svd *SVDInfo) U() *memory.Memory {
	if svd == nil {
		return nil
	}
	return svd.u
}

// VT returns the result-owned column-major transpose of the right singular
// vectors, or nil if not requested. It remains valid until Destroy and must
// not be freed separately.
func (svd *SVDInfo) VT() *memory.Memory {
	if svd == nil {
		return nil
	}
	return svd.vt
}

// Pivots returns the result-owned one-based int32 pivot allocation. It remains
// valid until Destroy and must not be freed separately.
func (lu *LUInfo) Pivots() *memory.Memory {
	if lu == nil {
		return nil
	}
	return lu.ipiv
}

// CreateSolverContext creates a new cuSOLVER context
func CreateSolverContext() (*SolverContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() {
		return createNativeSolverContext()
	}

	return &SolverContext{
		handle: unsafe.Pointer(uintptr(time.Now().UnixNano())),
	}, nil
}

// CreateNativeSolverContextOnDevice creates a device-bound cuSOLVER context.
// It returns an error rather than falling back to simulation.
func CreateNativeSolverContextOnDevice(deviceID int) (*SolverContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if !cuda.ShouldUseCuda() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= cuda.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var ctx *SolverContext
	err := internal.RunOnDevice(deviceID, func() error {
		var createErr error
		ctx, createErr = createNativeSolverContext()
		return createErr
	})
	if err != nil {
		if ctx != nil {
			return nil, errors.Join(err, ctx.DestroyContext())
		}
		return nil, err
	}
	return ctx, nil
}

// Backend reports whether this context uses native CUDA or Go simulation.
func (ctx *SolverContext) Backend() LibraryImplementationMode {
	if ctx == nil {
		return ""
	}
	if ctx.native {
		return ImplementationModeNativeCUDA
	}
	return ImplementationModeValidatedGo
}

// QRFactorization performs QR decomposition of column-major matrix A in place.
func (ctx *SolverContext) QRFactorization(A *memory.Memory, m, n int) (*QRInfo, error) {
	retention, err := ctx.lockMatrix(A, m, n)
	if err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()

	if ctx.native {
		return runSolverOnDevice(ctx, func() (*QRInfo, error) {
			return nativeQRFactorization(ctx, A, m, n)
		}, func(info *QRInfo) error {
			if info != nil {
				return info.Destroy()
			}
			return nil
		})
	}

	return simulatedQRFactorization(A, m, n)
}

// SVDDecomposition performs singular value decomposition of column-major A.
func (ctx *SolverContext) SVDDecomposition(A *memory.Memory, m, n int, computeUV bool) (*SVDInfo, error) {
	retention, err := ctx.lockMatrix(A, m, n)
	if err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()

	if computeUV {
		if _, err := solverCheckedSquareSize(m); err != nil {
			return nil, err
		}
		if _, err := solverCheckedSquareSize(n); err != nil {
			return nil, err
		}
	}
	if ctx.native {
		return runSolverOnDevice(ctx, func() (*SVDInfo, error) {
			return nativeSVDDecomposition(ctx, A, m, n, computeUV)
		}, func(info *SVDInfo) error {
			if info != nil {
				return info.Destroy()
			}
			return nil
		})
	}

	return simulatedSVDDecomposition(A, m, n, computeUV)
}

// LUFactorization performs in-place LU decomposition with partial pivoting on
// a column-major matrix. Pivot indices are one-based, matching cuSOLVER.
func (ctx *SolverContext) LUFactorization(A *memory.Memory, m, n int) (*LUInfo, error) {
	retention, err := ctx.lockMatrix(A, m, n)
	if err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()

	if ctx.native {
		return runSolverOnDevice(ctx, func() (*LUInfo, error) {
			return nativeLUFactorization(ctx, A, m, n)
		}, func(info *LUInfo) error {
			if info != nil {
				return info.Destroy()
			}
			return nil
		})
	}

	return simulatedLUFactorization(A, m, n)
}

// SolveLinearSystem solves Ax = b using LU factorization. It overwrites the
// column-major matrix A and returns a newly allocated solution.
func (ctx *SolverContext) SolveLinearSystem(A *memory.Memory, b *memory.Memory, n int) (*memory.Memory, error) {
	retention, err := ctx.lockMatrix(A, n, n)
	if err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()
	rhsRetention, err := b.Retain()
	if err != nil {
		return nil, err
	}
	defer rhsRetention.Release()
	if err := validateSolverMatrix(b, n, 1); err != nil {
		return nil, fmt.Errorf("right-hand side: %w", err)
	}

	if ctx.native {
		return runSolverOnDevice(ctx, func() (*memory.Memory, error) {
			return nativeSolveLinearSystem(ctx, A, b, n)
		}, func(result *memory.Memory) error {
			return result.Free()
		})
	}

	return simulatedSolveLinearSystem(A, b, n)
}

// Eigenvalues computes eigenvalues and optionally eigenvectors
func (ctx *SolverContext) Eigenvalues(A *memory.Memory, n int, computeVectors bool) (*memory.Memory, *memory.Memory, error) {
	retention, err := ctx.lockMatrix(A, n, n)
	if err != nil {
		return nil, nil, err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()

	if ctx.native {
		type eigenResult struct {
			values, vectors *memory.Memory
		}
		result, err := runSolverOnDevice(ctx, func() (eigenResult, error) {
			values, vectors, err := nativeEigenvalues(ctx, A, n, computeVectors)
			return eigenResult{values, vectors}, err
		}, func(result eigenResult) error {
			return errors.Join(result.values.Free(), result.vectors.Free())
		})
		return result.values, result.vectors, err
	}

	return simulatedEigenvalues(A, n, computeVectors)
}

// CholeskyFactorization replaces the upper triangle of a column-major positive
// definite matrix with its upper Cholesky factor.
func (ctx *SolverContext) CholeskyFactorization(A *memory.Memory, n int) error {
	retention, err := ctx.lockMatrix(A, n, n)
	if err != nil {
		return err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()

	if ctx.native {
		return internal.RunOnDevice(ctx.deviceID, func() error {
			return nativeCholeskyFactorization(ctx, A, n)
		})
	}

	return simulatedCholeskyFactorization(A, n)
}

// PseudoInverse computes the column-major Moore-Penrose pseudoinverse using SVD.
func (ctx *SolverContext) PseudoInverse(A *memory.Memory, m, n int) (*memory.Memory, error) {
	retention, err := ctx.lockMatrix(A, m, n)
	if err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	defer retention.Release()
	if ctx.native {
		return runSolverOnDevice(ctx, func() (*memory.Memory, error) {
			return nativePseudoInverse(ctx, A, m, n)
		}, func(result *memory.Memory) error {
			return result.Free()
		})
	}

	return simulatedPseudoInverse(A, m, n)
}

// Destroy methods for cleaning up

func (qr *QRInfo) Destroy() error {
	if qr == nil {
		return nil
	}
	return errors.Join(qr.tau.Free(), qr.workspace.Free())
}

func (svd *SVDInfo) Destroy() error {
	if svd == nil {
		return nil
	}
	return errors.Join(svd.s.Free(), svd.u.Free(), svd.vt.Free(), svd.workspace.Free())
}

func (lu *LUInfo) Destroy() error {
	if lu == nil {
		return nil
	}
	return errors.Join(lu.ipiv.Free(), lu.workspace.Free())
}

// DestroyContext cleans up the solver context
func (ctx *SolverContext) DestroyContext() error {
	if ctx == nil {
		return nil
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return nil
	}
	if ctx.native {
		return internal.RunOnDevice(ctx.deviceID, func() error {
			if err := destroyNativeSolverContext(ctx); err != nil {
				return err
			}
			ctx.handle = nil
			ctx.destroyed = true
			return nil
		})
	}
	ctx.handle = nil
	ctx.destroyed = true
	return nil
}
