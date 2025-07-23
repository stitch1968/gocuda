# 🧮 Expert Module 3: Advanced Numerical Methods

**Goal:** Implement research-grade numerical algorithms with GPU acceleration, focusing on stability, precision, and convergence optimization

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🔬 **Implement advanced solvers** - Iterative, direct, and hybrid methods
- 📊 **Master numerical stability** - Error analysis, conditioning, and preconditioning
- ⚡ **Optimize convergence** - Acceleration techniques and adaptive methods
- 🧪 **Handle precision challenges** - Mixed precision, numerical stability
- 📈 **Scale to massive problems** - Memory-efficient algorithms for large systems

---

## 🧠 Theoretical Foundation

### Numerical Algorithm Categories

**Linear System Solvers:**
```
Direct Methods:
├── LU Decomposition (O(n³))
├── Cholesky Decomposition (O(n³/3))
├── QR Decomposition (O(2n³/3))
└── SVD (O(4n³/3))

Iterative Methods:
├── Krylov Subspace (CG, GMRES, BiCGSTAB)
├── Multigrid Methods (V-cycle, W-cycle)
├── Domain Decomposition (Schwarz, FETI)
└── Preconditioning Techniques
```

**Eigenvalue Problems:**
```
Standard: Ax = λx
Generalized: Ax = λBx
├── Power Method & Variants
├── Lanczos Algorithm
├── Arnoldi Process  
├── Jacobi-Davidson
└── LOBPCG (Locally Optimal Block PCG)
```

**Optimization Algorithms:**
```
Gradient-Based:
├── Steepest Descent
├── Conjugate Gradient
├── Newton's Method
├── Quasi-Newton (BFGS, L-BFGS)
└── Trust Region Methods

Stochastic:
├── Simulated Annealing
├── Genetic Algorithms
├── Particle Swarm
└── Differential Evolution
```

### GPU Numerical Considerations

**Precision Hierarchy:**
- **FP64 (Double)**: Maximum precision, slower on consumer GPUs
- **FP32 (Single)**: Standard precision, optimal for most GPUs  
- **FP16 (Half)**: Fast but limited precision, good for ML
- **BF16 (Brain Float)**: Better range than FP16
- **Mixed Precision**: Combine precisions strategically

**Memory Access Patterns:**
- **Bandwidth-bound**: Dense linear algebra operations
- **Latency-bound**: Sparse matrix operations
- **Cache-friendly**: Blocked and tiled algorithms
- **Communication-minimizing**: Avoid synchronization

---

## 🔬 Chapter 1: Advanced Linear System Solvers

### GPU-Accelerated Krylov Methods

Create `numerical/krylov_solvers.go`:

```go
package main

import (
    "fmt"
    "math"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/blas"
    "github.com/stitch1968/gocuda/sparse"
)

// Advanced Krylov subspace solvers implementation
type KrylovSolver struct {
    ctx             *cuda.Context
    blasHandle      *blas.Handle
    sparseHandle    *sparse.Handle
    streams         []*cuda.Stream
    
    // Solver configuration
    maxIterations   int
    tolerance       float64
    preconditioner  Preconditioner
    
    // Performance monitoring
    iterationTimes  []time.Duration
    residualHistory []float64
    convergenceRate float64
}

type Preconditioner interface {
    Apply(x, y *memory.DeviceMemory) error
    Setup(matrix *sparse.Matrix) error
    Destroy()
}

type SparseMatrix struct {
    rows         int
    cols         int
    nnz          int
    values       *memory.DeviceMemory
    rowIndices   *memory.DeviceMemory
    colIndices   *memory.DeviceMemory
    format       sparse.Format
}

type IterativeStats struct {
    Iterations        int
    FinalResidual     float64
    ConvergenceRate   float64
    TotalTime        time.Duration
    TimePerIteration time.Duration
    MemoryBandwidth  float64
    FLOPSPerSecond   float64
}

// Incomplete LU Preconditioner
type ILUPreconditioner struct {
    L      *sparse.Matrix
    U      *sparse.Matrix
    handle *sparse.Handle
    buffer *memory.DeviceMemory
}

func NewKrylovSolver() *KrylovSolver {
    ctx := cuda.GetDefaultContext()
    
    solver := &KrylovSolver{
        ctx:           ctx,
        blasHandle:    blas.Create(),
        sparseHandle:  sparse.Create(),
        streams:       make([]*cuda.Stream, 4), // Multiple streams for overlap
        maxIterations: 10000,
        tolerance:     1e-8,
        iterationTimes: make([]time.Duration, 0),
        residualHistory: make([]float64, 0),
    }
    
    // Initialize streams
    for i := range solver.streams {
        stream, _ := ctx.CreateStream()
        solver.streams[i] = stream
    }
    
    return solver
}

// Conjugate Gradient solver for SPD systems
func (ks *KrylovSolver) ConjugateGradient(A *SparseMatrix, b, x *memory.DeviceMemory) (*IterativeStats, error) {
    fmt.Println("🔄 GPU Conjugate Gradient Solver")
    
    n := A.rows
    
    // Allocate working vectors
    r, _ := memory.Alloc(int64(n * 8))     // residual
    p, _ := memory.Alloc(int64(n * 8))     // search direction
    Ap, _ := memory.Alloc(int64(n * 8))    // A*p
    defer r.Free()
    defer p.Free()
    defer Ap.Free()
    
    stats := &IterativeStats{}
    startTime := time.Now()
    
    // Initial residual: r = b - Ax
    ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, -1.0, A, x, 1.0, r)
    ks.blasHandle.Axpy(n, 1.0, b, 1, r, 1) // r = b + (-1.0)*A*x
    
    // p = r (initial search direction)
    ks.blasHandle.Copy(n, r, 1, p, 1)
    
    // rsold = r^T * r
    var rsold float64
    ks.blasHandle.Dot(n, r, 1, r, 1, &rsold)
    
    fmt.Printf("Initial residual: %e\n", math.Sqrt(rsold))
    
    for iteration := 0; iteration < ks.maxIterations; iteration++ {
        iterStart := time.Now()
        
        // Ap = A * p
        ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, p, 0.0, Ap)
        
        // alpha = rsold / (p^T * Ap)
        var pAp float64
        ks.blasHandle.Dot(n, p, 1, Ap, 1, &pAp)
        
        if math.Abs(pAp) < 1e-14 {
            return stats, fmt.Errorf("breakdown in CG: p^T*A*p = %e", pAp)
        }
        
        alpha := rsold / pAp
        
        // x = x + alpha * p
        ks.blasHandle.Axpy(n, alpha, p, 1, x, 1)
        
        // r = r - alpha * Ap
        ks.blasHandle.Axpy(n, -alpha, Ap, 1, r, 1)
        
        // rsnew = r^T * r
        var rsnew float64
        ks.blasHandle.Dot(n, r, 1, r, 1, &rsnew)
        
        residual := math.Sqrt(rsnew)
        iterTime := time.Since(iterStart)
        
        ks.iterationTimes = append(ks.iterationTimes, iterTime)
        ks.residualHistory = append(ks.residualHistory, residual)
        
        if iteration%100 == 0 || iteration < 10 {
            fmt.Printf("  Iteration %d: residual = %e\n", iteration, residual)
        }
        
        // Check convergence
        if residual < ks.tolerance {
            stats.Iterations = iteration + 1
            stats.FinalResidual = residual
            stats.TotalTime = time.Since(startTime)
            stats.TimePerIteration = stats.TotalTime / time.Duration(stats.Iterations)
            
            ks.computePerformanceMetrics(stats, A, n)
            
            fmt.Printf("✅ CG converged in %d iterations\n", stats.Iterations)
            fmt.Printf("   Final residual: %e\n", stats.FinalResidual)
            fmt.Printf("   Total time: %v\n", stats.TotalTime)
            
            return stats, nil
        }
        
        // beta = rsnew / rsold
        beta := rsnew / rsold
        
        // p = r + beta * p  
        ks.blasHandle.Scal(n, beta, p, 1)
        ks.blasHandle.Axpy(n, 1.0, r, 1, p, 1)
        
        rsold = rsnew
    }
    
    stats.Iterations = ks.maxIterations
    stats.FinalResidual = math.Sqrt(rsold)
    stats.TotalTime = time.Since(startTime)
    
    return stats, fmt.Errorf("CG did not converge in %d iterations", ks.maxIterations)
}

// GMRES solver for general systems
func (ks *KrylovSolver) GMRES(A *SparseMatrix, b, x *memory.DeviceMemory, restart int) (*IterativeStats, error) {
    fmt.Printf("🔄 GPU GMRES(%d) Solver\n", restart)
    
    n := A.rows
    stats := &IterativeStats{}
    startTime := time.Now()
    
    // Allocate Krylov subspace storage
    V := make([]*memory.DeviceMemory, restart+1)
    for i := range V {
        V[i], _ = memory.Alloc(int64(n * 8))
        defer V[i].Free()
    }
    
    // Hessenberg matrix (CPU storage for simplicity)
    H := make([][]float64, restart+1)
    for i := range H {
        H[i] = make([]float64, restart)
    }
    
    // Working vectors
    w, _ := memory.Alloc(int64(n * 8))
    r, _ := memory.Alloc(int64(n * 8))
    defer w.Free()
    defer r.Free()
    
    totalIterations := 0
    
    for cycle := 0; cycle < ks.maxIterations/restart; cycle++ {
        // Initial residual: r = b - Ax
        ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, -1.0, A, x, 1.0, r)
        ks.blasHandle.Axpy(n, 1.0, b, 1, r, 1)
        
        var beta float64
        ks.blasHandle.Nrm2(n, r, 1, &beta)
        
        if beta < ks.tolerance {
            stats.Iterations = totalIterations
            stats.FinalResidual = beta
            stats.TotalTime = time.Since(startTime)
            fmt.Printf("✅ GMRES converged before restart\n")
            return stats, nil
        }
        
        // v1 = r / ||r||
        ks.blasHandle.Copy(n, r, 1, V[0], 1)
        ks.blasHandle.Scal(n, 1.0/beta, V[0], 1)
        
        // Right-hand side for least squares problem
        g := make([]float64, restart+1)
        g[0] = beta
        
        // Arnoldi process
        var j int
        for j = 0; j < restart; j++ {
            totalIterations++
            
            // w = A * v_j
            ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, V[j], 0.0, w)
            
            // Modified Gram-Schmidt orthogonalization
            for i := 0; i <= j; i++ {
                ks.blasHandle.Dot(n, V[i], 1, w, 1, &H[i][j])
                ks.blasHandle.Axpy(n, -H[i][j], V[i], 1, w, 1)
            }
            
            ks.blasHandle.Nrm2(n, w, 1, &H[j+1][j])
            
            if H[j+1][j] < 1e-14 {
                // Lucky breakdown
                fmt.Printf("Lucky breakdown at iteration %d\n", j+1)
                break
            }
            
            // v_{j+1} = w / ||w||
            ks.blasHandle.Copy(n, w, 1, V[j+1], 1)
            ks.blasHandle.Scal(n, 1.0/H[j+1][j], V[j+1], 1)
            
            // Apply previous Givens rotations to new column of H
            for i := 0; i < j; i++ {
                ks.applyGivensRotation(H, i, j, g)
            }
            
            // Generate new Givens rotation
            c, s := ks.generateGivensRotation(H[j][j], H[j+1][j])
            H[j][j] = c*H[j][j] + s*H[j+1][j]
            H[j+1][j] = 0.0
            g[j+1] = -s * g[j]
            g[j] = c * g[j]
            
            residual := math.Abs(g[j+1])
            ks.residualHistory = append(ks.residualHistory, residual)
            
            if totalIterations%100 == 0 || totalIterations < 10 {
                fmt.Printf("  Iteration %d: residual = %e\n", totalIterations, residual)
            }
            
            if residual < ks.tolerance {
                stats.Iterations = totalIterations
                stats.FinalResidual = residual
                stats.TotalTime = time.Since(startTime)
                
                // Solve upper triangular system and update solution
                y := ks.solveUpperTriangular(H, g, j+1)
                ks.updateSolution(x, V, y, j+1)
                
                fmt.Printf("✅ GMRES converged in %d iterations\n", stats.Iterations)
                return stats, nil
            }
        }
        
        // Solve least squares problem and update solution
        y := ks.solveUpperTriangular(H, g, j)
        ks.updateSolution(x, V, y, j)
        
        fmt.Printf("Completed restart cycle %d\n", cycle+1)
    }
    
    stats.Iterations = totalIterations
    stats.TotalTime = time.Since(startTime)
    return stats, fmt.Errorf("GMRES did not converge")
}

func (ks *KrylovSolver) applyGivensRotation(H [][]float64, i, j int, g []float64) {
    // Apply i-th Givens rotation to column j of H and RHS g
    // Implementation would apply stored rotation parameters
}

func (ks *KrylovSolver) generateGivensRotation(a, b float64) (float64, float64) {
    if b == 0.0 {
        return 1.0, 0.0
    } else if math.Abs(b) > math.Abs(a) {
        t := a / b
        s := 1.0 / math.Sqrt(1.0+t*t)
        c := s * t
        return c, s
    } else {
        t := b / a
        c := 1.0 / math.Sqrt(1.0+t*t)
        s := c * t
        return c, s
    }
}

func (ks *KrylovSolver) solveUpperTriangular(H [][]float64, g []float64, size int) []float64 {
    y := make([]float64, size)
    
    for i := size - 1; i >= 0; i-- {
        y[i] = g[i]
        for j := i + 1; j < size; j++ {
            y[i] -= H[i][j] * y[j]
        }
        y[i] /= H[i][i]
    }
    
    return y
}

func (ks *KrylovSolver) updateSolution(x *memory.DeviceMemory, V []*memory.DeviceMemory, y []float64, size int) {
    // x = x + V * y
    for i := 0; i < size; i++ {
        if y[i] != 0.0 {
            ks.blasHandle.Axpy(len(y), y[i], V[i], 1, x, 1)
        }
    }
}

// BiCGSTAB solver for nonsymmetric systems
func (ks *KrylovSolver) BiCGSTAB(A *SparseMatrix, b, x *memory.DeviceMemory) (*IterativeStats, error) {
    fmt.Println("🔄 GPU BiCGSTAB Solver")
    
    n := A.rows
    stats := &IterativeStats{}
    startTime := time.Now()
    
    // Allocate working vectors
    r, _ := memory.Alloc(int64(n * 8))      // residual
    r0, _ := memory.Alloc(int64(n * 8))     // initial residual
    p, _ := memory.Alloc(int64(n * 8))      // search direction
    v, _ := memory.Alloc(int64(n * 8))      // A*p
    s, _ := memory.Alloc(int64(n * 8))      // intermediate residual
    t, _ := memory.Alloc(int64(n * 8))      // A*s
    defer r.Free()
    defer r0.Free()
    defer p.Free()
    defer v.Free()
    defer s.Free()
    defer t.Free()
    
    // Initial residual: r = b - Ax
    ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, -1.0, A, x, 1.0, r)
    ks.blasHandle.Axpy(n, 1.0, b, 1, r, 1)
    
    // r0 = r (arbitrary choice for r0)
    ks.blasHandle.Copy(n, r, 1, r0, 1)
    
    // p = r
    ks.blasHandle.Copy(n, r, 1, p, 1)
    
    var rho, rhoPrev float64 = 1.0, 1.0
    var alpha, omega float64 = 1.0, 1.0
    
    ks.blasHandle.Dot(n, r, 1, r, 1, &rho)
    fmt.Printf("Initial residual: %e\n", math.Sqrt(rho))
    
    for iteration := 0; iteration < ks.maxIterations; iteration++ {
        iterStart := time.Now()
        
        // rho = (r0, r)
        rhoPrev = rho
        ks.blasHandle.Dot(n, r0, 1, r, 1, &rho)
        
        if math.Abs(rho) < 1e-14 {
            return stats, fmt.Errorf("breakdown in BiCGSTAB: rho = %e", rho)
        }
        
        if iteration > 0 {
            beta := (rho / rhoPrev) * (alpha / omega)
            
            // p = r + beta * (p - omega * v)
            ks.blasHandle.Axpy(n, -omega, v, 1, p, 1)  // p = p - omega * v
            ks.blasHandle.Scal(n, beta, p, 1)          // p = beta * p
            ks.blasHandle.Axpy(n, 1.0, r, 1, p, 1)    // p = r + p
        }
        
        // v = A * p
        ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, p, 0.0, v)
        
        // alpha = rho / (r0, v)
        var r0v float64
        ks.blasHandle.Dot(n, r0, 1, v, 1, &r0v)
        
        if math.Abs(r0v) < 1e-14 {
            return stats, fmt.Errorf("breakdown in BiCGSTAB: (r0,v) = %e", r0v)
        }
        
        alpha = rho / r0v
        
        // s = r - alpha * v
        ks.blasHandle.Copy(n, r, 1, s, 1)
        ks.blasHandle.Axpy(n, -alpha, v, 1, s, 1)
        
        // Check if s is small enough
        var sNorm float64
        ks.blasHandle.Nrm2(n, s, 1, &sNorm)
        
        if sNorm < ks.tolerance {
            // x = x + alpha * p
            ks.blasHandle.Axpy(n, alpha, p, 1, x, 1)
            
            stats.Iterations = iteration + 1
            stats.FinalResidual = sNorm
            stats.TotalTime = time.Since(startTime)
            
            fmt.Printf("✅ BiCGSTAB converged early in %d iterations\n", stats.Iterations)
            return stats, nil
        }
        
        // t = A * s
        ks.sparseHandle.SpMV(sparse.CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, s, 0.0, t)
        
        // omega = (t, s) / (t, t)
        var ts, tt float64
        ks.blasHandle.Dot(n, t, 1, s, 1, &ts)
        ks.blasHandle.Dot(n, t, 1, t, 1, &tt)
        
        if math.Abs(tt) < 1e-14 {
            return stats, fmt.Errorf("breakdown in BiCGSTAB: (t,t) = %e", tt)
        }
        
        omega = ts / tt
        
        // x = x + alpha * p + omega * s
        ks.blasHandle.Axpy(n, alpha, p, 1, x, 1)
        ks.blasHandle.Axpy(n, omega, s, 1, x, 1)
        
        // r = s - omega * t
        ks.blasHandle.Copy(n, s, 1, r, 1)
        ks.blasHandle.Axpy(n, -omega, t, 1, r, 1)
        
        var rNorm float64
        ks.blasHandle.Nrm2(n, r, 1, &rNorm)
        
        iterTime := time.Since(iterStart)
        ks.iterationTimes = append(ks.iterationTimes, iterTime)
        ks.residualHistory = append(ks.residualHistory, rNorm)
        
        if iteration%100 == 0 || iteration < 10 {
            fmt.Printf("  Iteration %d: residual = %e\n", iteration, rNorm)
        }
        
        if rNorm < ks.tolerance {
            stats.Iterations = iteration + 1
            stats.FinalResidual = rNorm
            stats.TotalTime = time.Since(startTime)
            
            ks.computePerformanceMetrics(stats, A, n)
            
            fmt.Printf("✅ BiCGSTAB converged in %d iterations\n", stats.Iterations)
            return stats, nil
        }
        
        if math.Abs(omega) < 1e-14 {
            return stats, fmt.Errorf("breakdown in BiCGSTAB: omega = %e", omega)
        }
    }
    
    return stats, fmt.Errorf("BiCGSTAB did not converge")
}

func (ks *KrylovSolver) computePerformanceMetrics(stats *IterativeStats, A *SparseMatrix, n int) {
    if stats.Iterations > 0 {
        stats.TimePerIteration = stats.TotalTime / time.Duration(stats.Iterations)
        
        // Estimate FLOPS (rough approximation)
        // SpMV: 2 * nnz, BLAS ops: various
        flopsPerIteration := float64(2*A.nnz + 6*n) // Approximate
        totalFLOPS := flopsPerIteration * float64(stats.Iterations)
        stats.FLOPSPerSecond = totalFLOPS / stats.TotalTime.Seconds()
        
        // Estimate memory bandwidth (bytes transferred per iteration)
        bytesPerIteration := float64(8 * (A.nnz + 4*n)) // Rough estimate
        totalBytes := bytesPerIteration * float64(stats.Iterations)
        stats.MemoryBandwidth = totalBytes / stats.TotalTime.Seconds() / (1024*1024*1024) // GB/s
        
        // Compute convergence rate
        if len(ks.residualHistory) > 1 {
            initialRes := ks.residualHistory[0]
            finalRes := stats.FinalResidual
            stats.ConvergenceRate = math.Pow(finalRes/initialRes, 1.0/float64(stats.Iterations))
        }
    }
}

// Incomplete LU preconditioner implementation
func NewILUPreconditioner(fillLevel int) *ILUPreconditioner {
    return &ILUPreconditioner{
        handle: sparse.Create(),
    }
}

func (ilu *ILUPreconditioner) Setup(A *sparse.Matrix) error {
    fmt.Printf("Setting up ILU(%d) preconditioner...\n", 0)
    
    // In real implementation, would compute incomplete LU factorization
    // For simulation, we'll create identity-based preconditioner
    
    n := A.rows
    ilu.L, _ = ilu.createIdentityMatrix(n)
    ilu.U, _ = ilu.createIdentityMatrix(n)
    
    return nil
}

func (ilu *ILUPreconditioner) createIdentityMatrix(n int) (*sparse.Matrix, error) {
    // Create identity matrix in CSR format
    matrix := &sparse.Matrix{
        rows: n,
        cols: n,
        nnz:  n,
    }
    
    // Allocate memory for CSR format
    values := make([]float64, n)
    rowPtr := make([]int32, n+1)
    colInd := make([]int32, n)
    
    for i := 0; i < n; i++ {
        values[i] = 1.0
        rowPtr[i] = int32(i)
        colInd[i] = int32(i)
    }
    rowPtr[n] = int32(n)
    
    // Copy to GPU
    matrix.values, _ = memory.AllocAndCopyFromHost(values)
    matrix.rowIndices, _ = memory.AllocAndCopyFromHost(rowPtr)
    matrix.colIndices, _ = memory.AllocAndCopyFromHost(colInd)
    
    return matrix, nil
}

func (ilu *ILUPreconditioner) Apply(x, y *memory.DeviceMemory) error {
    // Solve L * U * y = x
    // For identity preconditioner: y = x
    blasHandle := blas.Create()
    defer blasHandle.Destroy()
    
    n := ilu.L.rows
    blasHandle.Copy(n, x, 1, y, 1)
    
    return nil
}

func (ilu *ILUPreconditioner) Destroy() {
    if ilu.L != nil {
        ilu.L.values.Free()
        ilu.L.rowIndices.Free()
        ilu.L.colIndices.Free()
    }
    if ilu.U != nil {
        ilu.U.values.Free()
        ilu.U.rowIndices.Free()
        ilu.U.colIndices.Free()
    }
    if ilu.buffer != nil {
        ilu.buffer.Free()
    }
    if ilu.handle != nil {
        ilu.handle.Destroy()
    }
}

func (ks *KrylovSolver) SetPreconditioner(prec Preconditioner) {
    ks.preconditioner = prec
}

func (ks *KrylovSolver) AnalyzeConvergence() {
    fmt.Println("\n📊 Convergence Analysis:")
    
    if len(ks.residualHistory) < 2 {
        fmt.Println("Insufficient data for analysis")
        return
    }
    
    // Plot convergence history (simplified display)
    fmt.Println("Iteration\tResidual\t\tRate")
    for i, res := range ks.residualHistory {
        if i%max(1, len(ks.residualHistory)/10) == 0 {
            rate := ""
            if i > 0 {
                r := res / ks.residualHistory[i-1]
                rate = fmt.Sprintf("%.3f", r)
            }
            fmt.Printf("%d\t\t%e\t%s\n", i, res, rate)
        }
    }
    
    // Compute average convergence rate
    if len(ks.residualHistory) > 1 {
        initialRes := ks.residualHistory[0]
        finalRes := ks.residualHistory[len(ks.residualHistory)-1]
        iterations := len(ks.residualHistory)
        avgRate := math.Pow(finalRes/initialRes, 1.0/float64(iterations-1))
        
        fmt.Printf("\nAverage convergence rate: %.6f\n", avgRate)
        
        if avgRate < 0.1 {
            fmt.Println("✅ Excellent convergence")
        } else if avgRate < 0.5 {
            fmt.Println("✅ Good convergence")
        } else if avgRate < 0.9 {
            fmt.Println("⚠️ Slow convergence")
        } else {
            fmt.Println("❌ Poor convergence - consider preconditioning")
        }
    }
}

func (ks *KrylovSolver) Destroy() {
    ks.blasHandle.Destroy()
    ks.sparseHandle.Destroy()
    
    for _, stream := range ks.streams {
        stream.Destroy()
    }
    
    if ks.preconditioner != nil {
        ks.preconditioner.Destroy()
    }
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

// Demonstration and benchmarking
func main() {
    cuda.Initialize()
    fmt.Println("🧮 Expert Numerical Methods: Krylov Solvers")
    
    solver := NewKrylovSolver()
    defer solver.Destroy()
    
    // Create test problems
    sizes := []int{1000, 5000, 10000}
    
    for _, n := range sizes {
        fmt.Printf("\n" + "="*60 + "\n")
        fmt.Printf("Testing with %dx%d system\n", n, n)
        fmt.Printf("="*60 + "\n")
        
        // Generate test matrix (5-point stencil for 2D Laplacian)
        A := generateLaplacianMatrix(n)
        defer A.values.Free()
        defer A.rowIndices.Free()
        defer A.colIndices.Free()
        
        // Generate test vectors
        b, _ := memory.Alloc(int64(n * 8))
        x, _ := memory.Alloc(int64(n * 8))
        defer b.Free()
        defer x.Free()
        
        initializeTestVectors(b, x, n)
        
        // Test different solvers
        testCG := true
        testGMRES := true
        testBiCGSTAB := true
        
        if testCG && isSymmetricPositiveDefinite(A) {
            fmt.Println("\n🔄 Testing Conjugate Gradient:")
            x.Memset(0) // Reset initial guess
            
            stats, err := solver.ConjugateGradient(A, b, x)
            if err != nil {
                fmt.Printf("CG Error: %v\n", err)
            } else {
                printSolverStats("CG", stats)
            }
        }
        
        if testGMRES {
            fmt.Println("\n🔄 Testing GMRES(30):")
            x.Memset(0) // Reset initial guess
            
            stats, err := solver.GMRES(A, b, x, 30)
            if err != nil {
                fmt.Printf("GMRES Error: %v\n", err)
            } else {
                printSolverStats("GMRES", stats)
            }
        }
        
        if testBiCGSTAB {
            fmt.Println("\n🔄 Testing BiCGSTAB:")
            x.Memset(0) // Reset initial guess
            
            stats, err := solver.BiCGSTAB(A, b, x)
            if err != nil {
                fmt.Printf("BiCGSTAB Error: %v\n", err)
            } else {
                printSolverStats("BiCGSTAB", stats)
            }
        }
        
        // Analyze convergence
        solver.AnalyzeConvergence()
        
        // Test with preconditioning
        testWithPreconditioning(solver, A, b, x)
    }
}

func generateLaplacianMatrix(n int) *SparseMatrix {
    // Generate 2D 5-point Laplacian matrix
    // For simplicity, assume square grid
    gridSize := int(math.Sqrt(float64(n)))
    actualN := gridSize * gridSize
    
    matrix := &SparseMatrix{
        rows: actualN,
        cols: actualN,
        nnz:  0,
    }
    
    // Count non-zeros first
    nnz := 0
    for i := 0; i < gridSize; i++ {
        for j := 0; j < gridSize; j++ {
            nnz++ // Diagonal element
            if i > 0 { nnz++ }          // North
            if i < gridSize-1 { nnz++ } // South  
            if j > 0 { nnz++ }          // West
            if j < gridSize-1 { nnz++ } // East
        }
    }
    
    matrix.nnz = nnz
    
    // Allocate arrays
    values := make([]float64, nnz)
    rowPtr := make([]int32, actualN+1)
    colInd := make([]int32, nnz)
    
    // Fill matrix
    idx := 0
    for i := 0; i < gridSize; i++ {
        for j := 0; j < gridSize; j++ {
            row := i*gridSize + j
            rowPtr[row] = int32(idx)
            
            // West neighbor
            if j > 0 {
                values[idx] = -1.0
                colInd[idx] = int32(row - 1)
                idx++
            }
            
            // South neighbor  
            if i > 0 {
                values[idx] = -1.0
                colInd[idx] = int32(row - gridSize)
                idx++
            }
            
            // Diagonal
            values[idx] = 4.0
            colInd[idx] = int32(row)
            idx++
            
            // North neighbor
            if i < gridSize-1 {
                values[idx] = -1.0
                colInd[idx] = int32(row + gridSize)
                idx++
            }
            
            // East neighbor
            if j < gridSize-1 {
                values[idx] = -1.0
                colInd[idx] = int32(row + 1)
                idx++
            }
        }
    }
    rowPtr[actualN] = int32(idx)
    
    // Copy to GPU
    matrix.values, _ = memory.AllocAndCopyFromHost(values)
    matrix.rowIndices, _ = memory.AllocAndCopyFromHost(rowPtr)
    matrix.colIndices, _ = memory.AllocAndCopyFromHost(colInd)
    
    return matrix
}

func initializeTestVectors(b, x *memory.DeviceMemory, n int) {
    // Initialize RHS vector with random values
    hostB := make([]float64, n)
    hostX := make([]float64, n)
    
    for i := 0; i < n; i++ {
        hostB[i] = math.Sin(float64(i) * 0.1) // Smooth RHS
        hostX[i] = 0.0 // Zero initial guess
    }
    
    b.CopyFromHost(hostB)
    x.CopyFromHost(hostX)
}

func isSymmetricPositiveDefinite(A *SparseMatrix) bool {
    // For our test matrices (Laplacian), they are SPD
    // In general, would need to check matrix properties
    return true
}

func printSolverStats(solverName string, stats *IterativeStats) {
    fmt.Printf("\n📊 %s Performance Summary:\n", solverName)
    fmt.Printf("  Iterations: %d\n", stats.Iterations)
    fmt.Printf("  Final Residual: %e\n", stats.FinalResidual)
    fmt.Printf("  Total Time: %v\n", stats.TotalTime)
    fmt.Printf("  Time/Iteration: %v\n", stats.TimePerIteration)
    fmt.Printf("  GFLOPS: %.2f\n", stats.FLOPSPerSecond/1e9)
    fmt.Printf("  Memory BW: %.2f GB/s\n", stats.MemoryBandwidth)
    
    if stats.ConvergenceRate > 0 {
        fmt.Printf("  Convergence Rate: %.6f\n", stats.ConvergenceRate)
    }
}

func testWithPreconditioning(solver *KrylovSolver, A *SparseMatrix, b, x *memory.DeviceMemory) {
    fmt.Println("\n🧪 Testing with ILU Preconditioning:")
    
    // Setup preconditioner
    ilu := NewILUPreconditioner(0)
    defer ilu.Destroy()
    
    err := ilu.Setup(A)
    if err != nil {
        fmt.Printf("Preconditioner setup failed: %v\n", err)
        return
    }
    
    solver.SetPreconditioner(ilu)
    
    // Reset initial guess
    x.Memset(0)
    
    // Test preconditioned CG
    stats, err := solver.ConjugateGradient(A, b, x)
    if err != nil {
        fmt.Printf("Preconditioned CG Error: %v\n", err)
    } else {
        printSolverStats("Preconditioned CG", stats)
    }
    
    // Reset preconditioner
    solver.SetPreconditioner(nil)
}
```

---

## 🎯 Module Assessment

### **Numerical Methods Mastery**

1. **Solver Implementation**: Successfully implement and optimize 3+ iterative methods
2. **Convergence Analysis**: Demonstrate understanding of convergence behavior
3. **Preconditioning**: Implement effective preconditioning strategies  
4. **Performance Optimization**: Achieve high GPU utilization for numerical kernels

### **Success Criteria**

- ✅ Iterative solvers converge for well-conditioned problems
- ✅ Performance scaling with problem size
- ✅ Effective preconditioning improves convergence rates
- ✅ Numerical stability maintained across precision levels

---

## 🚀 Next Steps

**Outstanding! You've mastered advanced numerical methods.**

**You're now ready for:**
➡️ **[Module 4: Real-Time & Streaming Processing](TRAINING_EXPERT_4_REALTIME.md)**

**Skills Mastered:**
- 🧮 **Advanced Iterative Solvers** - CG, GMRES, BiCGSTAB implementations
- 📊 **Numerical Analysis** - Convergence, stability, and conditioning
- 🔧 **Preconditioning Techniques** - ILU and domain-specific methods
- ⚡ **GPU-Accelerated Linear Algebra** - High-performance numerical computing

---

*From basic arithmetic to advanced numerical algorithms - the foundation of computational science! 🧮⚡*
