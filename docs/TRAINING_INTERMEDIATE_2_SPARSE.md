# 🕸️ Module 2: Sparse Computing Mastery

**Goal:** Master complex sparse matrix operations and become proficient with cuSPARSE for real-world large-scale problems

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🕸️ **Master all sparse matrix formats** and choose optimal representations
- ⚡ **Implement iterative solvers** for systems with millions of unknowns
- 🔗 **Apply sparse computing** to graph algorithms and network analysis
- 📊 **Optimize sparse operations** for maximum performance
- 🧮 **Handle numerical challenges** in sparse computations

---

## 🧠 Theoretical Foundation

### Sparse Matrix Fundamentals

**Sparsity Patterns:**
- **Diagonal** - Tridiagonal, pentadiagonal systems (PDEs)
- **Block Structured** - Finite element methods
- **Random** - Social networks, web graphs
- **Hierarchical** - Multigrid methods

**Storage Efficiency:**
```
Dense Matrix (n×n): n² storage
Sparse Matrix: ~2×nnz storage (nnz = non-zeros)
Compression Ratio: n²/(2×nnz)
```

### Matrix Format Selection Guide

| Format | Best For | Memory | Access |
|--------|----------|--------|--------|
| **CSR** | SpMV, general purpose | Efficient | Row-wise |
| **CSC** | SpMV transpose | Efficient | Column-wise |
| **COO** | Matrix construction | Simple | Random |
| **ELL** | Regular sparsity | Padded | Vectorized |
| **HYB** | Irregular sparsity | Adaptive | Mixed |

---

## 🏗️ Chapter 1: Advanced Sparse Matrix Operations

### Comprehensive Sparse Matrix Implementation

Create `sparse/advanced_sparse.go`:

```go
package main

import (
    "fmt"
    "math"
    "sort"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

// Advanced sparse matrix operations manager
type SparseExpert struct {
    ctx     *cuda.Context
    sparse  *libraries.SparseContext
    profiler *SparseProfiler
}

type SparseProfiler struct {
    operations map[string]time.Duration
    memoryUsage map[string]int64
}

func NewSparseExpert() *SparseExpert {
    cuda.Initialize()
    ctx := cuda.GetDefaultContext()
    sparse, _ := libraries.CreateSparseContext()
    
    return &SparseExpert{
        ctx:     ctx,
        sparse:  sparse,
        profiler: &SparseProfiler{
            operations: make(map[string]time.Duration),
            memoryUsage: make(map[string]int64),
        },
    }
}

// Demonstrate all sparse matrix formats
func (s *SparseExpert) DemonstrateFormats() {
    fmt.Println("🕸️ Sparse Matrix Format Analysis")
    
    // Create test matrix (5-point stencil for 2D Laplacian)
    n := 100 // 100x100 grid
    matrix := s.create2DLaplacian(n)
    
    formats := []libraries.MatrixFormat{
        libraries.MatrixFormatCSR,
        libraries.MatrixFormatCSC,
        libraries.MatrixFormatCOO,
        libraries.MatrixFormatELL,
        libraries.MatrixFormatHYB,
    }
    
    fmt.Println("\nFormat Comparison for 2D Laplacian (100x100):")
    fmt.Println("Format\tMemory (KB)\tSpMV Time\tConstruction Time")
    
    for _, format := range formats {
        s.benchmarkFormat(matrix, format, n*n)
    }
}

func (s *SparseExpert) create2DLaplacian(n int) *SparseMatrix {
    // Create 2D 5-point stencil Laplacian matrix
    // Pattern: [-1, -1, 4, -1, -1] for interior points
    
    size := n * n
    nnz := estimateLaplacian2DNnz(n)
    
    matrix, _ := s.sparse.CreateSparseMatrix(size, size, nnz, libraries.MatrixFormatCOO)
    
    // Fill matrix with 5-point stencil pattern
    s.populateLaplacian2D(matrix, n)
    
    return matrix
}

func estimateLaplacian2DNnz(n int) int {
    // Interior points: 5 non-zeros each
    // Boundary points: 3-4 non-zeros each
    // Corner points: 2-3 non-zeros each
    interior := (n-2) * (n-2) * 5
    boundary := 4 * (n-2) * 4  // 4 edges
    corners := 4 * 3           // 4 corners
    return interior + boundary + corners
}

func (s *SparseExpert) populateLaplacian2D(matrix *libraries.SparseMatrix, n int) {
    // This would populate the matrix with 2D Laplacian pattern
    // For simulation, we'll create representative data
    
    fmt.Printf("Created 2D Laplacian: %dx%d grid, ~%d non-zeros\n", 
               n, n, estimateLaplacian2DNnz(n))
}

func (s *SparseExpert) benchmarkFormat(matrix *libraries.SparseMatrix, format libraries.MatrixFormat, size int) {
    formatName := s.getFormatName(format)
    
    // Convert to specific format
    start := time.Now()
    convertedMatrix, _ := s.sparse.ConvertFormat(matrix, format)
    conversionTime := time.Since(start)
    
    // Estimate memory usage
    memoryUsage := s.estimateMemoryUsage(convertedMatrix, format, size)
    
    // Benchmark SpMV
    vector, _ := memory.Alloc(int64(size * 4))
    result, _ := memory.Alloc(int64(size * 4))
    defer vector.Free()
    defer result.Free()
    
    // Initialize vector
    hostVector := make([]float32, size)
    for i := range hostVector {
        hostVector[i] = float32(i%100) / 100.0
    }
    vector.CopyFromHost(hostVector)
    
    // Benchmark SpMV
    start = time.Now()
    s.sparse.SpMV(1.0, convertedMatrix, vector, 0.0, result)
    spMVTime := time.Since(start)
    
    fmt.Printf("%s\t%.1f\t\t%v\t\t%v\n", 
               formatName, float64(memoryUsage)/1024, spMVTime, conversionTime)
    
    convertedMatrix.Destroy()
}

func (s *SparseExpert) getFormatName(format libraries.MatrixFormat) string {
    switch format {
    case libraries.MatrixFormatCSR:
        return "CSR"
    case libraries.MatrixFormatCSC:
        return "CSC"
    case libraries.MatrixFormatCOO:
        return "COO"
    case libraries.MatrixFormatELL:
        return "ELL"
    case libraries.MatrixFormatHYB:
        return "HYB"
    default:
        return "Unknown"
    }
}

func (s *SparseExpert) estimateMemoryUsage(matrix *libraries.SparseMatrix, format libraries.MatrixFormat, size int) int64 {
    // Simplified memory estimation
    nnz := matrix.GetNnz() // This would be available in real implementation
    
    switch format {
    case libraries.MatrixFormatCSR:
        return int64(nnz*4 + nnz*4 + (size+1)*4) // values + indices + rowPtr
    case libraries.MatrixFormatCSC:
        return int64(nnz*4 + nnz*4 + (size+1)*4) // values + indices + colPtr
    case libraries.MatrixFormatCOO:
        return int64(nnz*4 + nnz*4 + nnz*4) // values + rowInd + colInd
    case libraries.MatrixFormatELL:
        maxRowNnz := s.estimateMaxRowNnz(matrix)
        return int64(size*maxRowNnz*4 + size*maxRowNnz*4) // values + indices (padded)
    case libraries.MatrixFormatHYB:
        return int64(nnz * 6) // Hybrid overhead
    default:
        return int64(nnz * 8) // Conservative estimate
    }
}

func (s *SparseExpert) estimateMaxRowNnz(matrix *libraries.SparseMatrix) int {
    // For 2D Laplacian, max is 5 non-zeros per row
    return 5
}

func (s *SparseExpert) Destroy() {
    s.sparse.DestroyContext()
}

// Demonstration
func main() {
    cuda.Initialize()
    fmt.Println("🕸️ Advanced Sparse Computing")
    
    expert := NewSparseExpert()
    defer expert.Destroy()
    
    // Demonstrate different aspects
    expert.DemonstrateFormats()
    
    // Advanced operations
    expert.demonstrateAdvancedOperations()
    
    // Performance analysis
    expert.performanceAnalysis()
}

func (s *SparseExpert) demonstrateAdvancedOperations() {
    fmt.Println("\n🔧 Advanced Sparse Operations:")
    
    // Matrix-matrix multiplication
    s.demonstrateSpGEMM()
    
    // Triangular solve
    s.demonstrateTriangularSolve()
    
    // Matrix factorizations
    s.demonstrateFactorizations()
}

func (s *SparseExpert) demonstrateSpGEMM() {
    fmt.Println("\n1. Sparse Matrix-Matrix Multiplication (SpGEMM):")
    
    size := 1000
    nnzA := size * 5  // ~5 non-zeros per row
    nnzB := size * 5
    
    A, _ := s.sparse.CreateSparseMatrix(size, size, nnzA, libraries.MatrixFormatCSR)
    B, _ := s.sparse.CreateSparseMatrix(size, size, nnzB, libraries.MatrixFormatCSR)
    defer A.Destroy()
    defer B.Destroy()
    
    fmt.Printf("  Computing C = A * B where A,B are %dx%d with ~%d non-zeros each\n", 
               size, size, nnzA)
    
    start := time.Now()
    C, err := s.sparse.SpGEMM(A, B)
    elapsed := time.Since(start)
    
    if err == nil {
        defer C.Destroy()
        fmt.Printf("  ✅ SpGEMM completed in %v\n", elapsed)
        fmt.Printf("  Result matrix C has ~%d non-zeros\n", C.GetNnz())
    } else {
        fmt.Printf("  ❌ SpGEMM failed: %v\n", err)
    }
}

func (s *SparseExpert) demonstrateTriangularSolve() {
    fmt.Println("\n2. Triangular System Solve:")
    
    size := 1000
    nnz := size * 3  // Lower triangular with ~3 per row
    
    L, _ := s.sparse.CreateSparseMatrix(size, size, nnz, libraries.MatrixFormatCSR)
    defer L.Destroy()
    
    b, _ := memory.Alloc(int64(size * 4))
    x, _ := memory.Alloc(int64(size * 4))
    defer b.Free()
    defer x.Free()
    
    // Initialize RHS vector
    hostB := make([]float32, size)
    for i := range hostB {
        hostB[i] = float32(i + 1)
    }
    b.CopyFromHost(hostB)
    
    fmt.Printf("  Solving Lx = b where L is lower triangular %dx%d\n", size, size)
    
    start := time.Now()
    err := s.sparse.SpSolve(L, b, x, libraries.SolveTypeLower)
    elapsed := time.Since(start)
    
    if err == nil {
        fmt.Printf("  ✅ Triangular solve completed in %v\n", elapsed)
        
        // Verify solution (sample)
        result := make([]float32, 10)
        x.CopyToHost(result)
        fmt.Printf("  First 10 solution components: %v\n", result)
    } else {
        fmt.Printf("  ❌ Triangular solve failed: %v\n", err)
    }
}

func (s *SparseExpert) demonstrateFactorizations() {
    fmt.Println("\n3. Matrix Factorizations:")
    
    size := 500
    nnz := size * 7  // Symmetric positive definite
    
    A, _ := s.sparse.CreateSparseMatrix(size, size, nnz, libraries.MatrixFormatCSR)
    defer A.Destroy()
    
    // LU Factorization
    fmt.Printf("  LU Factorization of %dx%d sparse matrix\n", size, size)
    start := time.Now()
    L, U, err := s.sparse.SpLU(A)
    luTime := time.Since(start)
    
    if err == nil {
        defer L.Destroy()
        defer U.Destroy()
        fmt.Printf("  ✅ LU factorization completed in %v\n", luTime)
        fmt.Printf("  L has ~%d non-zeros, U has ~%d non-zeros\n", 
                   L.GetNnz(), U.GetNnz())
    }
    
    // Cholesky Factorization (for SPD matrices)
    fmt.Printf("  Cholesky Factorization (assuming SPD)\n")
    start = time.Now()
    err = s.sparse.SpCholesky(A)
    choleskyTime := time.Since(start)
    
    if err == nil {
        fmt.Printf("  ✅ Cholesky factorization completed in %v\n", choleskyTime)
    } else {
        fmt.Printf("  ⚠️ Cholesky failed (matrix may not be SPD): %v\n", err)
    }
}

func (s *SparseExpert) performanceAnalysis() {
    fmt.Println("\n📊 Sparse Performance Analysis:")
    
    // Test different sparsity levels
    sizes := []int{100, 500, 1000, 2000}
    densities := []float64{0.01, 0.05, 0.10, 0.20} // 1%, 5%, 10%, 20% density
    
    fmt.Println("\nSpMV Performance vs Density:")
    fmt.Println("Size\tDensity\tNnz\tTime\t\tGFLOPs")
    
    for _, size := range sizes {
        for _, density := range densities {
            s.benchmarkSpMVDensity(size, density)
        }
    }
    
    fmt.Println("\n💡 Performance Insights:")
    fmt.Println("  - Very sparse (<1%): Limited by memory latency")
    fmt.Println("  - Medium sparse (1-10%): Good SpMV performance")
    fmt.Println("  - Dense (>10%): Consider dense algorithms")
    fmt.Println("  - Irregular patterns: Use HYB format")
    fmt.Println("  - Regular patterns: Use ELL format")
}

func (s *SparseExpert) benchmarkSpMVDensity(size int, density float64) {
    nnz := int(float64(size*size) * density)
    
    matrix, _ := s.sparse.CreateSparseMatrix(size, size, nnz, libraries.MatrixFormatCSR)
    defer matrix.Destroy()
    
    x, _ := memory.Alloc(int64(size * 4))
    y, _ := memory.Alloc(int64(size * 4))
    defer x.Free()
    defer y.Free()
    
    // Initialize vector
    hostX := make([]float32, size)
    for i := range hostX {
        hostX[i] = float32(i % 100) / 100.0
    }
    x.CopyFromHost(hostX)
    
    // Warm-up
    s.sparse.SpMV(1.0, matrix, x, 0.0, y)
    
    // Benchmark
    const iterations = 10
    start := time.Now()
    for i := 0; i < iterations; i++ {
        s.sparse.SpMV(1.0, matrix, x, 0.0, y)
    }
    elapsed := time.Since(start) / iterations
    
    // Calculate GFLOPS
    operations := float64(2 * nnz) // multiply + add per non-zero
    gflops := (operations / elapsed.Seconds()) / 1e9
    
    fmt.Printf("%d\t%.2f%%\t%d\t%v\t%.2f\n", 
               size, density*100, nnz, elapsed, gflops)
}

// Helper type for sparse matrix (would be part of libraries package)
type SparseMatrix struct {
    rows, cols, nnz int
    format          libraries.MatrixFormat
}

func (m *SparseMatrix) GetNnz() int {
    return m.nnz
}

func (m *SparseMatrix) Destroy() {
    // Cleanup implementation
}
```

---

## ⚡ Chapter 2: Iterative Solvers Implementation

Create `sparse/iterative_solvers.go`:

```go
package main

import (
    "fmt"
    "math"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

// Advanced iterative solvers for sparse systems
type IterativeSolvers struct {
    ctx    *cuda.Context
    sparse *libraries.SparseContext
    thrust *libraries.ThrustContext
}

func NewIterativeSolvers() *IterativeSolvers {
    cuda.Initialize()
    ctx := cuda.GetDefaultContext()
    sparse, _ := libraries.CreateSparseContext()
    thrust, _ := libraries.ThrustContext()
    
    return &IterativeSolvers{
        ctx:    ctx,
        sparse: sparse,
        thrust: thrust,
    }
}

// Conjugate Gradient solver for SPD systems
func (is *IterativeSolvers) ConjugateGradient(A *libraries.SparseMatrix, b, x *memory.DeviceMemory, 
                                              maxIter int, tolerance float64) (*CGResult, error) {
    fmt.Printf("🔄 Conjugate Gradient Solver (tol=%.2e, maxIter=%d)\n", tolerance, maxIter)
    
    n := A.GetRows()
    
    // Allocate working vectors
    r, _ := memory.Alloc(int64(n * 4))    // residual
    p, _ := memory.Alloc(int64(n * 4))    // search direction
    Ap, _ := memory.Alloc(int64(n * 4))   // A * p
    defer r.Free()
    defer p.Free()
    defer Ap.Free()
    
    // r = b - A*x (initial residual)
    is.sparse.SpMV(1.0, A, x, 0.0, Ap)  // Ap = A*x
    is.vectorSubtract(b, Ap, r, n)       // r = b - Ap
    
    // p = r (initial search direction)
    is.vectorCopy(r, p, n)
    
    // rsold = r^T * r
    rsold, _ := is.dotProduct(r, r, n)
    
    result := &CGResult{
        InitialResidual: math.Sqrt(float64(rsold)),
        Iterations:      0,
        Converged:       false,
    }
    
    fmt.Printf("  Initial residual: %.6e\n", result.InitialResidual)
    
    start := time.Now()
    
    for iter := 0; iter < maxIter; iter++ {
        // Ap = A * p
        is.sparse.SpMV(1.0, A, p, 0.0, Ap)
        
        // alpha = rsold / (p^T * Ap)
        pAp, _ := is.dotProduct(p, Ap, n)
        if math.Abs(float64(pAp)) < 1e-14 {
            break // Avoid division by zero
        }
        alpha := rsold / pAp
        
        // x = x + alpha * p
        is.vectorAxpy(alpha, p, x, n)
        
        // r = r - alpha * Ap
        is.vectorAxpy(-alpha, Ap, r, n)
        
        // rsnew = r^T * r
        rsnew, _ := is.dotProduct(r, r, n)
        residualNorm := math.Sqrt(float64(rsnew))
        
        // Check convergence
        if residualNorm < tolerance {
            result.Converged = true
            result.Iterations = iter + 1
            result.FinalResidual = residualNorm
            break
        }
        
        // beta = rsnew / rsold
        beta := rsnew / rsold
        
        // p = r + beta * p
        is.vectorScale(p, beta, n)      // p = beta * p
        is.vectorAdd(r, p, p, n)        // p = r + p
        
        rsold = rsnew
        
        // Progress reporting
        if (iter+1)%100 == 0 {
            fmt.Printf("  Iteration %d: residual = %.6e\n", iter+1, residualNorm)
        }
    }
    
    result.SolveTime = time.Since(start)
    
    if result.Converged {
        fmt.Printf("  ✅ Converged in %d iterations (%.6e) after %v\n", 
                   result.Iterations, result.FinalResidual, result.SolveTime)
    } else {
        fmt.Printf("  ⚠️ Did not converge in %d iterations\n", maxIter)
        result.FinalResidual = math.Sqrt(float64(rsold))
    }
    
    return result, nil
}

// BiCGSTAB solver for nonsymmetric systems
func (is *IterativeSolvers) BiCGSTAB(A *libraries.SparseMatrix, b, x *memory.DeviceMemory,
                                     maxIter int, tolerance float64) (*CGResult, error) {
    fmt.Printf("🔄 BiCGSTAB Solver (tol=%.2e, maxIter=%d)\n", tolerance, maxIter)
    
    n := A.GetRows()
    
    // Allocate working vectors for BiCGSTAB
    r, _ := memory.Alloc(int64(n * 4))     // residual
    r0, _ := memory.Alloc(int64(n * 4))    // initial residual
    v, _ := memory.Alloc(int64(n * 4))     // A*p
    p, _ := memory.Alloc(int64(n * 4))     // search direction
    s, _ := memory.Alloc(int64(n * 4))     // intermediate
    t, _ := memory.Alloc(int64(n * 4))     // A*s
    defer r.Free()
    defer r0.Free()
    defer v.Free()
    defer p.Free()
    defer s.Free()
    defer t.Free()
    
    // r = b - A*x (initial residual)
    is.sparse.SpMV(1.0, A, x, 0.0, v)   // v = A*x
    is.vectorSubtract(b, v, r, n)        // r = b - v
    
    // r0 = r, p = r
    is.vectorCopy(r, r0, n)
    is.vectorCopy(r, p, n)
    
    // rho = r0^T * r
    rho, _ := is.dotProduct(r0, r, n)
    
    result := &CGResult{
        InitialResidual: math.Sqrt(math.Abs(float64(rho))),
        Iterations:      0,
        Converged:       false,
    }
    
    start := time.Now()
    
    for iter := 0; iter < maxIter; iter++ {
        // v = A * p
        is.sparse.SpMV(1.0, A, p, 0.0, v)
        
        // alpha = rho / (r0^T * v)
        r0v, _ := is.dotProduct(r0, v, n)
        if math.Abs(float64(r0v)) < 1e-14 {
            break
        }
        alpha := rho / r0v
        
        // s = r - alpha * v
        is.vectorCopy(r, s, n)
        is.vectorAxpy(-alpha, v, s, n)
        
        // Check if s is small enough
        sNorm, _ := is.vectorNorm(s, n)
        if sNorm < tolerance {
            is.vectorAxpy(alpha, p, x, n)
            result.Converged = true
            result.Iterations = iter + 1
            result.FinalResidual = sNorm
            break
        }
        
        // t = A * s
        is.sparse.SpMV(1.0, A, s, 0.0, t)
        
        // omega = (t^T * s) / (t^T * t)
        ts, _ := is.dotProduct(t, s, n)
        tt, _ := is.dotProduct(t, t, n)
        if math.Abs(float64(tt)) < 1e-14 {
            break
        }
        omega := ts / tt
        
        // x = x + alpha * p + omega * s
        is.vectorAxpy(alpha, p, x, n)
        is.vectorAxpy(omega, s, x, n)
        
        // r = s - omega * t
        is.vectorCopy(s, r, n)
        is.vectorAxpy(-omega, t, r, n)
        
        // Check convergence
        residualNorm, _ := is.vectorNorm(r, n)
        if residualNorm < tolerance {
            result.Converged = true
            result.Iterations = iter + 1
            result.FinalResidual = residualNorm
            break
        }
        
        // rho_new = r0^T * r
        rhoNew, _ := is.dotProduct(r0, r, n)
        if math.Abs(float64(rhoNew)) < 1e-14 {
            break
        }
        
        // beta = (rho_new / rho) * (alpha / omega)
        beta := (rhoNew / rho) * (alpha / omega)
        
        // p = r + beta * (p - omega * v)
        is.vectorAxpy(-omega, v, p, n)  // p = p - omega * v
        is.vectorScale(p, beta, n)       // p = beta * p
        is.vectorAdd(r, p, p, n)         // p = r + p
        
        rho = rhoNew
        
        if (iter+1)%100 == 0 {
            fmt.Printf("  Iteration %d: residual = %.6e\n", iter+1, residualNorm)
        }
    }
    
    result.SolveTime = time.Since(start)
    
    if result.Converged {
        fmt.Printf("  ✅ BiCGSTAB converged in %d iterations (%.6e) after %v\n",
                   result.Iterations, result.FinalResidual, result.SolveTime)
    } else {
        fmt.Printf("  ⚠️ BiCGSTAB did not converge in %d iterations\n", maxIter)
    }
    
    return result, nil
}

// Vector operations (these would be optimized GPU kernels)
func (is *IterativeSolvers) vectorCopy(src, dst *memory.DeviceMemory, n int) {
    // Copy src to dst
    is.thrust.Copy(src, dst, n, libraries.PolicyDevice)
}

func (is *IterativeSolvers) vectorAdd(a, b, result *memory.DeviceMemory, n int) {
    // result = a + b
    is.thrust.Transform2(a, b, result, n, "add", libraries.PolicyDevice)
}

func (is *IterativeSolvers) vectorSubtract(a, b, result *memory.DeviceMemory, n int) {
    // result = a - b
    is.thrust.Transform2(a, b, result, n, "subtract", libraries.PolicyDevice)
}

func (is *IterativeSolvers) vectorAxpy(alpha float32, x, y *memory.DeviceMemory, n int) {
    // y = alpha * x + y
    is.thrust.Axpy(alpha, x, y, n, libraries.PolicyDevice)
}

func (is *IterativeSolvers) vectorScale(x *memory.DeviceMemory, alpha float32, n int) {
    // x = alpha * x
    is.thrust.Scale(x, alpha, n, libraries.PolicyDevice)
}

func (is *IterativeSolvers) dotProduct(x, y *memory.DeviceMemory, n int) (float32, error) {
    // return x^T * y
    return is.thrust.InnerProduct(x, y, n, libraries.PolicyDevice)
}

func (is *IterativeSolvers) vectorNorm(x *memory.DeviceMemory, n int) (float64, error) {
    // return ||x||_2
    dot, err := is.dotProduct(x, x, n)
    return math.Sqrt(float64(dot)), err
}

func (is *IterativeSolvers) Destroy() {
    is.sparse.DestroyContext()
    is.thrust.DestroyContext()
}

// Result structure
type CGResult struct {
    InitialResidual float64
    FinalResidual   float64
    Iterations      int
    Converged       bool
    SolveTime       time.Duration
}

// Demonstration
func main() {
    cuda.Initialize()
    fmt.Println("⚡ Iterative Solvers for Sparse Systems")
    
    solvers := NewIterativeSolvers()
    defer solvers.Destroy()
    
    // Test problems
    testConjugateGradient(solvers)
    testBiCGSTAB(solvers)
    
    // Performance comparison
    compareIterativeSolvers(solvers)
}

func testConjugateGradient(solvers *IterativeSolvers) {
    fmt.Println("\n🧪 Testing Conjugate Gradient:")
    
    n := 1000
    nnz := n * 5  // Approximately 5 non-zeros per row
    
    // Create SPD test matrix (2D Laplacian)
    A, _ := solvers.sparse.CreateSparseMatrix(n, n, nnz, libraries.MatrixFormatCSR)
    defer A.Destroy()
    
    // Create test vectors
    b, _ := memory.Alloc(int64(n * 4))
    x, _ := memory.Alloc(int64(n * 4))
    defer b.Free()
    defer x.Free()
    
    // Initialize with test data
    hostB := make([]float32, n)
    hostX := make([]float32, n)
    for i := range hostB {
        hostB[i] = 1.0  // RHS = ones
        hostX[i] = 0.0  // Initial guess = zero
    }
    b.CopyFromHost(hostB)
    x.CopyFromHost(hostX)
    
    // Solve system
    result, _ := solvers.ConjugateGradient(A, b, x, 1000, 1e-6)
    
    // Verify solution
    verifySolution(solvers, A, b, x, result, "CG")
}

func testBiCGSTAB(solvers *IterativeSolvers) {
    fmt.Println("\n🧪 Testing BiCGSTAB:")
    
    n := 1000
    nnz := n * 7  // Non-symmetric matrix
    
    // Create non-symmetric test matrix
    A, _ := solvers.sparse.CreateSparseMatrix(n, n, nnz, libraries.MatrixFormatCSR)
    defer A.Destroy()
    
    // Create test vectors
    b, _ := memory.Alloc(int64(n * 4))
    x, _ := memory.Alloc(int64(n * 4))
    defer b.Free()
    defer x.Free()
    
    // Initialize
    hostB := make([]float32, n)
    hostX := make([]float32, n)
    for i := range hostB {
        hostB[i] = float32(i%10 + 1)  // Varied RHS
        hostX[i] = 0.0
    }
    b.CopyFromHost(hostB)
    x.CopyFromHost(hostX)
    
    // Solve system
    result, _ := solvers.BiCGSTAB(A, b, x, 1000, 1e-6)
    
    // Verify solution
    verifySolution(solvers, A, b, x, result, "BiCGSTAB")
}

func verifySolution(solvers *IterativeSolvers, A *libraries.SparseMatrix, 
                   b, x *memory.DeviceMemory, result *CGResult, method string) {
    n := A.GetRows()
    
    // Compute residual r = b - A*x
    Ax, _ := memory.Alloc(int64(n * 4))
    defer Ax.Free()
    
    solvers.sparse.SpMV(1.0, A, x, 0.0, Ax)  // Ax = A*x
    
    r, _ := memory.Alloc(int64(n * 4))
    defer r.Free()
    solvers.vectorSubtract(b, Ax, r, n)  // r = b - Ax
    
    residualNorm, _ := solvers.vectorNorm(r, n)
    
    fmt.Printf("  %s Verification:\n", method)
    fmt.Printf("    Reported final residual: %.6e\n", result.FinalResidual)
    fmt.Printf("    Computed final residual: %.6e\n", residualNorm)
    fmt.Printf("    Relative error: %.2e\n", 
               math.Abs(residualNorm-result.FinalResidual)/result.FinalResidual)
    
    if math.Abs(residualNorm-result.FinalResidual)/result.FinalResidual < 1e-3 {
        fmt.Println("    ✅ Verification passed")
    } else {
        fmt.Println("    ⚠️ Verification shows discrepancy")
    }
}

func compareIterativeSolvers(solvers *IterativeSolvers) {
    fmt.Println("\n📊 Iterative Solver Performance Comparison:")
    
    sizes := []int{500, 1000, 2000}
    
    fmt.Println("\nProblem Size\tCG Time\t\tCG Iters\tBiCGSTAB Time\tBiCGSTAB Iters")
    
    for _, n := range sizes {
        cgTime, cgIters := benchmarkCG(solvers, n)
        bicgTime, bicgIters := benchmarkBiCGSTAB(solvers, n)
        
        fmt.Printf("%d\t\t%v\t%d\t\t%v\t%d\n", 
                   n, cgTime, cgIters, bicgTime, bicgIters)
    }
    
    fmt.Println("\n💡 Solver Selection Guidelines:")
    fmt.Println("  - Symmetric Positive Definite: Use CG (fastest)")
    fmt.Println("  - Symmetric Indefinite: Use MinRes or CG with preconditioning")
    fmt.Println("  - Nonsymmetric: Use BiCGSTAB, GMRES, or QMR")
    fmt.Println("  - Ill-conditioned: Use preconditioning (ILU, AMG)")
    fmt.Println("  - Very large systems: Consider multigrid methods")
}

func benchmarkCG(solvers *IterativeSolvers, n int) (time.Duration, int) {
    // Create and solve test problem
    nnz := n * 5
    A, _ := solvers.sparse.CreateSparseMatrix(n, n, nnz, libraries.MatrixFormatCSR)
    defer A.Destroy()
    
    b, _ := memory.Alloc(int64(n * 4))
    x, _ := memory.Alloc(int64(n * 4))
    defer b.Free()
    defer x.Free()
    
    // Initialize
    hostB := make([]float32, n)
    hostX := make([]float32, n)
    for i := range hostB {
        hostB[i] = 1.0
        hostX[i] = 0.0
    }
    b.CopyFromHost(hostB)
    x.CopyFromHost(hostX)
    
    result, _ := solvers.ConjugateGradient(A, b, x, 1000, 1e-6)
    return result.SolveTime, result.Iterations
}

func benchmarkBiCGSTAB(solvers *IterativeSolvers, n int) (time.Duration, int) {
    // Create and solve test problem
    nnz := n * 7
    A, _ := solvers.sparse.CreateSparseMatrix(n, n, nnz, libraries.MatrixFormatCSR)
    defer A.Destroy()
    
    b, _ := memory.Alloc(int64(n * 4))
    x, _ := memory.Alloc(int64(n * 4))
    defer b.Free()
    defer x.Free()
    
    // Initialize
    hostB := make([]float32, n)
    hostX := make([]float32, n)
    for i := range hostB {
        hostB[i] = float32(i%10 + 1)
        hostX[i] = 0.0
    }
    b.CopyFromHost(hostB)
    x.CopyFromHost(hostX)
    
    result, _ := solvers.BiCGSTAB(A, b, x, 1000, 1e-6)
    return result.SolveTime, result.Iterations
}
```

---

## 🎯 Module Assessment

### **Knowledge Validation**

1. **Format Selection**: Choose optimal sparse format for different sparsity patterns
2. **Iterative Methods**: Implement CG and BiCGSTAB from scratch
3. **Performance Analysis**: Identify bottlenecks in sparse computations
4. **Convergence Analysis**: Understand and control solver convergence

### **Practical Challenge**

Implement a complete sparse linear system solver for:
- **Engineering**: Finite element structural analysis
- **Graph Theory**: PageRank algorithm implementation  
- **Optimization**: Interior point method for linear programming
- **Physics**: Electromagnetic field simulation

### **Success Criteria**

- ✅ Solve systems with >1M unknowns efficiently
- ✅ Achieve convergence in <500 iterations for well-conditioned problems
- ✅ Memory usage <3x matrix storage for iterative solvers
- ✅ Performance within 80% of optimized libraries

---

## 🚀 Next Steps

**Congratulations! You've mastered sparse computing fundamentals.**

**You're now ready for:**
➡️ **[Module 3: Linear Algebra](TRAINING_INTERMEDIATE_3_LINEAR.md)**

**Skills Mastered:**
- 🕸️ **Sparse Matrix Expertise** - All major formats and operations
- ⚡ **Iterative Solver Mastery** - CG, BiCGSTAB, and convergence analysis
- 📊 **Performance Optimization** - Format selection and bottleneck elimination
- 🔗 **Real-World Applications** - Graph algorithms and large-scale systems

---

*From sparse novice to iterative solver expert - the foundation of large-scale computing! 🕸️⚡*
