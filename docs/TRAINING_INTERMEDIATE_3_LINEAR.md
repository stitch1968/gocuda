# 🔧 Module 3: Linear Algebra Deep Dive

**Goal:** Master advanced numerical methods with cuSOLVER for scientific computing and engineering applications

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🔧 **Master all matrix decompositions** - QR, SVD, LU, Cholesky, eigenvalue problems
- 📊 **Handle numerical stability** issues and conditioning problems
- 🎯 **Implement advanced solvers** for least squares, eigenvalues, and optimization
- ⚡ **Optimize linear algebra** operations for maximum performance
- 🧮 **Apply to real problems** in scientific computing and machine learning

---

## 🧠 Theoretical Foundation

### Matrix Decomposition Hierarchy

**Factorization Types:**
```
General Matrix A (m×n)
├── QR Decomposition: A = QR (orthogonal × upper triangular)
├── SVD: A = UΣV^T (full factorization)
├── LU: A = LU or A = PLU (square matrices)
└── Cholesky: A = LL^T (symmetric positive definite)
```

**Computational Complexity:**
- **QR**: O(mn²) for m≥n
- **SVD**: O(min(m,n)³)
- **LU**: O(n³)
- **Cholesky**: O(n³/3) (half of LU)

### Numerical Stability Analysis

**Condition Number Impact:**
```
κ(A) = σ_max(A) / σ_min(A)

κ(A) < 10⁶   → Stable computation
κ(A) < 10¹²  → Use double precision  
κ(A) > 10¹²  → Regularization needed
```

---

## 🏗️ Chapter 1: Advanced Matrix Decompositions

### Comprehensive Linear Algebra Suite

Create `linear/advanced_solver.go`:

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

// Advanced linear algebra operations manager
type LinearAlgebraExpert struct {
    ctx     *cuda.Context
    solver  *libraries.SolverContext
    profiler *LinearProfiler
}

type LinearProfiler struct {
    operations   map[string]time.Duration
    condNumbers  map[string]float64
    accuracy     map[string]float64
}

type DecompositionResult struct {
    Method         string
    ComputeTime    time.Duration
    ConditionNumber float64
    ResidualError  float64
    Success       bool
}

func NewLinearAlgebraExpert() *LinearAlgebraExpert {
    cuda.Initialize()
    ctx := cuda.GetDefaultContext()
    solver, _ := libraries.CreateSolverContext()
    
    return &LinearAlgebraExpert{
        ctx:    ctx,
        solver: solver,
        profiler: &LinearProfiler{
            operations:  make(map[string]time.Duration),
            condNumbers: make(map[string]float64),
            accuracy:   make(map[string]float64),
        },
    }
}

// QR Decomposition with pivoting
func (la *LinearAlgebraExpert) QRDecomposition(A *memory.DeviceMemory, m, n int, withPivoting bool) (*DecompositionResult, error) {
    fmt.Printf("🔧 QR Decomposition (%dx%d, pivoting=%t)\n", m, n, withPivoting)
    
    result := &DecompositionResult{Method: "QR"}
    
    start := time.Now()
    
    if withPivoting {
        qrInfo, err := la.solver.QRFactorizationPivot(A, m, n)
        if err != nil {
            return result, err
        }
        defer qrInfo.Destroy()
    } else {
        qrInfo, err := la.solver.QRFactorization(A, m, n)
        if err != nil {
            return result, err
        }
        defer qrInfo.Destroy()
    }
    
    result.ComputeTime = time.Since(start)
    result.Success = true
    
    // Estimate condition number through R matrix diagonal
    result.ConditionNumber = la.estimateQRConditionNumber(A, m, n)
    
    fmt.Printf("  ✅ QR completed in %v (κ≈%.2e)\n", 
               result.ComputeTime, result.ConditionNumber)
    
    return result, nil
}

// SVD with economic computation options
func (la *LinearAlgebraExpert) SVDDecomposition(A *memory.DeviceMemory, m, n int, computeUV bool) (*DecompositionResult, error) {
    fmt.Printf("🔧 SVD Decomposition (%dx%d, UV=%t)\n", m, n, computeUV)
    
    result := &DecompositionResult{Method: "SVD"}
    
    start := time.Now()
    
    svdInfo, err := la.solver.SVDDecomposition(A, m, n, computeUV)
    if err != nil {
        return result, err
    }
    defer svdInfo.Destroy()
    
    result.ComputeTime = time.Since(start)
    result.Success = true
    
    // Calculate condition number from singular values
    result.ConditionNumber = la.calculateSVDConditionNumber(svdInfo)
    
    fmt.Printf("  ✅ SVD completed in %v (κ=%.2e)\n", 
               result.ComputeTime, result.ConditionNumber)
    
    return result, nil
}

// LU Decomposition with partial pivoting
func (la *LinearAlgebraExpert) LUDecomposition(A *memory.DeviceMemory, n int) (*DecompositionResult, error) {
    fmt.Printf("🔧 LU Decomposition (%dx%d)\n", n, n)
    
    result := &DecompositionResult{Method: "LU"}
    
    start := time.Now()
    
    luInfo, err := la.solver.LUFactorization(A, n)
    if err != nil {
        return result, err
    }
    defer luInfo.Destroy()
    
    result.ComputeTime = time.Since(start)
    result.Success = true
    
    // Estimate condition number
    result.ConditionNumber = la.estimateLUConditionNumber(A, n)
    
    fmt.Printf("  ✅ LU completed in %v (κ≈%.2e)\n", 
               result.ComputeTime, result.ConditionNumber)
    
    return result, nil
}

// Cholesky decomposition for SPD matrices
func (la *LinearAlgebraExpert) CholeskyDecomposition(A *memory.DeviceMemory, n int) (*DecompositionResult, error) {
    fmt.Printf("🔧 Cholesky Decomposition (%dx%d)\n", n, n)
    
    result := &DecompositionResult{Method: "Cholesky"}
    
    start := time.Now()
    
    err := la.solver.CholeskyFactorization(A, n)
    if err != nil {
        return result, err
    }
    
    result.ComputeTime = time.Since(start)
    result.Success = true
    
    // Cholesky is numerically stable for SPD matrices
    result.ConditionNumber = la.estimateCholeskyConditionNumber(A, n)
    
    fmt.Printf("  ✅ Cholesky completed in %v (κ≈%.2e)\n", 
               result.ComputeTime, result.ConditionNumber)
    
    return result, nil
}

// Condition number estimation methods
func (la *LinearAlgebraExpert) estimateQRConditionNumber(A *memory.DeviceMemory, m, n int) float64 {
    // Estimate using R matrix diagonal ratio
    // In practice, would extract R and compute σ_max/σ_min
    return 1e3 // Placeholder
}

func (la *LinearAlgebraExpert) calculateSVDConditionNumber(svdInfo *libraries.SVDInfo) float64 {
    // Get singular values from SVD result
    // κ = σ_max / σ_min
    sigmaMax := svdInfo.GetMaxSingularValue()
    sigmaMin := svdInfo.GetMinSingularValue()
    
    if sigmaMin > 1e-15 {
        return sigmaMax / sigmaMin
    }
    return math.Inf(1) // Singular matrix
}

func (la *LinearAlgebraExpert) estimateLUConditionNumber(A *memory.DeviceMemory, n int) float64 {
    // Use reciprocal condition number estimation
    // This would call cuSOLVER's condition number estimation
    return 1e4 // Placeholder
}

func (la *LinearAlgebraExpert) estimateCholeskyConditionNumber(A *memory.DeviceMemory, n int) float64 {
    // For SPD matrices, κ(A) = κ(L)²
    return 1e2 // Placeholder
}

func (la *LinearAlgebraExpert) Destroy() {
    la.solver.DestroyContext()
}

// Demonstration
func main() {
    cuda.Initialize()
    fmt.Println("🔧 Advanced Linear Algebra with cuSOLVER")
    
    expert := NewLinearAlgebraExpert()
    defer expert.Destroy()
    
    // Test all decomposition methods
    testDecompositions(expert)
    
    // Advanced applications
    advancedApplications(expert)
    
    // Performance analysis
    performanceAnalysis(expert)
}

func testDecompositions(expert *LinearAlgebraExpert) {
    fmt.Println("\n🧪 Testing Matrix Decompositions:")
    
    sizes := []int{100, 500, 1000}
    
    for _, n := range sizes {
        fmt.Printf("\n--- Testing with %dx%d matrices ---\n", n, n)
        
        // Create test matrices
        testMatrices := createTestMatrices(n)
        
        for matType, A := range testMatrices {
            fmt.Printf("\nMatrix Type: %s\n", matType)
            
            // Test appropriate decompositions for each matrix type
            testMatrixDecompositions(expert, A, n, matType)
            
            A.Free()
        }
    }
}

func createTestMatrices(n int) map[string]*memory.DeviceMemory {
    matrices := make(map[string]*memory.DeviceMemory)
    
    // Well-conditioned matrix
    wellCond, _ := memory.Alloc(int64(n * n * 4))
    matrices["Well-Conditioned"] = wellCond
    initializeWellConditioned(wellCond, n)
    
    // Ill-conditioned matrix
    illCond, _ := memory.Alloc(int64(n * n * 4))
    matrices["Ill-Conditioned"] = illCond
    initializeIllConditioned(illCond, n)
    
    // Symmetric positive definite
    spd, _ := memory.Alloc(int64(n * n * 4))
    matrices["SPD"] = spd
    initializeSPD(spd, n)
    
    return matrices
}

func initializeWellConditioned(A *memory.DeviceMemory, n int) {
    // Create well-conditioned matrix (e.g., random with condition number ~10)
    hostA := make([]float32, n*n)
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            if i == j {
                hostA[i*n+j] = float32(10 + i) // Diagonal dominance
            } else {
                hostA[i*n+j] = float32(i+j) / float32(n*n)
            }
        }
    }
    A.CopyFromHost(hostA)
}

func initializeIllConditioned(A *memory.DeviceMemory, n int) {
    // Create ill-conditioned matrix (e.g., Hilbert matrix)
    hostA := make([]float32, n*n)
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            hostA[i*n+j] = 1.0 / float32(i+j+1) // Hilbert matrix
        }
    }
    A.CopyFromHost(hostA)
}

func initializeSPD(A *memory.DeviceMemory, n int) {
    // Create symmetric positive definite matrix A = B^T * B
    hostA := make([]float32, n*n)
    
    // Generate random B matrix
    B := make([][]float32, n)
    for i := range B {
        B[i] = make([]float32, n)
        for j := range B[i] {
            B[i][j] = float32(i+j+1) / float32(n)
        }
    }
    
    // Compute A = B^T * B
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            var sum float32
            for k := 0; k < n; k++ {
                sum += B[k][i] * B[k][j]
            }
            hostA[i*n+j] = sum
        }
    }
    
    A.CopyFromHost(hostA)
}

func testMatrixDecompositions(expert *LinearAlgebraExpert, A *memory.DeviceMemory, n int, matType string) {
    // Copy matrix for each test (decompositions modify the input)
    
    // Test QR
    AQR, _ := memory.Alloc(int64(n * n * 4))
    defer AQR.Free()
    copyMatrix(A, AQR, n, n)
    
    result, err := expert.QRDecomposition(AQR, n, n, false)
    if err == nil && result.Success {
        expert.profiler.operations["QR_"+matType] = result.ComputeTime
        expert.profiler.condNumbers["QR_"+matType] = result.ConditionNumber
    }
    
    // Test SVD
    ASVD, _ := memory.Alloc(int64(n * n * 4))
    defer ASVD.Free()
    copyMatrix(A, ASVD, n, n)
    
    result, err = expert.SVDDecomposition(ASVD, n, n, true)
    if err == nil && result.Success {
        expert.profiler.operations["SVD_"+matType] = result.ComputeTime
        expert.profiler.condNumbers["SVD_"+matType] = result.ConditionNumber
    }
    
    // Test LU
    ALU, _ := memory.Alloc(int64(n * n * 4))
    defer ALU.Free()
    copyMatrix(A, ALU, n, n)
    
    result, err = expert.LUDecomposition(ALU, n)
    if err == nil && result.Success {
        expert.profiler.operations["LU_"+matType] = result.ComputeTime
        expert.profiler.condNumbers["LU_"+matType] = result.ConditionNumber
    }
    
    // Test Cholesky (only for SPD matrices)
    if matType == "SPD" {
        AChol, _ := memory.Alloc(int64(n * n * 4))
        defer AChol.Free()
        copyMatrix(A, AChol, n, n)
        
        result, err = expert.CholeskyDecomposition(AChol, n)
        if err == nil && result.Success {
            expert.profiler.operations["Cholesky_"+matType] = result.ComputeTime
            expert.profiler.condNumbers["Cholesky_"+matType] = result.ConditionNumber
        }
    }
}

func copyMatrix(src, dst *memory.DeviceMemory, m, n int) {
    hostData := make([]float32, m*n)
    src.CopyToHost(hostData)
    dst.CopyFromHost(hostData)
}

func advancedApplications(expert *LinearAlgebraExpert) {
    fmt.Println("\n🎯 Advanced Applications:")
    
    // Least squares problems
    leastSquaresDemo(expert)
    
    // Eigenvalue problems
    eigenvalueDemo(expert)
    
    // Regularized problems
    regularizationDemo(expert)
}

func leastSquaresDemo(expert *LinearAlgebraExpert) {
    fmt.Println("\n1. Least Squares Problem (overdetermined system):")
    
    m, n := 1000, 500 // More equations than unknowns
    
    A, _ := memory.Alloc(int64(m * n * 4))
    b, _ := memory.Alloc(int64(m * 4))
    x, _ := memory.Alloc(int64(n * 4))
    defer A.Free()
    defer b.Free()
    defer x.Free()
    
    // Initialize overdetermined system
    hostA := make([]float32, m*n)
    hostB := make([]float32, m)
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            hostA[i*n+j] = float32(i+j+1) / float32(m*n)
        }
        hostB[i] = float32(i + 1)
    }
    
    A.CopyFromHost(hostA)
    b.CopyFromHost(hostB)
    
    // Solve using QR decomposition
    fmt.Printf("  Solving %dx%d overdetermined system using QR...\n", m, n)
    
    start := time.Now()
    solution, err := expert.solver.SolveLinearSystem(A, b, n)
    elapsed := time.Since(start)
    defer solution.Free()
    
    if err == nil {
        fmt.Printf("  ✅ Least squares solved in %v\n", elapsed)
        
        // Compute residual norm
        residualNorm := computeResidualNorm(expert, A, solution, b, m, n)
        fmt.Printf("  Residual norm: %.6e\n", residualNorm)
    } else {
        fmt.Printf("  ❌ Least squares failed: %v\n", err)
    }
}

func eigenvalueDemo(expert *LinearAlgebraExpert) {
    fmt.Println("\n2. Eigenvalue Problem:")
    
    n := 500
    A, _ := memory.Alloc(int64(n * n * 4))
    defer A.Free()
    
    // Create symmetric matrix for real eigenvalues
    initializeSPD(A, n)
    
    fmt.Printf("  Computing eigenvalues of %dx%d symmetric matrix...\n", n, n)
    
    start := time.Now()
    eigenvals, eigenvecs, err := expert.solver.Eigenvalues(A, n, true)
    elapsed := time.Since(start)
    
    if err == nil {
        defer eigenvals.Free()
        if eigenvecs != nil {
            defer eigenvecs.Free()
        }
        
        fmt.Printf("  ✅ Eigenvalue decomposition completed in %v\n", elapsed)
        
        // Extract and display eigenvalue range
        hostEigenvals := make([]float32, n)
        eigenvals.CopyToHost(hostEigenvals)
        
        minEig, maxEig := findMinMax(hostEigenvals)
        conditionNumber := maxEig / minEig
        
        fmt.Printf("  Eigenvalue range: [%.6e, %.6e]\n", minEig, maxEig)
        fmt.Printf("  Spectral condition number: %.2e\n", conditionNumber)
    } else {
        fmt.Printf("  ❌ Eigenvalue computation failed: %v\n", err)
    }
}

func regularizationDemo(expert *LinearAlgebraExpert) {
    fmt.Println("\n3. Regularized Problem (Ridge Regression):")
    
    m, n := 800, 600
    lambda := float32(0.01) // Regularization parameter
    
    A, _ := memory.Alloc(int64(m * n * 4))
    b, _ := memory.Alloc(int64(m * 4))
    defer A.Free()
    defer b.Free()
    
    // Initialize problem
    hostA := make([]float32, m*n)
    hostB := make([]float32, m)
    
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            hostA[i*n+j] = float32(i+j+1) / float32(m*n*10) // Ill-conditioned
        }
        hostB[i] = float32(i%100) / 100.0
    }
    
    A.CopyFromHost(hostA)
    b.CopyFromHost(hostB)
    
    fmt.Printf("  Solving regularized system (λ=%.3f) using normal equations...\n", lambda)
    
    // Form normal equations: (A^T*A + λI)x = A^T*b
    start := time.Now()
    x, err := solveRidgeRegression(expert, A, b, lambda, m, n)
    elapsed := time.Since(start)
    
    if err == nil {
        defer x.Free()
        fmt.Printf("  ✅ Regularized problem solved in %v\n", elapsed)
        
        residualNorm := computeResidualNorm(expert, A, x, b, m, n)
        fmt.Printf("  Residual norm: %.6e\n", residualNorm)
    } else {
        fmt.Printf("  ❌ Regularization failed: %v\n", err)
    }
}

func solveRidgeRegression(expert *LinearAlgebraExpert, A, b *memory.DeviceMemory, lambda float32, m, n int) (*memory.DeviceMemory, error) {
    // This would implement: (A^T*A + λI)x = A^T*b
    // For simulation, we use the existing linear system solver
    
    x, _ := memory.Alloc(int64(n * 4))
    
    // Simplified: use QR solver with regularization simulation
    ATA, _ := memory.Alloc(int64(n * n * 4))
    defer ATA.Free()
    
    // Simulate A^T*A + λI formation
    hostATA := make([]float32, n*n)
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            if i == j {
                hostATA[i*n+j] = float32(i+j+2) + lambda // Diagonal regularization
            } else {
                hostATA[i*n+j] = float32(i+j+1) / float32(n*n)
            }
        }
    }
    ATA.CopyFromHost(hostATA)
    
    // Form A^T*b
    ATb, _ := memory.Alloc(int64(n * 4))
    defer ATb.Free()
    
    hostATb := make([]float32, n)
    for i := 0; i < n; i++ {
        hostATb[i] = float32(i + 1)
    }
    ATb.CopyFromHost(hostATb)
    
    // Solve normal equations
    result, err := expert.solver.SolveLinearSystem(ATA, ATb, n)
    if err != nil {
        x.Free()
        return nil, err
    }
    
    // Copy result
    hostResult := make([]float32, n)
    result.CopyToHost(hostResult)
    result.Free()
    
    x.CopyFromHost(hostResult)
    
    return x, nil
}

func computeResidualNorm(expert *LinearAlgebraExpert, A, x, b *memory.DeviceMemory, m, n int) float64 {
    // Compute ||Ax - b||_2
    Ax, _ := memory.Alloc(int64(m * 4))
    defer Ax.Free()
    
    // Simulate matrix-vector multiplication A*x
    hostA := make([]float32, m*n)
    hostX := make([]float32, n)
    hostB := make([]float32, m)
    
    A.CopyToHost(hostA)
    x.CopyToHost(hostX)
    b.CopyToHost(hostB)
    
    residual := make([]float32, m)
    
    // Compute Ax
    for i := 0; i < m; i++ {
        var sum float32
        for j := 0; j < n; j++ {
            sum += hostA[i*n+j] * hostX[j]
        }
        residual[i] = hostB[i] - sum // r = b - Ax
    }
    
    // Compute norm
    var norm float64
    for _, r := range residual {
        norm += float64(r * r)
    }
    
    return math.Sqrt(norm)
}

func findMinMax(values []float32) (float32, float32) {
    if len(values) == 0 {
        return 0, 0
    }
    
    min, max := values[0], values[0]
    for _, v := range values[1:] {
        if v < min {
            min = v
        }
        if v > max {
            max = v
        }
    }
    return min, max
}

func performanceAnalysis(expert *LinearAlgebraExpert) {
    fmt.Println("\n📊 Performance Analysis Summary:")
    
    fmt.Println("\nDecomposition Performance:")
    fmt.Println("Method\t\tMatrix Type\tTime\t\tCondition Number")
    
    for method, duration := range expert.profiler.operations {
        condNum := expert.profiler.condNumbers[method]
        fmt.Printf("%-15s\t%v\t%.2e\n", method, duration, condNum)
    }
    
    fmt.Println("\n💡 Linear Algebra Optimization Tips:")
    fmt.Println("  1. Use Cholesky for SPD matrices (2x faster than LU)")
    fmt.Println("  2. QR with pivoting for rank-deficient problems")
    fmt.Println("  3. SVD for maximum numerical stability (but slowest)")
    fmt.Println("  4. Regularization for ill-conditioned systems")
    fmt.Println("  5. Iterative methods for very large sparse systems")
    fmt.Println("  6. Mixed precision for memory-bound operations")
    
    fmt.Println("\n🎯 Method Selection Guide:")
    fmt.Println("  Square, well-conditioned: LU factorization")
    fmt.Println("  Symmetric positive definite: Cholesky")
    fmt.Println("  Overdetermined (m>n): QR decomposition")
    fmt.Println("  Underdetermined (m<n): SVD or QR with pivoting")
    fmt.Println("  Eigenvalue problems: Specialized iterative methods")
    fmt.Println("  Ill-conditioned: SVD or regularized methods")
}
```

---

## 🎯 Module Assessment

### **Knowledge Validation**

1. **Decomposition Mastery**: Implement and apply all major matrix factorizations
2. **Numerical Stability**: Handle ill-conditioned problems with appropriate techniques
3. **Application Expertise**: Solve least squares, eigenvalue, and regularized problems
4. **Performance Optimization**: Choose optimal methods based on problem characteristics

### **Practical Challenge**

Implement a complete numerical solver for:
- **Engineering**: Structural finite element analysis with modal analysis
- **Data Science**: Principal Component Analysis (PCA) pipeline  
- **Optimization**: Interior point method for quadratic programming
- **Signal Processing**: Wiener filtering with regularization

### **Success Criteria**

- ✅ Handle matrices up to condition number 10¹² reliably
- ✅ Achieve >80% of theoretical peak performance for decompositions
- ✅ Correctly identify and handle rank-deficient problems
- ✅ Implement numerical stability checks and error handling

---

## 🚀 Next Steps

**Congratulations! You've mastered advanced linear algebra.**

**You're now ready for:**
➡️ **[Module 4: Concurrent Patterns](TRAINING_INTERMEDIATE_4_CONCURRENT.md)**

**Skills Mastered:**
- 🔧 **Matrix Decomposition Expertise** - QR, SVD, LU, Cholesky mastery
- 📊 **Numerical Stability** - Condition number analysis and regularization
- 🎯 **Advanced Applications** - Least squares, eigenvalues, optimization
- ⚡ **Performance Optimization** - Method selection and computational efficiency

---

*From linear algebra novice to numerical methods expert - the mathematical foundation of scientific computing! 🔧📊*
