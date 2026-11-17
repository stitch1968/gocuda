package libraries

import (
	"fmt"
	"math"
	"testing"
)

func TestSparseSolveTranspose(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	lowerCSR := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 4}, []int32{0, 0, 0, 1}, []float32{1, 1, 1, 3})
	upperCSR := sparseTestCSR(t, ctx, 2, 2, []int32{0, 3, 4}, []int32{1, 0, 1, 1}, []float32{0.25, 2, 0.75, 3})
	lowerCOO, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 1, 0}, []int32{1, 0, 0, 0}, []float32{3, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer lowerCOO.Destroy()
	upperCOO, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0, 0}, []int32{1, 1, 0, 1}, []float32{3, 0.25, 2, 0.75})
	if err != nil {
		t.Fatal(err)
	}
	defer upperCOO.Destroy()
	for _, tc := range []struct {
		name   string
		matrix *SparseMatrix
		rhs    []float32
	}{
		{"lower-csr", lowerCSR, []float32{4, 6}},
		{"upper-csr", upperCSR, []float32{2, 7}},
		{"lower-coo", lowerCOO, []float32{4, 6}},
		{"upper-coo", upperCOO, []float32{2, 7}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rhs := solverTestBuffer(t, tc.rhs)
			output := solverTestBuffer(t, []float32{0, 0, 99})
			if err := ctx.SpSVEx(tc.matrix, rhs, output, BLASOpT); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, 3)
			if err != nil {
				t.Fatal(err)
			}
			for i, want := range []float32{1, 2, 99} {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want)) > 1e-5 {
					t.Fatalf("transpose solution[%d] = %v, want %v", i, got[i], want)
				}
			}
		})
	}
}

func TestSparseSolveTransposeRejectsInvalidInputs(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	notTriangular := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 4}, []int32{0, 1, 0, 1}, []float32{2, 1, 1, 3})
	missingDiagonal := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 1}, []int32{0}, []float32{2})
	nonsquare := sparseTestCSR(t, ctx, 1, 2, []int32{0, 1}, []int32{0}, []float32{1})
	rhs := solverTestBuffer(t, []float32{2, 7})
	output := solverTestBuffer(t, []float32{41, 42})
	short := solverTestBuffer(t, []float32{1})
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"invalid transpose", func() error { return ctx.SpSVEx(matrix, rhs, output, BLASOperation(-1)) }},
		{"non-square", func() error { return ctx.SpSVEx(nonsquare, rhs, output, BLASOpT) }},
		{"not triangular", func() error { return ctx.SpSVEx(notTriangular, rhs, output, BLASOpT) }},
		{"missing diagonal", func() error { return ctx.SpSVEx(missingDiagonal, rhs, output, BLASOpT) }},
		{"short input", func() error { return ctx.SpSVEx(matrix, short, output, BLASOpT) }},
		{"short output", func() error { return ctx.SpSVEx(matrix, rhs, short, BLASOpT) }},
		{"nil input", func() error { return ctx.SpSVEx(matrix, nil, output, BLASOpT) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid input to fail")
			}
		})
	}
	got, err := readMathFloat32Memory(output, 2)
	if err != nil || len(got) != 2 || got[0] != 41 || got[1] != 42 {
		t.Fatalf("invalid input changed output: %v, error = %v", got, err)
	}
}

func TestSparseSolveTransposeSimulationAlias(t *testing.T) {
	ctx := &SparseContext{}
	matrix := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	vector := solverTestBuffer(t, []float32{2, 7})
	if err := ctx.SpSVEx(matrix, vector, vector, BLASOpT); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(vector, 2)
	if err != nil || len(got) != 2 || got[0] != 1 || got[1] != 2 {
		t.Fatalf("simulation aliased transpose solution = %v, error = %v", got, err)
	}
}

func TestSparseMVZeroBeta(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	for _, tc := range []struct {
		op    BLASOperation
		input []float32
		want  []float32
	}{
		{BLASOpN, []float32{1, 3, 5}, []float32{11, 9}},
		{BLASOpT, []float32{4, 5}, []float32{4, 15, 8}},
	} {
		t.Run(fmt.Sprint(tc.op), func(t *testing.T) {
			input := solverTestBuffer(t, tc.input)
			initial := make([]float32, len(tc.want))
			for i := range initial {
				initial[i] = float32(math.NaN())
			}
			output := solverTestBuffer(t, initial)
			if err := ctx.SpMVEx(1, matrix, input, 0, output, tc.op); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, len(tc.want))
			if err != nil {
				t.Fatal(err)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("zero-beta output[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestSparseMVTranspose(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	csr := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{2, 0, 2, 1}, []float32{1, 1, 1, 3})
	coo, err := ctx.CreateCOOMatrix(2, 3, []int32{1, 0, 0, 0}, []int32{1, 2, 0, 2}, []float32{3, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	emptyCSR := sparseTestCSR(t, ctx, 2, 3, []int32{0, 0, 0}, nil, nil)
	emptyCOO, err := ctx.CreateCOOMatrix(2, 3, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyCOO.Destroy()
	input := solverTestBuffer(t, []float32{4, 5})
	for _, tc := range []struct {
		name   string
		matrix *SparseMatrix
		want   []float32
	}{
		{"csr-duplicates", csr, []float32{7, 28, 13, 99}},
		{"coo-unsorted-duplicates", coo, []float32{7, 28, 13, 99}},
		{"empty-csr", emptyCSR, []float32{-1, -2, -3, 99}},
		{"empty-coo", emptyCOO, []float32{-1, -2, -3, 99}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := solverTestBuffer(t, []float32{1, 2, 3, 99})
			if err := ctx.SpMVEx(2, tc.matrix, input, -1, output, BLASOpT); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, len(tc.want))
			if err != nil {
				t.Fatal(err)
			}
			for i := range tc.want {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-tc.want[i])) > 1e-5 {
					t.Fatalf("transpose output[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestSparseMVTransposeRejectsInvalidInputs(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	input := solverTestBuffer(t, []float32{4, 5})
	output := solverTestBuffer(t, []float32{1, 2, 3})
	shortInput := solverTestBuffer(t, []float32{4})
	shortOutput := solverTestBuffer(t, []float32{1, 2})
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"invalid transpose", func() error { return ctx.SpMVEx(1, matrix, input, 0, output, BLASOperation(-1)) }},
		{"short input", func() error { return ctx.SpMVEx(1, matrix, shortInput, 0, output, BLASOpT) }},
		{"short output", func() error { return ctx.SpMVEx(1, matrix, input, 0, shortOutput, BLASOpT) }},
		{"nil input", func() error { return ctx.SpMVEx(1, matrix, nil, 0, output, BLASOpT) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid input to fail")
			}
		})
	}
	got, err := readMathFloat32Memory(output, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, 2, 3} {
		if got[i] != want {
			t.Fatalf("invalid input changed output[%d] = %v", i, got[i])
		}
	}
}

func TestSparseMVTransposeSimulationAlias(t *testing.T) {
	ctx := &SparseContext{}
	matrix := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	vector := solverTestBuffer(t, []float32{1, 2})
	if err := ctx.SpMVEx(1, matrix, vector, 0, vector, BLASOpT); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(vector, 2)
	if err != nil || len(got) != 2 || got[0] != 2 || got[1] != 7 {
		t.Fatalf("simulation aliased transpose output = %v, error = %v", got, err)
	}
}

func TestSparseDenseMultiplicationBatched(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	csr := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{2, 0, 2, 1}, []float32{1, 1, 1, 3})
	coo, err := ctx.CreateCOOMatrix(2, 3, []int32{1, 0, 0, 0}, []int32{1, 2, 0, 2}, []float32{3, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	for _, matrix := range []*SparseMatrix{csr, coo} {
		for _, opA := range []BLASOperation{BLASOpN, BLASOpT} {
			for _, opB := range []BLASOperation{BLASOpN, BLASOpT} {
				t.Run(fmt.Sprintf("%d-%d-%d", matrix.format, opA, opB), func(t *testing.T) {
					logicalB := [][][]float32{{{1, 2}, {3, 4}, {5, 6}}, {{2, 0}, {1, -1}, {0, 3}}}
					product := [][][]float32{{{11, 14}, {9, 12}}, {{2, 6}, {3, -3}}}
					outputRows, inner := 2, 3
					if opA == BLASOpT {
						outputRows, inner = 3, 2
						product = [][][]float32{{{1, 2}, {9, 12}, {2, 4}}, {{2, 0}, {3, -3}, {4, 0}}}
					}
					storedRows := inner
					if opB == BLASOpT {
						storedRows = 2
					}
					const ldb, ldc = 4, 4
					strideB, strideC := storedRows*ldb+2, outputRows*ldc+3
					inputData := make([]float32, strideB+storedRows*ldb)
					outputData := make([]float32, strideC+outputRows*ldc+2)
					for i := range inputData {
						inputData[i] = 900
					}
					for i := range outputData {
						outputData[i] = 700
					}
					want := append([]float32(nil), outputData...)
					for b := range 2 {
						for k := range inner {
							for col := range 2 {
								index := b*strideB + k*ldb + col
								if opB == BLASOpT {
									index = b*strideB + col*ldb + k
								}
								inputData[index] = logicalB[b][k][col]
							}
						}
						for row := range outputRows {
							for col := range 2 {
								index := b*strideC + row*ldc + col
								initial := float32(1 + b*10 + row*2 + col)
								outputData[index] = initial
								want[index] = 2*product[b][row][col] - initial
							}
						}
					}
					input := solverTestBuffer(t, inputData)
					output := solverTestBuffer(t, outputData)
					if err := ctx.SpMMDenseBatched(2, matrix, input, 2, ldb, ldc, opA, opB, 2, strideB, strideC, -1, output); err != nil {
						t.Fatal(err)
					}
					got, err := readMathFloat32Memory(output, len(want))
					if err != nil {
						t.Fatal(err)
					}
					for i := range want {
						if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-5 {
							t.Fatalf("batch output[%d] = %v, want %v", i, got[i], want[i])
						}
					}
				})
			}
		}
	}
}

func TestSparseDenseMultiplicationBatchedRejectsInvalidInputs(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	input := solverTestBuffer(t, make([]float32, 24))
	output := solverTestBuffer(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16})
	shortInput := solverTestBuffer(t, make([]float32, 23))
	shortOutput := solverTestBuffer(t, make([]float32, 15))
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"zero batches", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 0, 12, 8, 0, output)
		}},
		{"too many batches", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 65536, 12, 8, 0, output)
		}},
		{"zero stride", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 2, 0, 8, 0, output)
		}},
		{"overlapping inputs", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 2, 11, 8, 0, output)
		}},
		{"overlapping outputs", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 2, 12, 7, 0, output)
		}},
		{"short input", func() error {
			return ctx.SpMMDenseBatched(1, matrix, shortInput, 2, 4, 4, BLASOpN, BLASOpN, 2, 12, 8, 0, output)
		}},
		{"short output", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 2, 12, 8, 0, shortOutput)
		}},
		{"stride overflow", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 2, int(^uint(0)>>1), 8, 0, output)
		}},
		{"aliased output", func() error {
			return ctx.SpMMDenseBatched(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 2, 12, 8, 0, input)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid batch to fail")
			}
		})
	}
	got, err := readMathFloat32Memory(output, 16)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != float32(i+1) {
			t.Fatalf("invalid batch changed output[%d] = %v", i, got[i])
		}
	}
}

func TestSparseDenseMultiplicationBatchedEdgeCases(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	emptyCSR := sparseTestCSR(t, ctx, 2, 3, []int32{0, 0, 0}, nil, nil)
	emptyCOO, err := ctx.CreateCOOMatrix(2, 3, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyCOO.Destroy()
	input := solverTestBuffer(t, []float32{1, 3, 5, 2, 1, 0})
	for _, tc := range []struct {
		name       string
		matrix     *SparseMatrix
		count      int
		beta       float32
		init, want []float32
	}{
		{"single-batch", matrix, 1, 1, []float32{1, 2}, []float32{12, 11}},
		{"single-column", matrix, 2, 0, []float32{1, 2, 3, 4}, []float32{11, 9, 2, 3}},
		{"empty-csr", emptyCSR, 2, -2, []float32{1, 2, 3, 4}, []float32{-2, -4, -6, -8}},
		{"empty-coo", emptyCOO, 2, -2, []float32{1, 2, 3, 4}, []float32{-2, -4, -6, -8}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := solverTestBuffer(t, tc.init)
			if err := ctx.SpMMDenseBatched(1, tc.matrix, input, 1, 1, 1, BLASOpN, BLASOpN, tc.count, 3, 2, tc.beta, output); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, len(tc.want))
			if err != nil {
				t.Fatal(err)
			}
			for i := range tc.want {
				if got[i] != tc.want[i] {
					t.Fatalf("batch output[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestSparseDenseMultiplicationBatchedMaximumCount(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 1, 1, []int32{0, 1}, []int32{0}, []float32{2})
	const count = 65535
	values := make([]float32, count)
	for i := range values {
		values[i] = float32(i % 11)
	}
	input := solverTestBuffer(t, values)
	output := solverTestBuffer(t, make([]float32, count))
	if err := ctx.SpMMDenseBatched(1, matrix, input, 1, 1, 1, BLASOpN, BLASOpN, count, 1, 1, 0, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, count)
	if err != nil {
		t.Fatal(err)
	}
	for i := range values {
		if got[i] != 2*values[i] {
			t.Fatalf("batch %d = %v, want %v", i, got[i], 2*values[i])
		}
	}
}

func TestSparseDenseMultiplicationEx(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	csr := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{2, 0, 2, 1}, []float32{1, 1, 1, 3})
	coo, err := ctx.CreateCOOMatrix(2, 3, []int32{1, 0, 0, 0}, []int32{1, 2, 0, 2}, []float32{3, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	for _, matrix := range []*SparseMatrix{csr, coo} {
		for _, tc := range []struct {
			name        string
			opA, opB    BLASOperation
			input, init []float32
			alpha, beta float32
			want        []float32
		}{
			{"NN", BLASOpN, BLASOpN, []float32{1, 2, 900, 900, 3, 4, 900, 900, 5, 6, 900, 900}, []float32{1, 2, 91, 92, 3, 4, 93, 94}, 2, -1, []float32{21, 26, 91, 92, 15, 20, 93, 94}},
			{"NT", BLASOpN, BLASOpT, []float32{1, 3, 5, 900, 2, 4, 6, 900}, []float32{1, 2, 91, 92, 3, 4, 93, 94}, 2, -1, []float32{21, 26, 91, 92, 15, 20, 93, 94}},
			{"TN", BLASOpT, BLASOpN, []float32{1, 2, 900, 900, 3, 4, 900, 900}, []float32{1, 2, 91, 92, 3, 4, 93, 94, 5, 6, 95, 96}, 2, -1, []float32{1, 2, 91, 92, 15, 20, 93, 94, -1, 2, 95, 96}},
			{"TT", BLASOpT, BLASOpT, []float32{1, 3, 900, 900, 2, 4, 900, 900}, []float32{1, 2, 91, 92, 3, 4, 93, 94, 5, 6, 95, 96}, 2, -1, []float32{1, 2, 91, 92, 15, 20, 93, 94, -1, 2, 95, 96}},
			{"zero-beta", BLASOpN, BLASOpN, []float32{1, 2, 900, 900, 3, 4, 900, 900, 5, 6, 900, 900}, []float32{float32(math.NaN()), float32(math.NaN()), 91, 92, float32(math.NaN()), float32(math.NaN()), 93, 94}, 1, 0, []float32{11, 14, 91, 92, 9, 12, 93, 94}},
		} {
			t.Run(fmt.Sprintf("%d-%s", matrix.format, tc.name), func(t *testing.T) {
				input := solverTestBuffer(t, tc.input)
				output := solverTestBuffer(t, tc.init)
				if err := ctx.SpMMDenseEx(tc.alpha, matrix, input, 2, 4, 4, tc.opA, tc.opB, tc.beta, output); err != nil {
					t.Fatal(err)
				}
				got, err := readMathFloat32Memory(output, len(tc.want))
				if err != nil {
					t.Fatal(err)
				}
				for i := range tc.want {
					if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-tc.want[i])) > 1e-5 {
						t.Fatalf("output[%d] = %v, want %v", i, got[i], tc.want[i])
					}
				}
			})
		}
	}
}

func TestSparseDenseMultiplicationExRejectsInvalidInputs(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	input := solverTestBuffer(t, []float32{1, 2, 0, 0, 3, 4, 0, 0, 5, 6, 0, 0})
	output := solverTestBuffer(t, []float32{1, 2, 91, 92, 3, 4, 93, 94})
	shortInput := solverTestBuffer(t, make([]float32, 11))
	shortOutput := solverTestBuffer(t, make([]float32, 7))
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"invalid sparse transpose", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 4, 4, BLASOperation(-1), BLASOpN, 0, output) }},
		{"invalid dense transpose", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 4, 4, BLASOpN, BLASOperation(2), 0, output) }},
		{"short input stride", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 1, 4, BLASOpN, BLASOpN, 0, output) }},
		{"short transposed input stride", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 2, 4, BLASOpN, BLASOpT, 0, output) }},
		{"short output stride", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 4, 1, BLASOpN, BLASOpN, 0, output) }},
		{"short padded input", func() error { return ctx.SpMMDenseEx(1, matrix, shortInput, 2, 4, 4, BLASOpN, BLASOpN, 0, output) }},
		{"short padded output", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 0, shortOutput) }},
		{"short transposed output", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 4, 4, BLASOpT, BLASOpN, 0, output) }},
		{"aliased output", func() error { return ctx.SpMMDenseEx(1, matrix, input, 2, 4, 4, BLASOpN, BLASOpN, 0, input) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid input to fail")
			}
		})
	}
	got, err := readMathFloat32Memory(output, 8)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, 2, 91, 92, 3, 4, 93, 94} {
		if got[i] != want {
			t.Fatalf("invalid input changed output[%d] = %v", i, got[i])
		}
	}
}

func TestSparseDenseMultiplication(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	csr := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{2, 0, 2, 1}, []float32{1, 1, 1, 3})
	uniqueCSR := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	coo, err := ctx.CreateCOOMatrix(2, 3, []int32{1, 0, 0, 0}, []int32{1, 2, 0, 2}, []float32{3, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	empty := sparseTestCSR(t, ctx, 2, 3, []int32{0, 0, 0}, nil, nil)
	emptyCOO, err := ctx.CreateCOOMatrix(2, 3, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyCOO.Destroy()
	for _, tc := range []struct {
		name        string
		matrix      *SparseMatrix
		n           int
		input, init []float32
		alpha, beta float32
		want        []float32
	}{
		{"csr-rectangular", csr, 2, []float32{1, 2, 3, 4, 5, 6}, []float32{1, 2, 3, 4}, 2, -1, []float32{21, 26, 15, 20}},
		{"coo-rectangular", coo, 2, []float32{1, 2, 3, 4, 5, 6}, []float32{1, 2, 3, 4}, 2, -1, []float32{21, 26, 15, 20}},
		{"csr-three-columns", uniqueCSR, 3, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []float32{1, 1, 1, 1, 1, 1}, 1, 1, []float32{16, 19, 22, 13, 16, 19}},
		{"single-column", coo, 1, []float32{1, 3, 5}, []float32{1, 2}, 2, -1, []float32{21, 16}},
		{"empty", empty, 2, []float32{1, 2, 3, 4, 5, 6}, []float32{1, 2, 3, 4}, 2, -1, []float32{-1, -2, -3, -4}},
		{"empty-coo", emptyCOO, 2, []float32{1, 2, 3, 4, 5, 6}, []float32{1, 2, 3, 4}, 2, -1, []float32{-1, -2, -3, -4}},
		{"zero-beta", csr, 2, []float32{1, 2, 3, 4, 5, 6}, []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN())}, 1, 0, []float32{11, 14, 9, 12}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := solverTestBuffer(t, tc.input)
			output := solverTestBuffer(t, tc.init)
			if err := ctx.SpMMDense(tc.alpha, tc.matrix, input, tc.n, tc.beta, output); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, len(tc.want))
			if err != nil {
				t.Fatal(err)
			}
			for i := range tc.want {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-tc.want[i])) > 1e-5 {
					t.Fatalf("output[%d] = %v, want %v", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestSparseDenseMultiplicationRejectsInvalidInputs(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{2, 0, 2, 1}, []float32{1, 1, 1, 3})
	input := solverTestBuffer(t, []float32{1, 2, 3, 4, 5, 6})
	output := solverTestBuffer(t, []float32{1, 2, 3, 4})
	short := solverTestBuffer(t, []float32{1})
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"zero columns", func() error { return ctx.SpMMDense(1, matrix, input, 0, 0, output) }},
		{"short input", func() error { return ctx.SpMMDense(1, matrix, short, 2, 0, output) }},
		{"short output", func() error { return ctx.SpMMDense(1, matrix, input, 2, 0, short) }},
		{"aliased dense", func() error { return ctx.SpMMDense(1, matrix, input, 2, 0, input) }},
		{"aliased sparse", func() error { return ctx.SpMMDense(1, matrix, input, 2, 0, matrix.values) }},
		{"nil input", func() error { return ctx.SpMMDense(1, matrix, nil, 2, 0, output) }},
		{"excess columns", func() error { return ctx.SpMMDense(1, matrix, input, math.MaxInt32, 0, output) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid input to fail")
			}
		})
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, 2, 3, 4} {
		if got[i] != want {
			t.Fatalf("invalid input changed output[%d] = %v", i, got[i])
		}
	}
}

func TestDenseToCOOConversion(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	tests := []struct {
		name       string
		rows, cols int
		input      []float32
		nnz        int
	}{
		{"empty", 2, 3, []float32{0, 0, 0, 0, 0, 0}, 0},
		{"rectangular", 2, 3, []float32{1, 0, -2, 0, 1e-20, 0}, 3},
		{"column", 3, 1, []float32{0, -1, 0}, 1},
		{"special", 2, 2, []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dense := solverTestBuffer(t, tt.input)
			sparse, err := ctx.DenseToSparse(dense, tt.rows, tt.cols, MatrixFormatCOO)
			if err != nil {
				t.Fatal(err)
			}
			defer sparse.Destroy()
			rows, cols, nnz, format := sparse.GetMatrixInfo()
			if rows != tt.rows || cols != tt.cols || nnz != tt.nnz || format != MatrixFormatCOO {
				t.Fatalf("dense-to-COO shape = %d x %d, nnz = %d, format = %d", rows, cols, nnz, format)
			}
			entries, err := readSparseRows(sparse)
			if err != nil {
				t.Fatal(err)
			}
			for row := range tt.rows {
				for col := range tt.cols {
					want := tt.input[row*tt.cols+col]
					got, exists := entries[row][col]
					if want == 0 {
						if exists {
							t.Fatalf("unexpected COO entry at (%d,%d)", row, col)
						}
					} else if !exists || (math.IsNaN(float64(want)) && !math.IsNaN(float64(got))) ||
						(!math.IsNaN(float64(want)) && got != want) {
						t.Fatalf("COO entry (%d,%d) = %v, present %v; want %v", row, col, got, exists, want)
					}
				}
			}
		})
	}
}

func TestSparseCOOTriangularSolve(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	tests := []struct {
		name        string
		rows, cols  []int32
		values, rhs []float32
		want        []float32
		invalid     bool
	}{
		{"lower-unsorted-duplicate", []int32{1, 0, 1, 0}, []int32{1, 0, 0, 0}, []float32{3, 1, 1, 1}, []float32{2, 7}, []float32{1, 2}, false},
		{"upper-unsorted-duplicate", []int32{1, 0, 0, 0}, []int32{1, 1, 0, 1}, []float32{3, .25, 2, .75}, []float32{4, 6}, []float32{1, 2}, false},
		{"diagonal", []int32{1, 0}, []int32{1, 0}, []float32{4, -2}, []float32{-2, 8}, []float32{1, 2}, false},
		{"missing-diagonal", []int32{0, 1}, []int32{0, 0}, []float32{2, 1}, []float32{2, 7}, nil, true},
		{"cancelled-diagonal", []int32{0, 0, 1}, []int32{0, 0, 1}, []float32{1, -1, 3}, []float32{2, 7}, nil, true},
		{"not-triangular", []int32{0, 0, 1, 1}, []int32{0, 1, 0, 1}, []float32{2, 1, 1, 3}, []float32{2, 7}, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matrix, err := ctx.CreateCOOMatrix(2, 2, tt.rows, tt.cols, tt.values)
			if err != nil {
				t.Fatal(err)
			}
			defer matrix.Destroy()
			rhs := solverTestBuffer(t, tt.rhs)
			out := solverTestBuffer(t, []float32{0, 0})
			err = ctx.SpSV(matrix, rhs, out)
			if tt.invalid {
				if err == nil {
					t.Fatal("expected invalid triangular system to fail")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(out, 2)
			if err != nil {
				t.Fatal(err)
			}
			for i := range tt.want {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-tt.want[i])) > 1e-6 {
					t.Fatalf("solution[%d] = %v, want %v", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestSparseScaledProduct(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	a := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{1, 2, 3})
	b, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0}, []int32{0, 1}, []float32{5, 4})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Destroy()
	c, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0, 0}, []int32{1, 0, 1, 0}, []float32{3, 0.5, -1, 1.5})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Destroy()
	if err := ctx.SpMM(2, a, b, -1, c); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, c, []float32{18, 9, 30, -3})

	identity := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 1})
	diagonal := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 2})
	cancel := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{-1, -2})
	if err := ctx.SpMM(1, diagonal, identity, 1, cancel); err != nil {
		t.Fatal(err)
	}
	if _, _, nnz, _ := cancel.GetMatrixInfo(); nnz != 0 {
		t.Fatalf("cancelled product has %d nonzeros", nnz)
	}
	sparseTestDense(t, ctx, cancel, []float32{0, 0, 0, 0})

	if err := ctx.SpMM(2, diagonal, identity, 3, diagonal); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, diagonal, []float32{5, 0, 0, 10})
	if err := ctx.SpMM(1, diagonal, identity, 1, identity); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, identity, []float32{6, 0, 0, 11})

	self := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 2})
	if err := ctx.SpMM(1, self, self, 2, self); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, self, []float32{3, 0, 0, 8})
	one := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 1})
	offDiagonal, err := ctx.CreateCOOMatrix(2, 2, []int32{0, 1}, []int32{1, 0}, []float32{7, 8})
	if err != nil {
		t.Fatal(err)
	}
	defer offDiagonal.Destroy()
	if err := ctx.SpMM(1, self, one, 0, offDiagonal); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, offDiagonal, []float32{3, 0, 0, 8})
	if _, _, nnz, _ := offDiagonal.GetMatrixInfo(); nnz != 2 {
		t.Fatalf("zero-scaled output should fit after pruning, got %d nonzeros", nnz)
	}
	onlyC, err := ctx.CreateCOOMatrix(2, 2, []int32{0, 1}, []int32{1, 0}, []float32{7, 8})
	if err != nil {
		t.Fatal(err)
	}
	defer onlyC.Destroy()
	if err := ctx.SpMM(0, self, one, -2, onlyC); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, onlyC, []float32{0, -14, -16, 0})

	selfCOO, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0, 0}, []int32{1, 1, 0, 1}, []float32{3, 0.25, 2, 0.75})
	if err != nil {
		t.Fatal(err)
	}
	defer selfCOO.Destroy()
	if err := ctx.SpMM(2, selfCOO, one, 1, selfCOO); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, selfCOO, []float32{6, 3, 0, 9})

	empty := sparseTestCSR(t, ctx, 2, 2, []int32{0, 0, 0}, nil, nil)
	keep := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{4, 5})
	if err := ctx.SpMM(3, empty, one, 2, keep); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, keep, []float32{8, 0, 0, 10})
	emptyOut, err := ctx.CreateCOOMatrix(2, 2, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyOut.Destroy()
	if err := ctx.SpMM(1, empty, empty, 1, emptyOut); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, emptyOut, []float32{0, 0, 0, 0})
	if _, _, nnz, _ := emptyOut.GetMatrixInfo(); nnz != 0 {
		t.Fatalf("empty product has %d nonzeros", nnz)
	}
}

func TestSparseMixedFormatProduct(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	aCSR := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	bCSR := sparseTestCSR(t, ctx, 3, 2, []int32{0, 1, 2, 3}, []int32{1, 0, 1}, []float32{4, 5, 6})
	aCOO, err := ctx.CreateCOOMatrix(2, 3, []int32{1, 0, 0, 0}, []int32{1, 2, 0, 2}, []float32{3, 1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer aCOO.Destroy()
	bCOO, err := ctx.CreateCOOMatrix(3, 2, []int32{2, 0, 1}, []int32{1, 1, 0}, []float32{6, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	defer bCOO.Destroy()
	for _, pair := range [][2]*SparseMatrix{{aCSR, bCOO}, {aCOO, bCSR}, {aCOO, bCOO}} {
		product, err := ctx.SpGEMM(pair[0], pair[1])
		if err != nil {
			t.Fatal(err)
		}
		rows, cols, _, format := product.GetMatrixInfo()
		if rows != 2 || cols != 2 || format != MatrixFormatCSR {
			t.Fatalf("mixed product shape = %d x %d, format = %d", rows, cols, format)
		}
		sparseTestDense(t, ctx, product, []float32{0, 16, 15, 0})
		if err := product.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
}

func sparseTestCSR(t *testing.T, ctx *SparseContext, rows, cols int, offsets, columns []int32, values []float32) *SparseMatrix {
	t.Helper()
	matrix, err := ctx.CreateCSRMatrix(rows, cols, offsets, columns, values)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	return matrix
}

func sparseTestDense(t *testing.T, ctx *SparseContext, matrix *SparseMatrix, want []float32) {
	t.Helper()
	dense, err := ctx.SparseToDense(matrix)
	if err != nil {
		t.Fatal(err)
	}
	defer dense.Free()
	values, err := readMathFloat32Memory(dense, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("dense[%d] = %v, want %v", i, values[i], want[i])
		}
	}
}

func testSparseNumerics(t *testing.T, ctx *SparseContext) {
	t.Helper()
	a := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	b := sparseTestCSR(t, ctx, 3, 2, []int32{0, 1, 2, 3}, []int32{1, 0, 1}, []float32{4, 5, 6})
	x := solverTestBuffer(t, []float32{2, 3, 4})
	y := solverTestBuffer(t, []float32{5, 7})
	if err := ctx.SpMV(2, a, x, -1, y); err != nil {
		t.Fatal(err)
	}
	values, err := readMathFloat32Memory(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 15 || values[1] != 11 {
		t.Fatalf("CSR SpMV = %v, want [15 11]", values)
	}
	duplicate := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{0, 2, 2, 1}, []float32{1, 1, 1, 3})
	if err := writeMathFloat32Memory(y, []float32{5, 7}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SpMV(2, duplicate, x, -1, y); err != nil {
		t.Fatal(err)
	}
	values, err = readMathFloat32Memory(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 15 || values[1] != 11 {
		t.Fatalf("duplicate CSR SpMV = %v, want [15 11]", values)
	}
	coo, err := ctx.CreateCOOMatrix(2, 3, []int32{0, 0, 0, 1}, []int32{0, 2, 2, 1}, []float32{1, 1, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	if err := writeMathFloat32Memory(y, []float32{5, 7}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SpMV(2, coo, x, -1, y); err != nil {
		t.Fatal(err)
	}
	values, err = readMathFloat32Memory(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 15 || values[1] != 11 {
		t.Fatalf("COO SpMV = %v, want [15 11]", values)
	}
	sparseTestDense(t, ctx, coo, []float32{1, 0, 2, 0, 3, 0})

	product, err := ctx.SpGEMM(a, b)
	if err != nil {
		t.Fatal(err)
	}
	defer product.Destroy()
	sparseTestDense(t, ctx, product, []float32{0, 16, 15, 0})
	duplicateProduct, err := ctx.SpGEMM(duplicate, b)
	if err != nil {
		t.Fatal(err)
	}
	defer duplicateProduct.Destroy()
	sparseTestDense(t, ctx, duplicateProduct, []float32{0, 16, 15, 0})
	cancelA := sparseTestCSR(t, ctx, 1, 2, []int32{0, 2}, []int32{0, 1}, []float32{1, 1})
	cancelB := sparseTestCSR(t, ctx, 2, 1, []int32{0, 1, 2}, []int32{0, 0}, []float32{1, -1})
	cancelProduct, err := ctx.SpGEMM(cancelA, cancelB)
	if err != nil {
		t.Fatal(err)
	}
	defer cancelProduct.Destroy()
	sparseTestDense(t, ctx, cancelProduct, []float32{0})
	mixedProduct, err := ctx.SpGEMM(coo, b)
	if err != nil {
		t.Fatal(err)
	}
	defer mixedProduct.Destroy()
	sparseTestDense(t, ctx, mixedProduct, []float32{0, 16, 15, 0})
	c := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 4}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
	if err := ctx.SpMM(2, a, b, 0.5, c); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, c, []float32{0.5, 33, 31.5, 2})
	cCOO, err := ctx.CreateCOOMatrix(2, 2, []int32{0, 0, 1, 1}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	defer cCOO.Destroy()
	if err := ctx.SpMM(2, a, b, 0.5, cCOO); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, cCOO, []float32{0.5, 33, 31.5, 2})
	cSmall := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 1})
	if err := ctx.SpMM(1, a, b, 1, cSmall); err == nil {
		t.Fatal("expected sparse multiplication to reject insufficient output capacity")
	}
	sparseTestDense(t, ctx, cSmall, []float32{1, 0, 0, 1})
	if _, _, nnz, _ := cSmall.GetMatrixInfo(); nnz != 2 {
		t.Fatalf("failed multiplication changed output nnz to %d", nnz)
	}

	empty := sparseTestCSR(t, ctx, 2, 2, []int32{0, 0, 0}, nil, nil)
	emptyProduct, err := ctx.SpGEMM(empty, empty)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyProduct.Destroy()
	sparseTestDense(t, ctx, emptyProduct, []float32{0, 0, 0, 0})
	emptyX := solverTestBuffer(t, []float32{1, 2})
	emptyY := solverTestBuffer(t, []float32{3, 4})
	if err := ctx.SpMV(1, empty, emptyX, 2, emptyY); err != nil {
		t.Fatal(err)
	}
	emptyValues, err := readMathFloat32Memory(emptyY, 2)
	if err != nil {
		t.Fatal(err)
	}
	if emptyValues[0] != 6 || emptyValues[1] != 8 {
		t.Fatalf("empty CSR SpMV = %v, want [6 8]", emptyValues)
	}
	emptyAllocated, err := ctx.CreateSparseMatrix(2, 2, 0, MatrixFormatCSR)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyAllocated.Destroy()
	if err := ctx.SpMV(1, emptyAllocated, emptyX, 2, emptyY); err != nil {
		t.Fatalf("new empty CSR matrix: %v", err)
	}
	emptyValues, err = readMathFloat32Memory(emptyY, 2)
	if err != nil {
		t.Fatal(err)
	}
	if emptyValues[0] != 12 || emptyValues[1] != 16 {
		t.Fatalf("new empty CSR SpMV = %v, want [12 16]", emptyValues)
	}

	for _, format := range []MatrixFormat{MatrixFormatCSR, MatrixFormatCOO} {
		dense := solverTestBuffer(t, []float32{0, 1e-20, -2, 0})
		sparse, err := ctx.DenseToSparse(dense, 2, 2, format)
		if err != nil {
			t.Fatal(err)
		}
		sparseTestDense(t, ctx, sparse, []float32{0, 1e-20, -2, 0})
		_, _, nnz, _ := sparse.GetMatrixInfo()
		if nnz != 2 {
			t.Fatalf("small nonzero entry was lost; nnz = %d", nnz)
		}
		entries, err := readSparseRows(sparse)
		if err != nil || entries[0][1] != float32(1e-20) || entries[1][0] != -2 {
			t.Fatalf("dense-to-sparse values = %v, error = %v", entries, err)
		}
		if err := sparse.Destroy(); err != nil {
			t.Fatal(err)
		}
		unsorted := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{2, 0, 1}, []float32{2, 1, 3})
		sparseTestDense(t, ctx, unsorted, []float32{1, 0, 2, 0, 3, 0})
	}
	for _, test := range []struct {
		rows, cols int
		values     []float32
		nnz        int
	}{
		{2, 3, []float32{0, 0, 0, 0, 0, 0}, 0},
		{2, 3, []float32{1, 0, -2, 0, 3, 0}, 3},
		{3, 1, []float32{0, -1, 0}, 1},
	} {
		dense := solverTestBuffer(t, test.values)
		sparse, err := ctx.DenseToSparse(dense, test.rows, test.cols, MatrixFormatCSR)
		if err != nil {
			t.Fatal(err)
		}
		rows, cols, nnz, format := sparse.GetMatrixInfo()
		if rows != test.rows || cols != test.cols || nnz != test.nnz || format != MatrixFormatCSR {
			t.Fatalf("dense-to-CSR dimensions = %d x %d, nnz = %d, format = %d", rows, cols, nnz, format)
		}
		sparseTestDense(t, ctx, sparse, test.values)
		if err := sparse.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
	special := solverTestBuffer(t, []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0})
	specialSparse, err := ctx.DenseToSparse(special, 2, 2, MatrixFormatCSR)
	if err != nil {
		t.Fatal(err)
	}
	defer specialSparse.Destroy()
	specialRows, err := readSparseRows(specialSparse)
	if err != nil || specialSparse.nnz != 3 || !math.IsNaN(float64(specialRows[0][0])) ||
		!math.IsInf(float64(specialRows[0][1]), 1) || !math.IsInf(float64(specialRows[1][0]), -1) {
		t.Fatalf("dense-to-CSR special values = %v, error = %v", specialRows, err)
	}
}

func testSparseLUAndTriangularSolve(t *testing.T, ctx *SparseContext) {
	t.Helper()
	matrix := sparseTestCSR(t, ctx, 3, 3, []int32{0, 2, 5, 7}, []int32{0, 1, 0, 1, 2, 1, 2}, []float32{2, 1, 4, 3, 1, 2, 4})
	lower, upper, err := ctx.SpLU(matrix)
	if err != nil {
		t.Fatal(err)
	}
	defer lower.Destroy()
	defer upper.Destroy()
	lDense, err := ctx.SparseToDense(lower)
	if err != nil {
		t.Fatal(err)
	}
	defer lDense.Free()
	uDense, err := ctx.SparseToDense(upper)
	if err != nil {
		t.Fatal(err)
	}
	defer uDense.Free()
	l, err := readMathFloat32Memory(lDense, 9)
	if err != nil {
		t.Fatal(err)
	}
	u, err := readMathFloat32Memory(uDense, 9)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 1, 0, 4, 3, 1, 0, 2, 4}
	for row := range 3 {
		for col := range 3 {
			var got float32
			for k := range 3 {
				got += l[row*3+k] * u[k*3+col]
			}
			if math.Abs(float64(got-want[row*3+col])) > 1e-5 {
				t.Fatalf("sparse LU[%d,%d]=%v, want %v", row, col, got, want[row*3+col])
			}
		}
	}
	triangular := sparseTestCSR(t, ctx, 3, 3, []int32{0, 1, 3, 6}, []int32{0, 0, 1, 0, 1, 2}, []float32{2, 1, 3, 4, -1, 2})
	rhs := solverTestBuffer(t, []float32{2, 7, 8})
	out := solverTestBuffer(t, []float32{0, 0, 0})
	if err := ctx.SpSV(triangular, rhs, out); err != nil {
		t.Fatal(err)
	}
	solution, err := readMathFloat32Memory(out, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range []float32{1, 2, 3} {
		if solution[i] != value {
			t.Fatalf("triangular solution[%d]=%v, want %v", i, solution[i], value)
		}
	}
	needsPivot := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{1, 0}, []float32{1, 1})
	if _, _, err := ctx.SpLU(needsPivot); err == nil {
		t.Fatal("expected sparse LU to reject unsupported pivoting")
	}
	if err := ctx.SpSV(matrix, rhs, out); err == nil {
		t.Fatal("expected non-triangular solve to fail")
	}
	upperTriangular := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	upperRHS := solverTestBuffer(t, []float32{4, 6})
	upperOut := solverTestBuffer(t, []float32{0, 0})
	if err := ctx.SpSV(upperTriangular, upperRHS, upperOut); err != nil {
		t.Fatal(err)
	}
	upperSolution, err := readMathFloat32Memory(upperOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if upperSolution[0] != 1 || upperSolution[1] != 2 {
		t.Fatalf("upper triangular solution = %v, want [1 2]", upperSolution)
	}
	unsorted := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 3}, []int32{0, 1, 0}, []float32{2, 3, 1})
	unsortedRHS := solverTestBuffer(t, []float32{2, 7})
	unsortedOut := solverTestBuffer(t, []float32{0, 0})
	if err := ctx.SpSV(unsorted, unsortedRHS, unsortedOut); err != nil {
		t.Fatal(err)
	}
	unsortedSolution, err := readMathFloat32Memory(unsortedOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if unsortedSolution[0] != 1 || unsortedSolution[1] != 2 {
		t.Fatalf("unsorted triangular solution = %v, want [1 2]", unsortedSolution)
	}
	coo, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 1}, []int32{0, 0, 1}, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	cooRHS := solverTestBuffer(t, []float32{2, 7})
	cooOut := solverTestBuffer(t, []float32{0, 0})
	if err := ctx.SpSV(coo, cooRHS, cooOut); err != nil {
		t.Fatal(err)
	}
	cooSolution, err := readMathFloat32Memory(cooOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if cooSolution[0] != 1 || cooSolution[1] != 2 {
		t.Fatalf("COO triangular solution = %v, want [1 2]", cooSolution)
	}
}

func TestSparseNumerics(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	testSparseNumerics(t, ctx)
	testSparseLUAndTriangularSolve(t, ctx)
}

func TestSparseRejectsInvalidIndices(t *testing.T) {
	ctx := &SparseContext{}
	if _, err := ctx.CreateCSRMatrix(2, 3, []int32{0, 2, 1}, []int32{0}, []float32{1}); err == nil {
		t.Fatal("expected invalid CSR offsets to fail")
	}
	if _, err := ctx.CreateCOOMatrix(2, 3, []int32{2}, []int32{0}, []float32{1}); err == nil {
		t.Fatal("expected invalid COO row to fail")
	}
	if _, err := ctx.CreateCSRMatrix(1, 1, []int32{0, 1}, []int32{1}, []float32{1}); err == nil {
		t.Fatal("expected invalid CSR column to fail")
	}
	matrix, err := ctx.CreateCSRMatrix(1, 1, []int32{0, 1}, []int32{0}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	if err := writeMathTypedMemory(matrix.colInds, []int32{2}); err != nil {
		t.Fatal(err)
	}
	if _, err := ctx.SparseToDense(matrix); err == nil {
		t.Fatal("expected malformed sparse buffer to fail")
	}
}
