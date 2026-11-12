package libraries

import (
	"strings"
	"testing"
)

func TestLibraryReadinessMatrixDoesNotClaimUnverifiedProductionReadiness(t *testing.T) {
	matrix := LibraryReadinessMatrix()
	if len(matrix) != 13 {
		t.Fatalf("expected 13 library entries, got %d", len(matrix))
	}
	names := make(map[string]bool)
	for _, entry := range matrix {
		if names[entry.Name] {
			t.Fatalf("duplicate library %q", entry.Name)
		}
		names[entry.Name] = true
		if entry.ProductionReady || entry.BlockingReason == "" {
			t.Fatalf("library %q must have an explicit readiness blocker", entry.Name)
		}
	}
	for _, name := range []string{"AmgX", "CUDA Math API", "cuDNN", "cuDSS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE", "cuTENSOR", "CUTLASS", "nvJPEG", "nvJPEG2000", "Thrust"} {
		if !names[name] {
			t.Fatalf("missing library %q", name)
		}
	}
	if ready := ProductionReadyLibraries(); len(ready) != 0 {
		t.Fatalf("unexpected production-ready libraries: %v", ready)
	}
	matrix[0].Name = "modified"
	if LibraryReadinessMatrix()[0].Name == "modified" {
		t.Fatal("readiness matrix must return a copy")
	}
}

func TestProductionReadinessReportsBlockers(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "0")
	violations := ProductionReadinessViolations()
	if len(violations) != len(LibraryReadinessMatrix()) {
		t.Fatalf("expected one blocker per library, got %v", violations)
	}
	if err := VerifyProductionReadiness(); err == nil {
		t.Fatal("expected readiness verification to fail while coverage is unverified")
	}
}

func TestProductionReadinessReportsExperimentalHelpers(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "1")
	for _, violation := range ProductionReadinessViolations() {
		if strings.Contains(violation, experimentalHelpersEnvVar) {
			return
		}
	}
	t.Fatal("expected experimental helpers to be reported")
}

func TestExperimentalHelpersEnabledParsing(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  bool
	}{
		{"true", true}, {"YES", true}, {"1", true}, {"on", true},
		{"0", false}, {"false", false}, {"", false}, {"unknown", false},
	} {
		t.Setenv(experimentalHelpersEnvVar, tc.value)
		if got := ExperimentalHelpersEnabled(); got != tc.want {
			t.Fatalf("%q: got %t, want %t", tc.value, got, tc.want)
		}
	}
}

func TestHelperBackedLibraryPolicy(t *testing.T) {
	if got := HelperBackedLibraries(); len(got) != 0 {
		t.Fatalf("unexpected helper-backed entries: %v", got)
	}
	t.Setenv(experimentalHelpersEnvVar, "0")
	if err := requireHelperBackedLibraryForMode("nvJPEG2000", true); err != nil {
		t.Fatalf("existing execution policy changed: %v", err)
	}
	if err := requireHelperBackedLibraryForMode("nvJPEG2000", false); err != nil {
		t.Fatalf("existing simulation policy changed: %v", err)
	}
}
