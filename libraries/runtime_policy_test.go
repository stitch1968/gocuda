package libraries

import (
	"strings"
	"testing"
)

func TestHelperBackedLibrariesContainsKnownWrappers(t *testing.T) {
	libraries := HelperBackedLibraries()
	if len(libraries) != 0 {
		t.Fatalf("expected helper-backed library list to be empty, got %v", libraries)
	}

	if libraryIsHelperBacked("nvJPEG2000") {
		t.Fatal("did not expect nvJPEG2000 to remain helper-backed")
	}
	if libraryIsHelperBacked("AmgX") {
		t.Fatal("did not expect AmgX to remain helper-backed")
	}
	if libraryIsHelperBacked("cuDNN") {
		t.Fatal("did not expect cuDNN to remain helper-backed")
	}
	if libraryIsHelperBacked("cuFFT") {
		t.Fatal("did not expect cuFFT to remain helper-backed")
	}
	if libraryIsHelperBacked("cuRAND") {
		t.Fatal("did not expect cuRAND to remain helper-backed")
	}
	if libraryIsHelperBacked("CUDA Math API") {
		t.Fatal("did not expect CUDA Math API to remain helper-backed")
	}
}

func TestLibraryReadinessMatrixMarksCurrentSurfaceNonProduction(t *testing.T) {
	matrix := LibraryReadinessMatrix()
	if len(matrix) == 0 {
		t.Fatal("expected readiness matrix to be populated")
	}

	foundNativeCuFFT := false
	foundNativeCuRAND := false
	foundNativeCuSOLVER := false
	foundNativeCuSPARSE := false
	foundValidatedCudaMath := false
	foundValidatedThrust := false
	foundValidatedNVJPEG := false
	foundValidatedCuTensor := false
	foundValidatedCutlass := false
	foundValidatedCuDSS := false
	foundValidatedCuDNN := false
	foundValidatedAmgX := false
	foundValidatedNVJPEG2000 := false
	for _, entry := range matrix {
		if entry.Name == "cuFFT" {
			if !entry.ProductionReady {
				t.Fatal("expected cuFFT to be marked production-ready")
			}
			if entry.Mode != ImplementationModeNativeCUDA {
				t.Fatalf("expected cuFFT to be native-cuda, got %s", entry.Mode)
			}
			foundNativeCuFFT = true
			continue
		}
		if entry.Name == "cuRAND" {
			if !entry.ProductionReady {
				t.Fatal("expected cuRAND to be marked production-ready")
			}
			if entry.Mode != ImplementationModeNativeCUDA {
				t.Fatalf("expected cuRAND to be native-cuda, got %s", entry.Mode)
			}
			foundNativeCuRAND = true
			continue
		}
		if entry.Name == "cuSOLVER" {
			if !entry.ProductionReady {
				t.Fatal("expected cuSOLVER to be marked production-ready")
			}
			if entry.Mode != ImplementationModeNativeCUDA {
				t.Fatalf("expected cuSOLVER to be native-cuda, got %s", entry.Mode)
			}
			foundNativeCuSOLVER = true
			continue
		}
		if entry.Name == "cuSPARSE" {
			if !entry.ProductionReady {
				t.Fatal("expected cuSPARSE to be marked production-ready")
			}
			if entry.Mode != ImplementationModeNativeCUDA {
				t.Fatalf("expected cuSPARSE to be native-cuda, got %s", entry.Mode)
			}
			foundNativeCuSPARSE = true
			continue
		}
		if entry.Name == "CUDA Math API" {
			if !entry.ProductionReady {
				t.Fatal("expected CUDA Math API to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected CUDA Math API to be validated-go, got %s", entry.Mode)
			}
			foundValidatedCudaMath = true
			continue
		}
		if entry.Name == "Thrust" {
			if !entry.ProductionReady {
				t.Fatal("expected Thrust to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected Thrust to be validated-go, got %s", entry.Mode)
			}
			foundValidatedThrust = true
			continue
		}
		if entry.Name == "nvJPEG" {
			if !entry.ProductionReady {
				t.Fatal("expected nvJPEG to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected nvJPEG to be validated-go, got %s", entry.Mode)
			}
			foundValidatedNVJPEG = true
			continue
		}
		if entry.Name == "cuTENSOR" {
			if !entry.ProductionReady {
				t.Fatal("expected cuTENSOR to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected cuTENSOR to be validated-go, got %s", entry.Mode)
			}
			foundValidatedCuTensor = true
			continue
		}
		if entry.Name == "CUTLASS" {
			if !entry.ProductionReady {
				t.Fatal("expected CUTLASS to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected CUTLASS to be validated-go, got %s", entry.Mode)
			}
			foundValidatedCutlass = true
			continue
		}
		if entry.Name == "cuDSS" {
			if !entry.ProductionReady {
				t.Fatal("expected cuDSS to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected cuDSS to be validated-go, got %s", entry.Mode)
			}
			foundValidatedCuDSS = true
			continue
		}
		if entry.Name == "cuDNN" {
			if !entry.ProductionReady {
				t.Fatal("expected cuDNN to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected cuDNN to be validated-go, got %s", entry.Mode)
			}
			foundValidatedCuDNN = true
			continue
		}
		if entry.Name == "AmgX" {
			if !entry.ProductionReady {
				t.Fatal("expected AmgX to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected AmgX to be validated-go, got %s", entry.Mode)
			}
			foundValidatedAmgX = true
			continue
		}
		if entry.Name == "nvJPEG2000" {
			if !entry.ProductionReady {
				t.Fatal("expected nvJPEG2000 to be marked production-ready")
			}
			if entry.Mode != ImplementationModeValidatedGo {
				t.Fatalf("expected nvJPEG2000 to be validated-go, got %s", entry.Mode)
			}
			foundValidatedNVJPEG2000 = true
			continue
		}
		if entry.ProductionReady {
			t.Fatalf("did not expect %s to be marked production-ready yet", entry.Name)
		}
		if entry.BlockingReason == "" {
			t.Fatalf("expected blocking reason for %s", entry.Name)
		}
	}
	if !foundNativeCuFFT {
		t.Fatal("expected cuFFT entry in readiness matrix")
	}
	if !foundNativeCuRAND {
		t.Fatal("expected cuRAND entry in readiness matrix")
	}
	if !foundNativeCuSOLVER {
		t.Fatal("expected cuSOLVER entry in readiness matrix")
	}
	if !foundNativeCuSPARSE {
		t.Fatal("expected cuSPARSE entry in readiness matrix")
	}
	if !foundValidatedCudaMath {
		t.Fatal("expected CUDA Math API entry in readiness matrix")
	}
	if !foundValidatedThrust {
		t.Fatal("expected Thrust entry in readiness matrix")
	}
	if !foundValidatedNVJPEG {
		t.Fatal("expected nvJPEG entry in readiness matrix")
	}
	if !foundValidatedCuTensor {
		t.Fatal("expected cuTENSOR entry in readiness matrix")
	}
	if !foundValidatedCutlass {
		t.Fatal("expected CUTLASS entry in readiness matrix")
	}
	if !foundValidatedCuDSS {
		t.Fatal("expected cuDSS entry in readiness matrix")
	}
	if !foundValidatedCuDNN {
		t.Fatal("expected cuDNN entry in readiness matrix")
	}
	if !foundValidatedAmgX {
		t.Fatal("expected AmgX entry in readiness matrix")
	}
	if !foundValidatedNVJPEG2000 {
		t.Fatal("expected nvJPEG2000 entry in readiness matrix")
	}

	ready := ProductionReadyLibraries()
	if len(ready) != 13 || ready[0] != "AmgX" || ready[1] != "CUDA Math API" || ready[2] != "cuDNN" || ready[3] != "cuDSS" || ready[4] != "cuFFT" || ready[5] != "cuRAND" || ready[6] != "cuSOLVER" || ready[7] != "cuSPARSE" || ready[8] != "cuTENSOR" || ready[9] != "CUTLASS" || ready[10] != "nvJPEG" || ready[11] != "nvJPEG2000" || ready[12] != "Thrust" {
		t.Fatalf("expected AmgX, CUDA Math API, cuDNN, cuDSS, cuFFT, cuRAND, cuSOLVER, cuSPARSE, cuTENSOR, CUTLASS, nvJPEG, nvJPEG2000, and Thrust to be production-ready, got %v", ready)
	}
}

func TestProductionReadinessViolationsIncludesBlockingLibraries(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "0")
	violations := ProductionReadinessViolations()
	if len(violations) != 0 {
		t.Fatalf("expected no production-readiness violations, got %v", violations)
	}
}

func TestVerifyProductionReadinessFailsWhileHelperSurfaceRemains(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "0")
	if err := VerifyProductionReadiness(); err != nil {
		t.Fatalf("expected production readiness verification to pass, got %v", err)
	}
}

func TestProductionReadinessViolationsIncludesExperimentalHelpersOptIn(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "1")
	violations := ProductionReadinessViolations()
	found := false
	for _, violation := range violations {
		if strings.Contains(violation, experimentalHelpersEnvVar) {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected %s violation in %v", experimentalHelpersEnvVar, violations)
	}
}

func TestExperimentalHelpersEnabledParsing(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "true")
	if !ExperimentalHelpersEnabled() {
		t.Fatal("expected experimental helpers to be enabled")
	}

	t.Setenv(experimentalHelpersEnvVar, "0")
	if ExperimentalHelpersEnabled() {
		t.Fatal("expected experimental helpers to be disabled")
	}
}

func TestRequireHelperBackedLibraryForModeBlocksCudaByDefault(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "0")
	if err := requireHelperBackedLibraryForMode("nvJPEG2000", true); err != nil {
		t.Fatalf("expected production-ready library to be allowed in CUDA mode: %v", err)
	}
}

func TestRequireHelperBackedLibraryForModeAllowsSimulationAndOptIn(t *testing.T) {
	t.Setenv(experimentalHelpersEnvVar, "0")
	if err := requireHelperBackedLibraryForMode("nvJPEG2000", false); err != nil {
		t.Fatalf("expected simulation mode to allow production-ready library: %v", err)
	}

	t.Setenv(experimentalHelpersEnvVar, "1")
	if err := requireHelperBackedLibraryForMode("nvJPEG2000", true); err != nil {
		t.Fatalf("expected CUDA mode to allow production-ready library regardless of helper opt-in: %v", err)
	}
}
