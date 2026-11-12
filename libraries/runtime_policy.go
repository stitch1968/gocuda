package libraries

import (
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"

	cuda "github.com/stitch1968/gocuda"
)

const experimentalHelpersEnvVar = "GOCUDA_EXPERIMENTAL_HELPERS"

type LibraryImplementationMode string

const (
	ImplementationModeNativeCUDA   LibraryImplementationMode = "native-cuda"
	ImplementationModeValidatedGo  LibraryImplementationMode = "validated-go"
	ImplementationModeHelperBacked LibraryImplementationMode = "helper-backed"
)

type LibraryReadiness struct {
	Name            string
	Mode            LibraryImplementationMode
	ProductionReady bool
	BlockingReason  string
}

var libraryReadinessMatrix = []LibraryReadiness{
	{Name: "AmgX", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "CUDA Math API", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "cuDNN", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "cuDSS", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "cuFFT", Mode: ImplementationModeNativeCUDA, BlockingReason: readinessNotEstablished},
	{Name: "cuRAND", Mode: ImplementationModeNativeCUDA, BlockingReason: readinessNotEstablished},
	{Name: "cuSOLVER", Mode: ImplementationModeNativeCUDA, BlockingReason: readinessNotEstablished},
	{Name: "cuSPARSE", Mode: ImplementationModeNativeCUDA, BlockingReason: readinessNotEstablished},
	{Name: "cuTENSOR", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "CUTLASS", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "nvJPEG", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "nvJPEG2000", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
	{Name: "Thrust", Mode: ImplementationModeValidatedGo, BlockingReason: readinessNotEstablished},
}

const readinessNotEstablished = "operation-level backend coverage and production validation are not established"

var helperBackedLibraries = helperBackedLibraryNames(libraryReadinessMatrix)

func helperBackedLibraryNames(matrix []LibraryReadiness) []string {
	result := make([]string, 0, len(matrix))
	for _, entry := range matrix {
		if entry.Mode == ImplementationModeHelperBacked {
			result = append(result, entry.Name)
		}
	}
	return result
}

// LibraryReadinessMatrix returns the current readiness state for high-level
// CUDA ecosystem wrappers exposed by this repository. Mode describes the
// primary implementation strategy, not a guarantee for every operation.
func LibraryReadinessMatrix() []LibraryReadiness {
	result := make([]LibraryReadiness, len(libraryReadinessMatrix))
	copy(result, libraryReadinessMatrix)
	return result
}

// ProductionReadyLibraries returns the set of high-level wrappers currently
// considered production-ready for CUDA mode.
func ProductionReadyLibraries() []string {
	result := make([]string, 0, len(libraryReadinessMatrix))
	for _, entry := range libraryReadinessMatrix {
		if entry.ProductionReady {
			result = append(result, entry.Name)
		}
	}
	return result
}

// ProductionReadinessViolations returns the set of readiness issues that block
// the current high-level library surface from being considered production-ready.
func ProductionReadinessViolations() []string {
	violations := make([]string, 0, len(libraryReadinessMatrix)+1)
	for _, entry := range libraryReadinessMatrix {
		if entry.ProductionReady {
			continue
		}
		violations = append(violations, fmt.Sprintf("%s: %s", entry.Name, entry.BlockingReason))
	}
	if ExperimentalHelpersEnabled() {
		violations = append(violations, fmt.Sprintf("%s is enabled; production builds must not opt into helper-backed compatibility paths", experimentalHelpersEnvVar))
	}
	return violations
}

// VerifyProductionReadiness reports whether the currently exposed high-level
// library surface meets the repository's production-readiness gate.
func VerifyProductionReadiness() error {
	violations := ProductionReadinessViolations()
	if len(violations) == 0 {
		return nil
	}
	return errors.New("production readiness check failed:\n - " + strings.Join(violations, "\n - "))
}

// HelperBackedLibraries returns the high-level wrappers that currently rely on
// helper-backed or simulated implementations in this repository.
func HelperBackedLibraries() []string {
	result := make([]string, len(helperBackedLibraries))
	copy(result, helperBackedLibraries)
	return result
}

// ExperimentalHelpersEnabled reports whether helper-backed library execution is
// explicitly allowed in CUDA mode.
func ExperimentalHelpersEnabled() bool {
	value, ok := os.LookupEnv(experimentalHelpersEnvVar)
	if !ok {
		return false
	}

	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func libraryIsHelperBacked(name string) bool {
	return slices.Contains(helperBackedLibraries, name)
}

func helperBackedLibraryError(name string) error {
	return fmt.Errorf("%s is helper-backed in this repository and is not production-ready in CUDA mode; set %s=1 to opt into compatibility helpers", name, experimentalHelpersEnvVar)
}

func requireHelperBackedLibrary(name string) error {
	return requireHelperBackedLibraryForMode(name, cuda.ShouldUseCuda())
}

func requireHelperBackedLibraryForMode(name string, cudaMode bool) error {
	if !libraryIsHelperBacked(name) {
		return nil
	}
	if !cudaMode || ExperimentalHelpersEnabled() {
		return nil
	}
	return helperBackedLibraryError(name)
}
