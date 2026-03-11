package main

import (
	"fmt"
	"os"
	"strings"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
)

func main() {
	if err := cuda.Initialize(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	runtimeInfo := cuda.GetCudaRuntime()
	deviceID, err := cuda.CurrentCudaDeviceID()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	devices, err := cuda.GetDevices()
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	mode := "simulation"
	if cuda.ShouldUseCuda() {
		mode = "cuda"
	}

	ready := libraries.ProductionReadyLibraries()
	helperBacked := libraries.HelperBackedLibraries()

	fmt.Printf("Mode: %s\n", mode)
	fmt.Printf("CUDA Available: %t\n", runtimeInfo.Available)
	fmt.Printf("Runtime Version: %d\n", runtimeInfo.RuntimeVersion)
	fmt.Printf("Driver Version: %d\n", runtimeInfo.DriverVersion)
	fmt.Printf("Current Device ID: %d\n", deviceID)
	fmt.Printf("Detected Devices: %d\n", len(devices))
	for _, device := range devices {
		fmt.Printf("  - Device %d: %s (Compute %d.%d, GlobalMem %d)\n", device.ID, device.Name, device.Properties.Major, device.Properties.Minor, device.Properties.TotalGlobalMem)
	}
	fmt.Printf("Production-Ready Libraries: %s\n", strings.Join(ready, ", "))
	fmt.Printf("Helper-Backed Libraries: %s\n", strings.Join(helperBacked, ", "))
	if violations := libraries.ProductionReadinessViolations(); len(violations) > 0 {
		fmt.Println("Production Gate Violations:")
		for _, violation := range violations {
			fmt.Printf("  - %s\n", violation)
		}
	}
}
