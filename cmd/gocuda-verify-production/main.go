package main

import (
	"fmt"
	"os"

	"github.com/stitch1968/gocuda/libraries"
)

func main() {
	if err := libraries.VerifyProductionReadiness(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Println("GoCUDA production readiness check passed")
}
