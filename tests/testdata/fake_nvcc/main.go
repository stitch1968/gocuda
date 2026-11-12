package main

import (
	"fmt"
	"os"
	"strings"
	"time"
)

func main() {
	args := os.Args[1:]
	var output string
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--output-file" {
			output = args[i+1]
		}
	}
	if output == "" || len(args) == 0 {
		fmt.Fprintln(os.Stderr, "missing fake compiler arguments")
		os.Exit(2)
	}
	input := args[len(args)-1]
	source, err := os.ReadFile(input)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if marker := os.Getenv("GOCUDA_FAKE_NVCC_MARKER"); marker != "" {
		if err := os.WriteFile(marker, []byte(input+"\n"+strings.Join(args, "\n")), 0o600); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	if strings.Contains(string(source), "SLOW") {
		time.Sleep(10 * time.Second)
	}
	if strings.Contains(string(source), "FAIL") {
		fmt.Fprintln(os.Stderr, "synthetic compilation error")
		os.Exit(1)
	}
	if err := os.WriteFile(output, []byte("// fake PTX\n"), 0o600); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	fmt.Println("synthetic compiler diagnostic")
}
