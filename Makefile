# GoCUDA Makefile

.PHONY: all build test benchmark clean examples deps lint format

# Default target
all: build test

# Build the library
build:
	@echo "Building GoCUDA..."
	go build ./...

# Run tests
test:
	@echo "Running tests..."
	go test -v ./...

# Run benchmarks
benchmark:
	@echo "Running benchmarks..."
	go test -bench=. -benchmem ./...

# Run examples
examples: build
	@echo "Running basic demo..."
	cd demos && go run basic_demo.go

# Run comprehensive demo
demo-missing-features: build
	@echo "Running missing features demo..."
	cd demos/missing_features && go run main.go

# Run advanced features demo
demo-advanced: build
	@echo "Running advanced features demo..."
	cd demos/advanced_features && go run main.go

# Run all demos
demos: build
	@echo "Running all demos..."
	cd demos && go run basic_demo.go
	cd demos/missing_features && go run main.go
	cd demos/advanced_features && go run main.go

# Install dependencies
deps:
	@echo "Installing dependencies..."
	go mod tidy
	go mod download

# Lint code
lint:
	@echo "Running linter..."
	golangci-lint run

# Format code
format:
	@echo "Formatting code..."
	go fmt ./...
	goimports -w .

# Clean build artifacts
clean:
	@echo "Cleaning..."
	go clean ./...
	rm -f demos/demos.exe
	rm -f demos/*/*.exe
	rm -f *.exe

# Development setup
dev-setup:
	@echo "Setting up development environment..."
	go install golang.org/x/tools/cmd/goimports@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html
	@echo "Coverage report generated: coverage.html"

# Build for Windows
build-windows:
	@echo "Building for Windows..."
	GOOS=windows GOARCH=amd64 go build -o gocuda-windows.exe ./demos/missing_features

# Build for Linux
build-linux:
	@echo "Building for Linux..."
	GOOS=linux GOARCH=amd64 go build -o gocuda-linux ./demos/missing_features

# Build for macOS
build-darwin:
	@echo "Building for macOS..."
	GOOS=darwin GOARCH=amd64 go build -o gocuda-darwin ./demos/missing_features

# Build all platforms
build-all: build-windows build-linux build-darwin

# Generate documentation
docs:
	@echo "Generating documentation..."
	godoc -http=:6060 &
	@echo "Documentation server running at http://localhost:6060"

# Run example programs
demo-basic:
	@echo "Running basic demo..."
	cd demos && go run basic_demo.go

demo-memory:
	@echo "Running memory demo..."
	cd demos && go run memory_demo.go

demo-detection:
	@echo "Running CUDA detection demo..."
	cd demos && go run cuda_detection_demo.go

# Performance profiling
profile:
	@echo "Running performance profile..."
	go test -cpuprofile=cpu.prof -memprofile=mem.prof -bench=. ./...
	@echo "CPU profile: cpu.prof"
	@echo "Memory profile: mem.prof"

# Check for updates
update:
	@echo "Checking for updates..."
	go list -u -m all

# Install as library
install:
	@echo "Installing GoCUDA library..."
	go install ./...

# Verify installation
verify:
	@echo "Verifying installation..."
	go list -m github.com/stitch1968/gocuda

# Create release package
package:
	@echo "Creating release package..."
	mkdir -p release
	cp -r . release/gocuda
	cd release && tar -czf gocuda.tar.gz gocuda/
	@echo "Release package created: release/gocuda.tar.gz"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build and test"
	@echo "  build        - Build the library"
	@echo "  test         - Run tests"
	@echo "  benchmark    - Run benchmarks"
	@echo "  examples     - Run basic demo"
	@echo "  demos        - Run all demos"
	@echo "  demo-missing-features - Run comprehensive library demo"
	@echo "  demo-advanced- Run advanced features demo"
	@echo "  demo-basic   - Run basic demo"
	@echo "  demo-memory  - Run memory demo"
	@echo "  demo-detection - Run CUDA detection demo"
	@echo "  deps         - Install dependencies"
	@echo "  lint         - Run linter"
	@echo "  format       - Format code"
	@echo "  clean        - Clean build artifacts"
	@echo "  dev-setup    - Setup development environment"
	@echo "  test-coverage- Run tests with coverage"
	@echo "  build-all    - Build for all platforms"
	@echo "  docs         - Generate documentation"
	@echo "  profile      - Run performance profiling"
	@echo "  install      - Install as library"
	@echo "  package      - Create release package"
	@echo "  help         - Show this help"
