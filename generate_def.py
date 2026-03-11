import re
import sys

def parse_exports(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # Find the start of the Name Pointer Table
    start_marker = "[Ordinal/Name Pointer] Table"
    start_index = content.find(start_marker)
    if start_index == -1:
        print("Could not find Name Pointer Table")
        sys.exit(1)

    # Extract lines after the marker
    lines = content[start_index:].splitlines()
    
    exports = []
    # Regex to match: 	[   0] +base[   1]  0000 __cudaGetKernel
    # We want ordinal (1) and name (__cudaGetKernel)
    # The ordinal in the brackets is +base[ X ].
    
    pattern = re.compile(r"^\s*\[\s*\d+\]\s*\+base\[\s*(\d+)\]\s*[0-9a-fA-F]+\s+(\S+)")

    for line in lines:
        match = pattern.search(line)
        if match:
            ordinal = match.group(1)
            name = match.group(2)
            exports.append((name, ordinal))

    print(f"Found {len(exports)} exports.")

    with open(output_file, 'w') as f:
        f.write("LIBRARY cudart64_13.dll\n")
        f.write("EXPORTS\n")
        for name, ordinal in exports:
            f.write(f"  {name} @{ordinal}\n")

if __name__ == "__main__":
    parse_exports("cudart.exports", "cudart.def")
