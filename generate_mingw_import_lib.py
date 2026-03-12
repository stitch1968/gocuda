import argparse
import re
import subprocess
import sys
from pathlib import Path


EXPORT_PATTERN = re.compile(r"^\s*\[\s*\d+\]\s*\+base\[\s*(\d+)\]\s*[0-9a-fA-F]+\s+(\S+)")


def parse_exports(objdump_output: str):
    exports = []
    start_marker = "[Ordinal/Name Pointer] Table"
    start_index = objdump_output.find(start_marker)
    if start_index == -1:
        raise RuntimeError("could not find export name table in objdump output")

    for line in objdump_output[start_index:].splitlines():
        match = EXPORT_PATTERN.search(line)
        if match:
            ordinal = match.group(1)
            name = match.group(2)
            exports.append((name, ordinal))
    if not exports:
        raise RuntimeError("no exports were parsed from objdump output")
    return exports


def run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command failed: {' '.join(command)}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Generate a MinGW import library from a CUDA DLL using objdump and dlltool.")
    parser.add_argument("dll", help="Path to the source DLL")
    parser.add_argument("--output-dir", default="lib_mingw", help="Directory for generated .exports, .def, and .a files")
    parser.add_argument("--import-lib-name", help="Override the generated import library base name (without lib prefix or .a suffix)")
    parser.add_argument("--dlltool", default="dlltool", help="dlltool executable")
    parser.add_argument("--objdump", default="objdump", help="objdump executable")
    args = parser.parse_args()

    dll_path = Path(args.dll).resolve()
    if not dll_path.exists():
        raise RuntimeError(f"DLL not found: {dll_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = dll_path.stem
    exports_path = output_dir / f"{stem}.exports"
    def_path = output_dir / f"{stem}.def"
    import_lib_name = args.import_lib_name or stem.split('64_')[0]
    import_lib_path = output_dir / f"lib{import_lib_name}.a"

    objdump_output = run_command([args.objdump, "-p", str(dll_path)])
    exports_path.write_text(objdump_output, encoding="utf-8")

    exports = parse_exports(objdump_output)
    with def_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"LIBRARY {dll_path.name}\n")
        handle.write("EXPORTS\n")
        for name, ordinal in exports:
            handle.write(f"  {name} @{ordinal}\n")

    run_command([args.dlltool, "-d", str(def_path), "-l", str(import_lib_path)])

    print(f"Generated {def_path}")
    print(f"Generated {import_lib_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)