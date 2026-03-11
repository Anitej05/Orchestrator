"""LibreOffice (soffice) wrapper for Office → PDF conversion.

Handles headless mode, user profile isolation, and common error cases.

Usage:
    python soffice.py --headless --convert-to pdf input.pptx
    python soffice.py --headless --convert-to pdf input.docx --outdir output/

Examples:
    # Convert PPTX to PDF for slide image extraction
    python soffice.py --headless --convert-to pdf presentation.pptx
    # Then use pdftoppm to create slide images:
    #   pdftoppm -jpeg -r 150 presentation.pdf slide
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_soffice() -> str:
    """Find the LibreOffice soffice executable."""
    # Check if it's in PATH
    soffice = shutil.which("soffice")
    if soffice:
        return soffice

    # Check common installation locations
    system = platform.system()
    candidates = []

    if system == "Windows":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    elif system == "Darwin":
        candidates = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    else:  # Linux
        candidates = [
            "/usr/bin/soffice",
            "/usr/lib/libreoffice/program/soffice",
            "/snap/bin/libreoffice",
        ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def convert(
    input_file: str,
    output_format: str = "pdf",
    output_dir: str = None,
    timeout: int = 120,
) -> tuple:
    """Convert an Office file using LibreOffice.

    Args:
        input_file: Path to the input file
        output_format: Target format (pdf, png, jpg, etc.)
        output_dir: Output directory (default: same as input file)
        timeout: Timeout in seconds

    Returns:
        Tuple of (output_path_or_none, message_string)
    """
    soffice = find_soffice()
    if not soffice:
        return None, "Error: LibreOffice (soffice) not found. Install LibreOffice."

    input_path = Path(input_file)
    if not input_path.exists():
        return None, f"Error: {input_file} does not exist"

    if output_dir is None:
        output_dir = str(input_path.parent)

    # Use a unique user profile to avoid conflicts with running instances
    with tempfile.TemporaryDirectory(prefix="soffice_profile_") as profile_dir:
        cmd = [
            soffice,
            "--headless",
            "--norestore",
            "--nologo",
            f"-env:UserInstallation=file:///{Path(profile_dir).as_posix()}",
            "--convert-to",
            output_format,
            "--outdir",
            output_dir,
            str(input_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip() if result.stderr else "Unknown error"
                return None, f"Error: soffice conversion failed: {stderr}"

            # Determine output file path
            output_name = input_path.stem + "." + output_format
            output_path = Path(output_dir) / output_name

            if output_path.exists():
                return str(output_path), f"Converted {input_file} → {output_path}"
            else:
                return None, f"Warning: Conversion ran but output file not found at {output_path}"

        except subprocess.TimeoutExpired:
            return None, f"Error: Conversion timed out after {timeout}s"
        except FileNotFoundError:
            return None, f"Error: Could not execute {soffice}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert Office files using LibreOffice"
    )
    parser.add_argument("input_file", help="Office file to convert")
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run in headless mode (default: true)"
    )
    parser.add_argument(
        "--convert-to", dest="format", default="pdf",
        help="Output format (default: pdf)"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout in seconds (default: 120)"
    )
    args = parser.parse_args()

    output_path, message = convert(
        args.input_file,
        output_format=args.format,
        output_dir=args.outdir,
        timeout=args.timeout,
    )
    print(message)

    if output_path is None and "Error" in message:
        sys.exit(1)


if __name__ == "__main__":
    main()
