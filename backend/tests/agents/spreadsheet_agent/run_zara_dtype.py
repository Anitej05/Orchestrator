import sys
import os
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.spreadsheet_agent.utils import load_dataframe


def resolve_zara_path() -> Path:
    """Locate the Zara test dataset regardless of cwd."""
    candidates = [
        ROOT / "backend" / "tests" / "test_data" / "zara.xlsx",
        Path(__file__).parent / "test_data" / "zara.xlsx",
        ROOT / "tests" / "test_data" / "zara.xlsx",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Last resort: search under backend/tests/test_data for any zara*.xlsx
    search_root = ROOT / "backend" / "tests" / "test_data"
    if search_root.exists():
        matches = list(search_root.glob("zara*.xlsx"))
        if matches:
            return matches[0]

    raise FileNotFoundError("zara.xlsx not found in backend/tests/test_data; checked: " + ", ".join(str(c) for c in candidates))


def main():
    zara_path = resolve_zara_path()
    print(f"Using Zara file: {zara_path}")

    df = load_dataframe(str(zara_path))
    result = df["Sales Volume"].dtype
    print(f"df['Sales Volume'].dtype => {result}")


if __name__ == "__main__":
    main()
