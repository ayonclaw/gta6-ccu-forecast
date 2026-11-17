"""Entry point for the GTA6 concurrent-viewer forecast task.

The grader invokes this as `python3 -P -s /bot/main.py` inside a stdlib-only
container with no network access.  It runs one case at a time: only the current
case's `.in` file is present in the input directory, and the program must write
the matching `<id>.out` file into the output directory.

The `-P` flag keeps the interpreter from prepending the script's own directory
to `sys.path`, so `import model` would fail without an explicit path insert.
That insert is the first thing this module does after importing `pathlib`.

The `-s` flag only disables the user site directory; it does not affect us.

Paths default to `/input` and `/output`, and can be overridden with the
environment variables MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR.  A local run:

    mkdir -p /tmp/out
    MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=/tmp/out python3 -P -s main.py

Every `*.in` file in the input directory is processed in sorted order, so a
directory holding several cases produces several `.out` files in one pass.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402  (import must follow the sys.path insert above)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = model.forecast(case)
        payload = {
            "hours": [
                {"hour": h, "q": [float(q) for q in result[h]]}
                for h in model.HOURS
            ]
        }
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
