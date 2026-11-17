#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch concurrent viewers, hour by hour.

The grader runs `python3 -P -s /bot/main.py` in python:3.13-slim with no network and no
third-party packages, one case at a time: only that case's file sits in /input, and the run
must leave its forecast at /output/<id>.out. `-P` keeps this file's directory off sys.path,
so the import of the model beside it needs the path insert below.

Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:

    mkdir -p out
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

The forecast itself is in model.py, together with the evidence behind each of its choices.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402  (after the path insert, by design)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def render(forecast):
    """The contract's exact shape: one object, one key, 23 rows, hours 2..24."""
    rows = [{"hour": h, "q": forecast[h]} for h in model.HOURS]
    return json.dumps({"hours": rows}, separators=(",", ":")) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Name order, so two runs of the same case write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        payload = render(model.forecast(case))
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
