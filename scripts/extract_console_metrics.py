#!/usr/bin/env python3
"""Extract scalar metrics from Ray/verl console logs into long-form CSV."""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Set, Tuple


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
INLINE_METRIC_RE = re.compile(
    rf"(?:^|\s-\s)([A-Za-z0-9_./@+\-]+):\s*({NUMBER})(?=\s|$|,)"
)
QUOTED_METRIC_RE = re.compile(rf"""['"]([^'"]+)['"]:\s*({NUMBER})""")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def wanted(key, includes):
    return not includes or any(token in key for token in includes)


def iter_metrics(path, includes):
    current_step = ""
    with path.open("r", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = strip_ansi(raw.rstrip("\n"))
            step_match = STEP_RE.search(line)
            if step_match:
                current_step = step_match.group(1)

            seen = set()  # type: Set[Tuple[str, str]]
            for pattern in (QUOTED_METRIC_RE, INLINE_METRIC_RE):
                for match in pattern.finditer(line):
                    key, value = match.group(1), match.group(2)
                    if key == "step" or not wanted(key, includes):
                        continue
                    item = (key, value)
                    if item in seen:
                        continue
                    seen.add(item)
                    yield {
                        "source": str(path),
                        "line": line_no,
                        "step": current_step,
                        "key": key,
                        "value": value,
                    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--include", action="append", default=[], help="Keep keys containing this substring.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dedupe-step-key", action="store_true", help="Drop repeated source/step/key/value rows.")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    emitted = set()
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "line", "step", "key", "value"])
        writer.writeheader()
        for path in args.logs:
            for row in iter_metrics(path, args.include):
                if args.dedupe_step_key:
                    marker = (row["source"], row["step"], row["key"], row["value"])
                    if marker in emitted:
                        continue
                    emitted.add(marker)
                writer.writerow(row)
                count += 1

    print(f"wrote {count} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
