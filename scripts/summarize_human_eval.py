"""Summarize filled human evaluation sheet into per-method averages."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def to_float(x: str):
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/eval/human_eval_sheet.csv")
    parser.add_argument("--output", default="outputs/eval/human_eval_summary.csv")
    args = parser.parse_args()

    path = Path(args.input)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    bucket = defaultdict(lambda: {"h": [], "c": [], "f": [], "n": 0})
    for r in rows:
        m = r["method"]
        overall = to_float(r.get("overall_1_5", ""))
        h = to_float(r.get("helpfulness_1_5", ""))
        c = to_float(r.get("correctness_1_5", ""))
        fl = to_float(r.get("fluency_1_5", ""))

        # Fast mode: if only overall score is provided, reuse it.
        if overall is not None:
            if h is None:
                h = overall
            if c is None:
                c = overall
            if fl is None:
                fl = overall
        if h is not None:
            bucket[m]["h"].append(h)
        if c is not None:
            bucket[m]["c"].append(c)
        if fl is not None:
            bucket[m]["f"].append(fl)
        if h is not None and c is not None and fl is not None:
            bucket[m]["n"] += 1

    out_rows = []
    for method, d in sorted(bucket.items()):
        avg_h = sum(d["h"]) / len(d["h"]) if d["h"] else None
        avg_c = sum(d["c"]) / len(d["c"]) if d["c"] else None
        avg_f = sum(d["f"]) / len(d["f"]) if d["f"] else None
        out_rows.append(
            {
                "method": method,
                "num_fully_scored_rows": d["n"],
                "avg_helpfulness": "" if avg_h is None else f"{avg_h:.3f}",
                "avg_correctness": "" if avg_c is None else f"{avg_c:.3f}",
                "avg_fluency": "" if avg_f is None else f"{avg_f:.3f}",
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "num_fully_scored_rows",
                "avg_helpfulness",
                "avg_correctness",
                "avg_fluency",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
