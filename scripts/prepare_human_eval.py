"""Prepare a hand-evaluation sheet from predictions.csv.

Creates a CSV with empty rating columns so you can score response quality.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(path: Path, rows):
    fieldnames = [
        "query_id",
        "query",
        "reference",
        "method",
        "prediction",
        "overall_1_5",
        "helpfulness_1_5",
        "correctness_1_5",
        "fluency_1_5",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/eval/predictions.csv")
    parser.add_argument("--output", default="outputs/eval/human_eval_sheet.csv")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["zero_shot", "tfidf", "lora", "prefix"],
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.predictions))
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    # Choose query set from zero_shot by default (stable order from file).
    anchor = by_method.get("zero_shot", [])
    selected = anchor[: args.num_queries]
    selected_queries = [r["query"] for r in selected]
    ref_map = {r["query"]: r["reference"] for r in selected}

    out = []
    for idx, q in enumerate(selected_queries, start=1):
        for m in args.methods:
            pred_row = next((r for r in by_method.get(m, []) if r["query"] == q), None)
            out.append(
                {
                    "query_id": idx,
                    "query": q,
                    "reference": ref_map.get(q, ""),
                    "method": m,
                    "prediction": "" if pred_row is None else pred_row["prediction"],
                    "overall_1_5": "",
                    "helpfulness_1_5": "",
                    "correctness_1_5": "",
                    "fluency_1_5": "",
                    "notes": "",
                }
            )

    write_rows(Path(args.output), out)
    print(f"Wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
