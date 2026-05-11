"""Auto-fill overall_1_5 for human-eval sheet using simple heuristics.

This is a first-pass assistant score to reduce manual effort.
User should still spot-check a subset before submission.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def norm_tokens(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def score_row(query: str, reference: str, prediction: str) -> int:
    p = (prediction or "").strip()
    if not p:
        return 1

    # Very short / broken text tends to be low quality for this task.
    if len(p.split()) <= 3:
        return 1

    qt = norm_tokens(query)
    rt = norm_tokens(reference)
    pt = norm_tokens(prediction)

    # Relevance and lexical fit signals.
    q_sim = jaccard(qt, pt)
    r_sim = jaccard(rt, pt)

    # Penalize obvious garbage patterns.
    bad_markers = ["you :", "thanksToSupport", "tosueyourorder", "you'rewith"]
    if any(m.lower() in p.lower() for m in bad_markers):
        return 1

    # Generic but coherent fallback text.
    if q_sim < 0.02 and r_sim < 0.01:
        return 2

    # Strong lexical overlap usually indicates task-appropriate response.
    if r_sim >= 0.20:
        return 5
    if r_sim >= 0.12:
        return 4
    if r_sim >= 0.06 or q_sim >= 0.08:
        return 3

    return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/eval/human_eval_sheet_20.csv")
    parser.add_argument("--output", default="outputs/eval/human_eval_sheet_20_prefilled.csv")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = f.readline

    # reload header properly
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for r in rows:
        s = score_row(r.get("query", ""), r.get("reference", ""), r.get("prediction", ""))
        r["overall_1_5"] = str(s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote prefilled sheet: {out_path}")


if __name__ == "__main__":
    main()

