"""Evaluate chatbot methods on a held-out split with automatic metrics.

Methods supported:
- zero_shot (base DialoGPT)
- tfidf
- lora (adapter in weights/lora/checkpoint-1250)
- prefix (adapter in weights/prefix/checkpoint-1250)
- full (checkpoint in weights/full_finetune/checkpoint-1250)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tfidf_baseline import TFIDFBaseline


BASE_MODEL = "microsoft/DialoGPT-large"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "weights"
CHECKPOINTS = {
    "full": WEIGHTS / "full_finetune" / "checkpoint-1250",
    "lora": WEIGHTS / "lora" / "checkpoint-1250",
    "prefix": WEIGHTS / "prefix" / "checkpoint-1250",
}


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().split())


def _try_load_metric(name: str):
    try:
        import evaluate

        return evaluate.load(name)
    except Exception:
        return None


def build_splits(dataset_name: str, max_samples: int, seed: int) -> Tuple[Dataset, Dataset]:
    ds = load_dataset(dataset_name, split="train")
    ds = ds.select(range(min(max_samples, len(ds))))
    split = ds.train_test_split(test_size=0.2, seed=seed)
    return split["train"], split["test"]


def format_prompt(query: str) -> str:
    return f"Customer: {_normalize(query)}\nAgent:"


def decode_new_tokens(tokenizer, output_ids, prompt_len: int) -> str:
    text = tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()
    return _normalize(text)


def generate_with_model(model, tokenizer, query: str, max_new_tokens: int) -> str:
    prompt = format_prompt(query)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return decode_new_tokens(tokenizer, out[0], input_ids.shape[-1])


def load_base_model_and_tokenizer(allow_downloads: bool):
    local_files_only = not allow_downloads
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer


def load_method_model(method: str, allow_downloads: bool):
    if method == "zero_shot":
        return load_base_model_and_tokenizer(allow_downloads)

    if method in ("lora", "prefix"):
        ckpt = CHECKPOINTS[method]
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        base, tokenizer = load_base_model_and_tokenizer(allow_downloads)
        model = PeftModel.from_pretrained(base, str(ckpt), local_files_only=not allow_downloads)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model, tokenizer

    if method == "full":
        ckpt = CHECKPOINTS["full"]
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        local_files_only = not allow_downloads
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            dtype=dtype,
            local_files_only=local_files_only,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        return model, tokenizer

    raise ValueError(f"Unknown method: {method}")


def evaluate_method(
    method: str,
    train_ds: Dataset,
    test_ds: Dataset,
    max_new_tokens: int,
    allow_downloads: bool,
) -> Tuple[List[Dict[str, str]], str]:
    rows: List[Dict[str, str]] = []
    status = "ok"

    try:
        if method == "tfidf":
            retriever = TFIDFBaseline()
            train_queries = [_normalize(x) for x in train_ds["instruction"]]
            train_responses = [_normalize(x) for x in train_ds["response"]]
            retriever.fit(train_queries, train_responses)
            for ex in test_ds:
                pred = _normalize(retriever.retrieve(_normalize(ex["instruction"]), top_k=1)[0][0])
                rows.append(
                    {
                        "method": method,
                        "query": _normalize(ex["instruction"]),
                        "reference": _normalize(ex["response"]),
                        "prediction": pred,
                    }
                )
            return rows, status

        model, tokenizer = load_method_model(method, allow_downloads=allow_downloads)
        for ex in test_ds:
            query = _normalize(ex["instruction"])
            reference = _normalize(ex["response"])
            try:
                pred = generate_with_model(model, tokenizer, query, max_new_tokens=max_new_tokens)
            except Exception as gen_err:
                status = f"failed_generation: {type(gen_err).__name__}"
                pred = ""
            rows.append(
                {
                    "method": method,
                    "query": query,
                    "reference": reference,
                    "prediction": _normalize(pred),
                }
            )
        return rows, status
    except Exception as e:
        status = f"failed_load_or_eval: {type(e).__name__}"
        for ex in test_ds:
            rows.append(
                {
                    "method": method,
                    "query": _normalize(ex["instruction"]),
                    "reference": _normalize(ex["response"]),
                    "prediction": "",
                }
            )
        return rows, status


def compute_metrics(references: List[str], predictions: List[str]) -> Dict[str, float | None]:
    results: Dict[str, float | None] = {"bleu": None, "rougeL": None, "bertscore_f1": None}

    bleu = _try_load_metric("bleu")
    rouge = _try_load_metric("rouge")
    bertscore = _try_load_metric("bertscore")

    if bleu is not None:
        try:
            res = bleu.compute(
                predictions=predictions,
                references=[[r] for r in references],
            )
            results["bleu"] = float(res["bleu"])
        except Exception:
            pass

    if rouge is not None:
        try:
            res = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
            results["rougeL"] = float(res.get("rougeL", 0.0))
        except Exception:
            pass

    if bertscore is not None:
        try:
            res = bertscore.compute(
                predictions=predictions,
                references=references,
                model_type="microsoft/deberta-xlarge-mnli",
                lang="en",
            )
            f1 = res.get("f1", [])
            if f1:
                results["bertscore_f1"] = float(sum(f1) / len(f1))
        except Exception:
            pass

    return results


def write_outputs(output_dir: Path, all_rows: List[Dict[str, str]], summary_rows: List[Dict[str, str]]):
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_path = output_dir / "predictions.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "query", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(all_rows)

    summary_csv = output_dir / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "status", "bleu", "rougeL", "bertscore_f1", "notes"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = output_dir / "metrics_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["zero_shot", "tfidf", "lora", "prefix", "full"],
        choices=["zero_shot", "tfidf", "lora", "prefix", "full"],
    )
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()

    train_ds, test_ds = build_splits(args.dataset, max_samples=args.max_samples, seed=args.seed)
    print(f"Train size: {len(train_ds)} | Test size: {len(test_ds)}")
    print(f"Methods: {args.methods}")

    all_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    for method in args.methods:
        print(f"\n=== Evaluating {method} ===")
        rows, status = evaluate_method(
            method=method,
            train_ds=train_ds,
            test_ds=test_ds,
            max_new_tokens=args.max_new_tokens,
            allow_downloads=args.allow_downloads,
        )
        refs = [r["reference"] for r in rows]
        preds = [r["prediction"] for r in rows]
        metrics = compute_metrics(refs, preds) if status == "ok" else {"bleu": None, "rougeL": None, "bertscore_f1": None}
        note = "full checkpoint unstable if generation fails with nan/inf" if method == "full" else ""
        summary = {
            "method": method,
            "status": status,
            "bleu": metrics["bleu"],
            "rougeL": metrics["rougeL"],
            "bertscore_f1": metrics["bertscore_f1"],
            "notes": note,
        }
        all_rows.extend(rows)
        summary_rows.append(summary)
        print(summary)

    write_outputs(Path(args.output_dir), all_rows, summary_rows)
    print(f"\nSaved predictions and metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
