"""Run customer-service demo conversations from local training checkpoints.

Usage examples:
  python scripts/demo_from_weights.py --method lora
  python scripts/demo_from_weights.py --method full
  python scripts/demo_from_weights.py --method prefix
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "microsoft/DialoGPT-medium"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "weights"
PATHS = {
    "full": ROOT / "full_finetune" / "checkpoint-1250",
    "lora": ROOT / "lora" / "checkpoint-1250",
    "prefix": ROOT / "prefix" / "checkpoint-1250",
}


PROMPTS = [
    "I need help canceling my order #A1029.",
    "My package says delivered but I never received it.",
    "How can I return a damaged item?",
    "I was charged twice for the same purchase.",
    "The app crashes when I try to check out.",
]

DEFAULT_MAX_NEW_TOKENS = 32


def _cached_hf_snapshot(model_id: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    refs_main = repo_dir / "refs" / "main"
    if not refs_main.exists():
        return None

    snapshot_hash = refs_main.read_text(encoding="utf-8").strip()
    snapshot_dir = repo_dir / "snapshots" / snapshot_hash
    return snapshot_dir if snapshot_dir.exists() else None


def _infer_base_model_from_adapter(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return DEFAULT_BASE_MODEL
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
    except Exception:
        return DEFAULT_BASE_MODEL


def _get_model_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_model(method: str, allow_downloads: bool = False):
    ckpt = PATHS[method]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    dtype = _get_model_dtype()
    local_files_only = not allow_downloads

    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt),
            dtype=dtype,
            local_files_only=local_files_only,
        )
        source = str(ckpt)
    else:
        base_model_name = _infer_base_model_from_adapter(ckpt)
        if local_files_only:
            cached_base = _cached_hf_snapshot(base_model_name)
            if cached_base is None:
                raise FileNotFoundError(
                    f"Cached base model not found for {base_model_name}. "
                    "Re-run with --allow-downloads once to populate the cache."
                )
            base_model_name = str(cached_base)

        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        model = PeftModel.from_pretrained(base, str(ckpt), local_files_only=local_files_only)
        source = f"{base_model_name} + adapter({ckpt})"

    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to(torch.float32)
    model.eval()
    return model, tokenizer, source


def generate(model, tokenizer, query: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    prompt = f"{query.strip()}\n"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            min_new_tokens=4,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["full", "lora", "prefix"], default="lora")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Hugging Face downloads if the base model is not already cached locally.",
    )
    args = parser.parse_args()

    model, tokenizer, source = load_model(args.method, allow_downloads=args.allow_downloads)
    print(f"Loaded: {source}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 72)

    for p in PROMPTS:
        resp = generate(model, tokenizer, p, max_new_tokens=args.max_new_tokens)
        print(f"Customer: {p}")
        print(f"Agent:    {resp}")
        print("-" * 72)


if __name__ == "__main__":
    main()