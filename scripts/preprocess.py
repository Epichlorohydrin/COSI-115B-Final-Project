"""Interactive chatbot runner using saved checkpoints in weights/.

Examples:
  python scripts/chatbot_from_weights.py --method lora --allow-downloads
  python scripts/chatbot_from_weights.py --method prefix --allow-downloads
  python scripts/chatbot_from_weights.py --method full
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = PROJECT_ROOT / "weights"
CHECKPOINTS = {
    "full": WEIGHTS_ROOT / "full_finetune" / "checkpoint-1250",
    "lora": WEIGHTS_ROOT / "lora" / "checkpoint-1250",
    "prefix": WEIGHTS_ROOT / "prefix" / "checkpoint-1250",
}

DEFAULT_BASE = "microsoft/DialoGPT-large"


def _infer_base_model(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return DEFAULT_BASE
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return cfg.get("base_model_name_or_path", DEFAULT_BASE)


def load_model(method: str, allow_downloads: bool):
    ckpt = CHECKPOINTS[method]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    local_files_only = not allow_downloads
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if method == "full":
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), dtype=dtype, local_files_only=local_files_only
        )
        source = str(ckpt)
    else:
        base_name = _infer_base_model(ckpt)
        base = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=dtype, local_files_only=local_files_only
        )
        model = PeftModel.from_pretrained(base, str(ckpt), local_files_only=local_files_only)
        source = f"{base_name} + adapter({ckpt})"

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, tokenizer, source


def reply(model, tokenizer, message: str, max_new_tokens: int) -> str:
    prompt = f"Customer: {message.strip()}\nAgent:"
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            top_k=40,
            temperature=0.8,
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow downloading base model files from Hugging Face if needed.",
    )
    args = parser.parse_args()

    model, tokenizer, source = load_model(args.method, allow_downloads=args.allow_downloads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loaded model: {source}")
    print(f"Device: {device}")
    print("Type your message. Enter 'exit' to quit.")
    print("-" * 72)

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        bot = reply(model, tokenizer, user, max_new_tokens=args.max_new_tokens)
        print(f"Bot: {bot}")
        print("-" * 72)


if __name__ == "__main__":
    main()

