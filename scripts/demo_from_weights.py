"""Non-interactive demo runner for saved checkpoints."""

from __future__ import annotations

import argparse

from chatbot_from_weights import load_model, reply


PROMPTS = [
    "I need help canceling my order #A1029.",
    "My package says delivered but I never received it.",
    "How can I return a damaged item?",
    "I was charged twice for the same purchase.",
    "The app crashes when I try to check out.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["full", "lora", "prefix"], default="lora")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-turns", type=int, default=2)
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()

    model, tokenizer, source = load_model(args.method, allow_downloads=args.allow_downloads)
    print(f"Loaded model: {source}")
    print("-" * 72)

    history = []
    for prompt in PROMPTS:
        answer = reply(
            model,
            tokenizer,
            history,
            prompt,
            max_new_tokens=args.max_new_tokens,
            context_turns=args.context_turns,
        )
        print(f"Customer: {prompt}")
        print(f"Agent:    {answer}")
        print("-" * 72)
        history.extend([("Customer", prompt), ("Agent", answer)])


if __name__ == "__main__":
    main()