"""Shared data preparation helpers for the project scripts."""

from __future__ import annotations

from typing import Iterable, Tuple


def infer_text_columns(column_names: Iterable[str]) -> Tuple[str, str]:
    names = set(column_names)
    if {"instruction", "response"}.issubset(names):
        return "instruction", "response"
    if {"query", "response"}.issubset(names):
        return "query", "response"
    raise ValueError(
        "Could not find query/response columns. Expected either instruction/response or query/response."
    )


def format_causal_example(query: str, response: str, eos_token: str) -> str:
    query = (query or "").strip()
    response = (response or "").strip()
    return f"Customer: {query}\nAgent: {response}{eos_token}"


def to_causal_lm_examples(dataset, tokenizer, max_length: int = 512, max_samples: int | None = None):
    """Convert a dataset of query/response pairs into causal LM examples."""

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    query_column, response_column = infer_text_columns(dataset.column_names)
    eos_token = tokenizer.eos_token or ""

    def tokenize_row(example):
        prompt = f"Customer: {(example.get(query_column) or '').strip()}\nAgent:"
        full_text = format_causal_example(
            example.get(query_column, ""), example.get(response_column, ""), eos_token
        )

        tokenized = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_ids = tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=max_length
        )["input_ids"]
        labels = list(tokenized["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        tokenized["labels"] = labels
        return tokenized

    return dataset.map(tokenize_row, remove_columns=list(dataset.column_names))
