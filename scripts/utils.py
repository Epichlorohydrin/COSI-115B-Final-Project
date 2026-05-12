"""Utility helpers for dataset loading and preprocessing."""
from typing import List, Dict, Optional
from datasets import load_dataset, Dataset


def load_hf_dataset(name: str, split: str = 'train') -> Dataset:
    """Load a HuggingFace dataset by name. Returns a Dataset object."""
    ds = load_dataset(name, split=split)
    return ds


def extract_pairs_from_dataset(ds, query_field: str = None, response_field: str = None):
    """Try to extract (query, response) pairs from a HF dataset.

    If `query_field` and `response_field` are provided, use them. Otherwise attempt
    to guess common fields.
    Returns a dict with keys 'query' and 'response'.
    """
    cols = ds.column_names
    if query_field and response_field:
        return {'query': ds[query_field], 'response': ds[response_field]}

    # Common heuristics
    if 'query' in cols and 'response' in cols:
        return {'query': ds['query'], 'response': ds['response']}
    if 'text' in cols:
        # assume each example is a single text; cannot extract pairs automatically
        raise ValueError("Dataset has 'text' column only; please provide mapping to (query,response)")
    if 'conversation' in cols or 'conversations' in cols:
        # leave to user to preprocess; return raw column
        key = 'conversation' if 'conversation' in cols else 'conversations'
        return {'conversation': ds[key]}

    # fallback: return all columns for user inspection
    raise ValueError(f"Could not find obvious query/response fields. Columns: {cols}")


def infer_pair_fields(ds) -> Dict[str, str]:
    """Infer likely input/target fields from a dataset schema."""
    cols = set(ds.column_names)
    input_candidates = ["instruction", "query", "question", "prompt", "text", "input"]
    target_candidates = ["response", "answer", "output", "completion", "target"]

    input_field = next((c for c in input_candidates if c in cols), None)
    target_field = next((c for c in target_candidates if c in cols), None)

    if input_field and target_field:
        return {"input": input_field, "target": target_field}

    if "conversation" in cols:
        return {"conversation": "conversation"}
    if "conversations" in cols:
        return {"conversation": "conversations"}

    raise ValueError(f"Could not infer input/target fields from columns: {sorted(cols)}")


def to_causal_lm_examples(ds: Dataset, tokenizer, max_length: int = 512, max_samples: Optional[int] = None):
    """Convert a dataset with paired text fields into tokenized causal-LM examples."""
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    fields = infer_pair_fields(ds)

    if "conversation" in fields:
        raise ValueError(
            "Conversation-style datasets need a custom preprocessing step before training."
        )

    input_field = fields["input"]
    target_field = fields["target"]

    def _tokenize(example):
        text = f"{example[input_field]}\n{example[target_field]}"
        return tokenizer(text, truncation=True, max_length=max_length)

    tokenized = ds.map(_tokenize, batched=False)
    return tokenized
