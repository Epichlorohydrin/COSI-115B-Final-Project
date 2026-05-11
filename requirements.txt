from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/DialoGPT-large"
SMOKE_TEST_MODEL = "microsoft/DialoGPT-medium"
MODEL_CANDIDATES = [
    Path("outputs/lora"),
    Path("outputs/lora_smoke"),
    Path("outputs/lora_test"),
    Path("lora"),
]


def find_adapter_path():
    for candidate in MODEL_CANDIDATES:
        if not candidate.is_dir():
            continue
        if (candidate / "adapter_config.json").exists():
            return candidate
        if any((candidate / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin")):
            return candidate
    return None


def load_model():
    print("Loading model...", flush=True)
    use_smoke_model = not torch.cuda.is_available()
    model_name = SMOKE_TEST_MODEL if use_smoke_model else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    base_model.config.tie_word_embeddings = False

    adapter_path = find_adapter_path()
    if adapter_path is not None:
        print(f"Loading LoRA adapter from: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model_source = f"LoRA adapter at {adapter_path}"
    else:
        if use_smoke_model:
            print(f"No local LoRA adapter found, using {SMOKE_TEST_MODEL} for a fast smoke test.", flush=True)
        else:
            print("No local LoRA adapter found, using base DialoGPT model.", flush=True)
        model = base_model
        model_source = model_name

    model.eval()
    return model, tokenizer, model_source


def generate_response(model, tokenizer, customer_query, max_new_tokens=50):
    """Generate a customer service response."""
    prompt = f"Customer: {customer_query}\nAgent:"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response or tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def main():
    model, tokenizer, model_source = load_model()

    if torch.cuda.is_available():
        model = model.to("cuda")
        print("✓ Using GPU", flush=True)
    else:
        print("⚠ No GPU; using CPU (will be slow)", flush=True)

    cpu_mode = not torch.cuda.is_available()
    test_queries = [
        "I need help canceling my order",
        "How do I return an item?",
    ] if cpu_mode else [
        "I need help canceling my order",
        "How do I return an item?",
        "The app keeps crashing",
        "Can I get a refund?",
        "What is your return policy?",
    ]

    print("\n" + "=" * 60, flush=True)
    print("CUSTOMER SERVICE CHATBOT - RESPONSES", flush=True)
    print("=" * 60, flush=True)
    print(f"Model source: {model_source}", flush=True)

    for query in test_queries:
        response = generate_response(model, tokenizer, query, max_new_tokens=16 if cpu_mode else 50)
        print(f"\n👤 Customer: {query}", flush=True)
        print(f"🤖 Bot: {response}\n", flush=True)


if __name__ == "__main__":
    main()