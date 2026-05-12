# Model Card: DialoGPT Customer Service Chatbot

## Overview
This repository contains fine-tuned variants of `microsoft/DialoGPT-large` trained on the Bitext customer support dataset using three parameter-efficient methods: Full Fine-Tuning, LoRA, and Prefix Tuning.

## Base Model
- **Model**: `microsoft/DialoGPT-large`
- **Architecture**: Transformer-based causal language model (~762M parameters)
- **Paper**: [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536)
- **License**: OpenRAIL-M
- **Auto-download**: ~1.75 GB (first run only; subsequent runs use cache)

## Training Data
- **Dataset**: [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Size**: ~26,800 conversation examples
- **Domain**: Customer support queries and responses
- **Columns**: `instruction` (query), `response`, `category`, `intent`, `flags`
- **License**: OpenRAIL-M
- **Auto-download**: ~500 MB (first run only; subsequent runs use cache)

## Training Frameworks & Dependencies
- **PyTorch**: 2.0+
- **HuggingFace Transformers**: 4.30.0+
- **PEFT (Parameter-Efficient Fine-Tuning)**: 0.19.1
- **Evaluation**: BLEU (SacreBLEU), ROUGE-L, BERTScore

## Fine-Tuning Methods

### 1. Full Fine-Tuning
- **Trainable Parameters**: 100% (~762M)
- **Memory**: High (GPU recommended)
- **Inference Speed**: Baseline
- **Quality**: Best, but may overfit on smaller samples
- **Config File**: `scripts/train_full.py`

### 2. LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: ~1%
- **Memory**: Much lower than full fine-tuning
- **Inference Speed**: Negligible overhead
- **Quality**: Strong tradeoff between quality and efficiency
- **Config**: r=8, alpha=32, dropout=0.1
- **Config File**: `scripts/train_lora.py`

### 3. Prefix Tuning
- **Trainable Parameters**: ~0.1%
- **Memory**: Highest efficiency
- **Inference Speed**: Small overhead
- **Quality**: Good for deployment-efficient tuning
- **Config**: num_virtual_tokens=20
- **Config File**: `scripts/train_prefix.py`

## Results

See [`outputs/eval/metrics_summary.csv`](outputs/eval/metrics_summary.csv) for quantitative results:
- **BLEU**: lexical n-gram overlap
- **ROUGE-L**: longest common subsequence
- **BERTScore**: contextual semantic similarity

**Key Finding**: LoRA achieves near-full fine-tuning quality while using far fewer trainable parameters.

## Usage

### Interactive Chatbot
```bash
python scripts/chatbot_from_weights.py --method lora --allow-downloads
```

### Batch Inference
```bash
python scripts/demo_from_weights.py --method lora --allow-downloads
```

### Evaluation
```bash
python scripts/evaluate_generations.py --allow-downloads --max-samples 1000
```

## Limitations
1. **Single-Turn**: Responses are generated for one user message at a time; conversation history is not tracked.
2. **Domain-Specific**: Fine-tuned for customer support, so it may not generalize well to unrelated domains.
3. **No Safety Filtering**: The model is not optimized for toxicity or safety moderation.
4. **Context Length**: Maximum sequence length is 512 tokens in the current scripts.
5. **CPU Inference**: Works on CPU but is slower than GPU.

## Requirements
- Python 3.9+
- Internet connection on first run
- GPU recommended for training

## Environment Setup
```bash
pip install -r requirements.txt
```

## Attribution
- Base model: Microsoft DialoGPT-large
- Dataset: Bitext customer support dataset
- Fine-tuning methods: Full FT, LoRA, Prefix Tuning
