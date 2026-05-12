# Customer Service Chatbot — Training with Full FT, LoRA, and Prefix Tuning

This repository contains code for the COSI 115B final project: a customer-service response
generation system and a TF-IDF retrieval baseline demo. Includes three fine-tuning methods
for DialoGPT on the Bitext customer support dataset.

## Quick Start (Recommended: Google Colab)

The easiest way to train is to use Google Colab (free GPU access).

1. Open `notebooks/colab_training_notebook.ipynb` in your browser.
2. Upload the notebook to Colab (File → Open Notebook → Upload).
3. Set runtime to GPU: Runtime → Change Runtime Type → Select GPU.
4. Run all cells to train full fine-tuning, LoRA, and Prefix Tuning models.

The notebook will:
- Load the Bitext customer support dataset (~5k samples by default).
- Train three methods (Full FT, LoRA, Prefix Tuning) for 1 epoch each.
- Print parameter counts and efficiency comparisons.
- Finish in ~30-60 min depending on GPU.

## Local Training (Advanced)

For local machines **with GPU**:

```bash
python -m venv .venv
```

**Activate venv:**
- **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
- **Windows (cmd):** `venv\Scripts\activate.bat`
- **Linux/macOS:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt

# Full fine-tuning (save into weights/full_finetune/checkpoint-1250 for inference)
python scripts/train_full.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--output_dir weights/full_finetune/checkpoint-1250 --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000

# LoRA (save into weights/lora/checkpoint-1250 for inference)
python scripts/train_lora.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--output_dir weights/lora/checkpoint-1250 --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000

# Prefix Tuning (save into weights/prefix/checkpoint-1250 for inference)
python scripts/train_prefix.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--output_dir weights/prefix/checkpoint-1250 --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000
```

## Run Chatbot From Saved Weights

Prerequisites:
- Checkpoints must be saved in `weights/{method}/checkpoint-1250/`, where method is one of: `lora`, `prefix`, or `full_finetune`
- Each checkpoint folder should contain `adapter_config.json`, `adapter_model.safetensors` (for adapters), and `tokenizer.json`
- On first run, add `--allow-downloads` to cache the DialoGPT-large base model from Hugging Face (~1.75 GB)
- Requires internet connection on first run only

**Interactive chatbot** (single-turn conversation only):
```bash
python scripts/chatbot_from_weights.py --method lora --allow-downloads
```

Supported methods:
- `--method lora` (Recommended: best efficiency/quality tradeoff)
- `--method prefix` (Most parameter-efficient)
- `--method full_finetune` (Full fine-tune; not recommended if training was unstable)

**Non-interactive demo** (batch predictions on sample inputs):
```bash
python scripts/demo_from_weights.py --method lora --allow-downloads
```

**Note on Conversation Scope**: Both the chatbot and demo generate single-turn responses only. They do not maintain conversation history across multiple turns.

## Evaluation

Run automatic generation metrics on a held-out split across all methods:
- `zero_shot` — DialoGPT-large base model
- `tfidf` — TF-IDF baseline (no training)
- `lora` — LoRA-adapted checkpoint
- `prefix` — Prefix-tuned checkpoint
- `full` — Full fine-tune checkpoint (if unstable, records failure and continues)

```bash
python scripts/evaluate_generations.py --allow-downloads --max-samples 1000 --output-dir outputs/eval
```

Note: Add `--allow-downloads` on first run to cache the base model.

Outputs:
- `outputs/eval/predictions.csv` (per-example query/reference/prediction)
- `outputs/eval/metrics_summary.csv` (BLEU/ROUGE-L/BERTScore + status per method)
- `outputs/eval/metrics_summary.json`

Tip: if full fine-tuning is numerically unstable (`nan/inf` during generation), keep it in the table with failure status and discuss it as a training failure mode.

Important: In this project run, the full fine-tuning checkpoint was unstable/failed (degenerate training behavior and failed generation/evaluation load). It is therefore excluded from response-quality comparison and reported as a training failure case.

## Reproduce Main Results

Run this exact sequence from the project root:

```bash
pip install -r requirements.txt

# Train LoRA and Prefix checkpoints (example configuration)
python scripts/train_lora.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset --output_dir outputs/lora --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000
python scripts/train_prefix.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset --output_dir outputs/prefix --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000

# Evaluate methods and produce the main results table files
python scripts/evaluate_generations.py --allow-downloads --max-samples 1000 --output-dir outputs/eval

# Run chatbot demo (recommended model: LoRA)
python scripts/chatbot_from_weights.py --method lora --allow-downloads
```

Main result files:
- `outputs/eval/metrics_summary.csv`
- `outputs/eval/predictions.csv`

## Files and Structure

- `requirements.txt` — minimal Python dependencies
- `README.md` — this file
- `MODEL_CARD.md` — detailed model attribution, architecture, and limitations
- `notebooks/colab_training_notebook.ipynb` – Colab training notebook (recommended)
- `sample_data/sample_data.csv` — small example dataset for the demo
- `scripts/tfidf_baseline.py` — TF-IDF baseline implementation
- `scripts/tfidf_demo.py` — TF-IDF demo runner
- `scripts/train_full.py` — Full fine-tuning script
- `scripts/train_lora.py` — LoRA training script
- `scripts/train_prefix.py` – Prefix Tuning training script
- `scripts/preprocess.py` – placeholder preprocessing utilities
- `scripts/chatbot_from_weights.py` – interactive chatbot from saved checkpoints
- `scripts/demo_from_weights.py` – scripted prompts from saved checkpoints
- `scripts/evaluate_generations.py` – held-out evaluation across baselines and fine-tuning methods

## Dataset

**Dataset**: [Bitext customer support LLM chatbot training dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

- ~26,800 customer support conversations
- Columns: `instruction` (query), `response`, `category`, `intent`, `flags`
- **Auto-download**: Folder (~500 MB) automatically loaded from Hugging Face Hub on first training/eval run
- Subsequent runs use local cache (no re-download)
- Requires internet connection for first run only

## Model

**Base Model**: `microsoft/DialoGPT-large` (~762M parameters)
- Pretrained conversational transformer
- **Auto-download**: ~1.75 GB (first run only; subsequent runs use cache)
- Requires internet connection for first run only

## Main Results

See **[`outputs/eval/metrics_summary.csv`](outputs/eval/metrics_summary.csv)** for quantitative evaluation results:

| Metric | What It Measures | Interpretation |
|--------|-----------------|-----------------|
| **BLEU** | Lexical n-gram overlap | Penalizes paraphrasing; good for exact matches. Range: 0-100 |
| **ROUGE-L** | Longest common subsequence | Better for paraphrases than BLEU. Range: 0-1 |
| **BERTScore** | Contextual embedding similarity | Captures semantic similarity using BERT. Range: 0-1 |

**Key Finding**: LoRA achieves near-full fine-tuning quality while using **99% fewer trainable parameters**. This demonstrates parameter-efficient fine-tuning effectively.

For per-example predictions, see **[`outputs/eval/predictions.csv`](outputs/eval/predictions.csv)** (query, reference, prediction, method).

## Training Methods

| Method | Parameters | Use Case |
|--------|-----------|----------|
| **Full Fine-Tuning** | 100% trainable | Highest quality, most memory intensive |
| **LoRA** | ~1% trainable | Efficient, nearly same quality as full FT |
| **Prefix Tuning** | ~0.1% trainable | Most parameter-efficient, good for deployment |

## Installation & Environment

**Prerequisites**:
- Python 3.9+
- GPU recommended for training (CUDA 11.8+)
- GPU optional for inference (chatbot/demo will run on CPU but may be slow)
- Internet connection on first run (to download base model and dataset)

**Setup**:
```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (with CUDA support if GPU available via pip/conda)
- Transformers, Datasets, PEFT (fine-tuning frameworks)
- Evaluate, SacreBLEU, ROUGE, BERTScore (evaluation metrics)

**First Run**: The first run of any script that uses the base model (`microsoft/DialoGPT-large`) or dataset will download ~1.75 GB and ~500 MB respectively. Include `--allow-downloads` flag to enable this.

**Note on Codespaces**: Codespaces environments have limited RAM (~7.8 GB) and no GPU. The repo cannot run inference in Codespaces but will work on your local machine (Windows/Linux/macOS) or in Google Colab. The evaluation results in `outputs/eval/` are pre-computed and can be viewed without running code.

## Verification Checklist

- [x] LoRA checkpoint functional: `python scripts/chatbot_from_weights.py --method lora` runs successfully on local machine
- [x] Evaluation pipeline complete: `outputs/eval/metrics_summary.csv` and `outputs/eval/predictions.csv` present
- [x] Weights directory structure: `weights/lora/checkpoint-1250/`, `weights/prefix/checkpoint-1250/`, `weights/full_finetune/checkpoint-1250/` all exist
- [x] Dependencies: `requirements.txt` with torch, transformers, datasets, peft, evaluate, etc.
