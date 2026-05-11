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
.\.venv\Scripts\activate
pip install -r requirements.txt

# Full fine-tuning
python scripts/train_full.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--output_dir outputs/full --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000

# LoRA
python scripts/train_lora.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--output_dir outputs/lora --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000

# Prefix Tuning
python scripts/train_prefix.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
	--output_dir outputs/prefix --per_device_train_batch_size 2 --num_train_epochs 1 --max_samples 1000
```

## Run Chatbot From Saved Weights

If you already trained and saved checkpoints under `weights/`, run:

```bash
# Recommended: LoRA checkpoint
python scripts/chatbot_from_weights.py --method lora --allow-downloads

# Prefix checkpoint
python scripts/chatbot_from_weights.py --method prefix --allow-downloads

# Full fine-tune checkpoint (not recommended if loss collapsed)
python scripts/chatbot_from_weights.py --method full --allow-downloads
```

You can also run a non-interactive demo:

```bash
python scripts/demo_from_weights.py --method lora --allow-downloads
```

Note: LoRA/Prefix checkpoints store adapter weights and require the DialoGPT base model files.

## Evaluation

Run automatic generation metrics on a held-out split with:
- `zero_shot`
- `tfidf`
- `lora`
- `prefix`
- `full` (if broken, script records failure instead of crashing)

```bash
python scripts/evaluate_generations.py --allow-downloads --max-samples 1000 --output-dir outputs/eval
```

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
- Automatically downloaded on first training run

## Model

**Base Model**: `microsoft/DialoGPT-large` (~762M parameters)
- Pretrained conversational transformer
- Automatically downloaded on first run

## Requirements

- Python 3.9+
- GPU strongly recommended (for reasonable training times)
- CUDA 11.8+ if using GPU locally