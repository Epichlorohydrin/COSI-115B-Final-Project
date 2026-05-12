"""Full fine-tuning script for causal LM (DialoGPT).

Usage (example):
python scripts/train_full.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
  --output_dir outputs/full --per_device_train_batch_size 2 --num_train_epochs 1
"""
import argparse
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

from utils import to_causal_lm_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/full')
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-large')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataset = to_causal_lm_examples(
        load_dataset(args.dataset, split='train'),
        tokenizer=tokenizer,
        max_length=512,
        max_samples=args.max_samples,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_total_limit=2,
        fp16=True if os.getenv('USE_FP16', '1') == '1' else False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    # Also save tokenizer into the same checkpoint directory so inference loaders can
    # find tokenizer.json / tokenizer_config.json when loading from the checkpoint.
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception:
        # Non-fatal: tokenizer saving should generally succeed; if it fails, keep training output.
        pass


if __name__ == '__main__':
    main()
