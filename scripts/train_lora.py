"""LoRA fine-tuning script using PEFT.

Usage example:
python scripts/train_lora.py --dataset bitext/Bitext-customer-support-llm-chatbot-training-dataset \
  --output_dir outputs/lora --model microsoft/DialoGPT-large --num_train_epochs 1
"""
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForLanguageModeling

from utils import to_causal_lm_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default='microsoft/DialoGPT-large')
    parser.add_argument('--output_dir', type=str, default='outputs/lora')
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

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
    model.push_to_hub = False
    model.save_pretrained(args.output_dir)
    # Save tokenizer alongside adapter files so loaders can find tokenizer in checkpoint dir
    try:
        tokenizer.save_pretrained(args.output_dir)
    except Exception:
        pass


if __name__ == '__main__':
    main()
