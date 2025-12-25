"""
Supervised Fine-Tuning (SFT) for tool usage format.

Trains the model on demonstration traces to learn:
- Tool call format
- Multi-turn interaction pattern
- When to use which tool
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from .data import load_sft_dataset


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    model_name: str = "ai-sage/GigaChat-Lightning-Instruct"
    output_dir: str = "./checkpoints/sft"
    data_path: str = "./data/datasets/sft_train.jsonl"

    # Training params
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_length: int = 4096

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for tool usage.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def prepare_dataset(self) -> Dataset:
        """Load and prepare dataset."""
        dataset = load_sft_dataset(Path(self.config.data_path))

        def tokenize(examples):
            # Format messages into single text
            texts = []
            for messages in examples["messages"]:
                text = ""
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        text += f"User: {content}\n\n"
                    else:
                        text += f"Assistant: {content}\n\n"
                texts.append(text)

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )

    def train(self):
        """Run SFT training."""
        if self.model is None:
            self.load_model()

        dataset = self.prepare_dataset()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            bf16=self.config.bf16,
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            report_to="wandb",
            run_name="deepresearch-sft",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            ),
        )

        trainer.train()
        trainer.save_model()

        print(f"SFT training complete! Model saved to {self.config.output_dir}")


def train_sft(
    model_name: str,
    data_path: str,
    output_dir: str,
    **kwargs,
):
    """
    Convenience function for SFT training.

    Args:
        model_name: HuggingFace model name
        data_path: Path to SFT data
        output_dir: Output directory for checkpoints
        **kwargs: Additional config options
    """
    config = SFTConfig(
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        **kwargs,
    )

    trainer = SFTTrainer(config)
    trainer.train()
