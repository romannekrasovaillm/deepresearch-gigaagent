"""
Supervised Fine-Tuning trainer for tool usage.

Trains the model to correctly format tool calls based on
SFT demonstrations generated from the library.
"""

import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name: str = "GigaChat-Lightning-Instruct"
    model_path: Optional[str] = None

    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_seq_length: int = 4096

    # Data
    train_data_path: str = ""
    val_data_path: Optional[str] = None
    val_split: float = 0.1

    # Optimization
    precision: str = "bf16"
    gradient_checkpointing: bool = True

    # Logging
    output_dir: str = "./outputs/sft"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500

    # Hardware
    num_gpus: int = 4
    use_fsdp: bool = True


class SFTTrainer:
    """
    Trainer for supervised fine-tuning on tool demonstrations.

    Uses transformers Trainer with custom data collation
    for multi-turn tool use format.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._trainer = None

    def _load_model(self):
        """Load model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path or self.config.model_name,
                trust_remote_code=True,
            )

            # Add special tokens for tool format if needed
            special_tokens = {
                "additional_special_tokens": [
                    "<|tool_call|>",
                    "<|tool_result|>",
                    "<|thought|>",
                    "<|answer|>",
                ]
            }
            self._tokenizer.add_special_tokens(special_tokens)

            # Load model
            dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16

            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path or self.config.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )

            # Resize embeddings for new tokens
            self._model.resize_token_embeddings(len(self._tokenizer))

            if self.config.gradient_checkpointing:
                self._model.gradient_checkpointing_enable()

        except ImportError as e:
            raise ImportError(f"Required packages not installed: {e}")

    def _format_demonstration(self, demo: dict) -> str:
        """
        Format a demonstration into training text.

        Converts turn-by-turn format to model input format.
        """
        parts = []

        # Task/question
        parts.append(f"<|user|>\n{demo['task']}")

        # Each turn
        for turn in demo.get("turns", []):
            # Thought
            if turn.get("thought"):
                parts.append(f"<|thought|>\n{turn['thought']}")

            # Tool call
            if turn.get("action"):
                action_input = json.dumps(turn.get("action_input", {}), ensure_ascii=False)
                parts.append(f"<|tool_call|>\n{turn['action']}({action_input})")

            # Tool result
            if turn.get("observation"):
                parts.append(f"<|tool_result|>\n{turn['observation']}")

        # Final answer
        if demo.get("final_answer"):
            parts.append(f"<|answer|>\n{demo['final_answer']}")

        return "\n".join(parts)

    def _load_data(self) -> tuple[list, list]:
        """Load and process training data."""
        train_data = []
        val_data = []

        # Load JSONL file
        data_path = Path(self.config.train_data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f]

        # Format demonstrations
        formatted = [
            {"text": self._format_demonstration(demo)}
            for demo in all_data
        ]

        # Split train/val
        if self.config.val_data_path:
            val_path = Path(self.config.val_data_path)
            if val_path.exists():
                with open(val_path, 'r', encoding='utf-8') as f:
                    val_raw = [json.loads(line) for line in f]
                val_data = [{"text": self._format_demonstration(d)} for d in val_raw]
            train_data = formatted
        else:
            # Random split
            import random
            random.shuffle(formatted)
            split_idx = int(len(formatted) * (1 - self.config.val_split))
            train_data = formatted[:split_idx]
            val_data = formatted[split_idx:]

        return train_data, val_data

    def train(self):
        """Run SFT training."""
        try:
            from transformers import (
                Trainer,
                TrainingArguments,
                DataCollatorForLanguageModeling,
            )
            from datasets import Dataset
        except ImportError:
            raise ImportError("transformers and datasets packages required")

        print("Loading model...")
        self._load_model()

        print("Loading data...")
        train_data, val_data = self._load_data()

        print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data) if val_data else None

        # Tokenize
        def tokenize_fn(examples):
            return self._tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )

        train_dataset = train_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )
        if val_dataset:
            val_dataset = val_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=["text"],
            )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            bf16=self.config.precision == "bf16",
            fp16=self.config.precision == "fp16",
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=self.config.eval_steps if val_dataset else None,
            save_total_limit=3,
            gradient_checkpointing=self.config.gradient_checkpointing,
            fsdp="full_shard auto_wrap" if self.config.use_fsdp else "",
            fsdp_config={
                "fsdp_transformer_layer_cls_to_wrap": "GigaChatDecoderLayer",
            } if self.config.use_fsdp else None,
            report_to="tensorboard",
            dataloader_num_workers=4,
        )

        # Create trainer
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self._tokenizer,
        )

        print("Starting training...")
        self._trainer.train()

        # Save final model
        print("Saving model...")
        self._trainer.save_model(self.config.output_dir)
        self._tokenizer.save_pretrained(self.config.output_dir)

        print(f"Training complete! Model saved to {self.config.output_dir}")

    def evaluate(self) -> dict:
        """Evaluate model on validation set."""
        if self._trainer is None:
            raise RuntimeError("Trainer not initialized. Run train() first.")

        metrics = self._trainer.evaluate()
        return metrics


def train_sft_cli():
    """CLI entry point for SFT training."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--train-data", type=str, help="Path to training data")
    parser.add_argument("--model", type=str, default="GigaChat-Lightning-Instruct")
    parser.add_argument("--output-dir", type=str, default="./outputs/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()

    # Load config from file or args
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = SFTConfig(**config_dict)
    else:
        config = SFTConfig(
            model_name=args.model,
            train_data_path=args.train_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    trainer = SFTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    train_sft_cli()
