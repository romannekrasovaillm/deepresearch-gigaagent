"""SFT Trainer for tool usage format."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


@dataclass
class SFTConfig:
    """Configuration for SFT training."""

    model_name: str = "GigaChat-Lightning-Instruct"
    model_path: Optional[str] = None

    output_dir: str = "./checkpoints/sft"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    max_length: int = 4096
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    bf16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10


class ToolSFTDataset(Dataset):
    """Dataset for tool usage SFT."""

    SYSTEM_PROMPT = """You are a research agent with access to a library of scientific papers.
You can use these tools:
- grep_search(pattern, path): Search for pattern in files
- read_file_chunk(path, start, num_lines): Read file content
- find_files(name_pattern): Find files by name
- list_papers(topic): List papers on topic
- execute_python(code): Execute Python code
- add_to_notes(text, section): Save notes
- read_notes(): Read saved notes
- verify_quote(path, snippet): Verify quote exists

To use a tool, output JSON: {"name": "tool_name", "arguments": {"arg": "value"}}
Provide final answer with [ANSWER] prefix."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path: Path) -> list[dict]:
        """Load SFT samples from JSONL."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") == "sft_demo" and data.get("sft_trace"):
                    samples.append(data)
        return samples

    def _format_sample(self, sample: dict) -> str:
        """Format sample into conversation format."""
        messages = []

        # System message
        messages.append(f"<|system|>\n{self.SYSTEM_PROMPT}")

        # User question
        messages.append(f"<|user|>\n{sample['question']}")

        # Assistant turns (from trace)
        assistant_parts = []
        for turn in sample.get("sft_trace", []):
            thought = turn.get("thought", "")
            action = turn.get("action")
            observation = turn.get("observation", "")

            if thought:
                assistant_parts.append(f"Thought: {thought}")
            if action:
                action_json = json.dumps(action, ensure_ascii=False)
                assistant_parts.append(f"Action: {action_json}")
            if observation:
                assistant_parts.append(f"Observation: {observation}")

        # Final answer
        if sample.get("golden_answer"):
            assistant_parts.append(f"\n[ANSWER] {sample['golden_answer']}")

        messages.append(f"<|assistant|>\n" + "\n".join(assistant_parts))

        return "\n".join(messages)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = self._format_sample(sample)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
        }


class SFTTrainer:
    """Trainer for SFT on tool usage demonstrations."""

    def __init__(self, config: SFTConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self):
        """Load tokenizer."""
        model_path = self.config.model_path or self.config.model_name
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_model(self):
        """Load model with optional LoRA."""
        model_path = self.config.model_path or self.config.model_name

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            trust_remote_code=True,
            device_map="auto",
        )

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if self.config.use_lora and LoraConfig is not None:
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model

    def train(
        self,
        train_path: str | Path,
        eval_path: Optional[str | Path] = None,
    ) -> None:
        """Run SFT training."""
        # Create datasets
        train_dataset = ToolSFTDataset(
            train_path,
            self.tokenizer,
            self.config.max_length,
        )

        eval_dataset = None
        if eval_path:
            eval_dataset = ToolSFTDataset(
                eval_path,
                self.tokenizer,
                self.config.max_length,
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            bf16=self.config.bf16,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            logging_steps=self.config.logging_steps,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["wandb"],
            run_name="deepresearch-sft",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save final model
        trainer.save_model(self.config.output_dir + "/final")
        self.tokenizer.save_pretrained(self.config.output_dir + "/final")

    def save_model(self, path: str | Path) -> None:
        """Save model to path."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
