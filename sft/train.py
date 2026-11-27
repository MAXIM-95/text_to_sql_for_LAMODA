import json
import math
from pathlib import Path
import random

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training #библиотека для LoRA/QLoRA и подготовки квантованной модели к обучению.
from torch.utils.data import Dataset #базовый класс PyTorch для создания собственного датасета.
from transformers import (
    AutoModelForCausalLM, #загрузка токенизатора 
    AutoTokenizer, #загрузка модели
    BitsAndBytesConfig, #настройка 4-битной квантизации
    Trainer, #сам цикл обучения
    TrainingArguments,
    default_data_collator,
)


class SqlSftDataset(Dataset):
    """Датасет для SFT: обучаем loss только на токенах ответа (SQL)."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for rec in records:
            messages = rec["messages"]
            prompt_messages = messages[:-1]
            full_messages = messages

            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                add_special_tokens=False,
            )

            input_ids = full_enc["input_ids"]
            attention_mask = full_enc["attention_mask"]
            labels = list(input_ids)

            # Маскируем токены промпта и padding-токены
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100
            for i in range(len(labels)):
                if attention_mask[i] == 0:
                    labels[i] = -100

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def load_jsonl_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Проверь установку PyTorch с поддержкой GPU.")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    data_path = Path("data/processed/sft_train.jsonl")
    output_dir = Path("outputs/qwen2.5-3b-sql-lora")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Не найден {data_path}. Сначала запусти: python sft/dataset_generator.py"
        )

    records = load_jsonl_records(data_path)
    rng = random.Random(42)
    rng.shuffle(records)

    val_size = max(1, int(len(records) * 0.15))
    val_records = records[:val_size]
    train_records = records[val_size:]

    print(f"Всего примеров: {len(records)} | train: {len(train_records)} | val: {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Загрузка модели {model_name} в 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = SqlSftDataset(train_records, tokenizer, max_length=1024)
    val_dataset = SqlSftDataset(val_records, tokenizer, max_length=1024)

    use_bf16 = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    print("Старт обучения...")
    train_result = trainer.train()
    print(f"Обучение завершено. Итоговый train loss: {train_result.training_loss:.4f}")

    final_adapter_dir = output_dir / "final_adapter"
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    print(f"LoRA-адаптер сохранён в: {final_adapter_dir}")


if __name__ == "__main__":
    main()