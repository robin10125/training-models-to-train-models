from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .io_utils import atomic_write_text, write_jsonl


@dataclass(frozen=True)
class RlvrTrainingExample:
    prompt: str
    completion: str
    reward: float
    advantage: float
    source: str
    completion_token_ids: list[int] | None = None
    old_logprobs: list[float] | None = None


@dataclass(frozen=True)
class RlvrTrainerConfig:
    records_path: str
    output_dir: str
    model_id: str
    adapter_path: str | None
    max_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    load_in_4bit: bool
    algorithm: str
    clip_epsilon: float
    beta_kl: float
    resume: bool = True
    pause_path: str | None = None
    checkpoint_retention: str = "all"


class RlvrDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[RlvrTrainingExample], tokenizer: Any, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        prompt_ids = self.tokenizer(example.prompt, add_special_tokens=False).input_ids
        completion_ids = (
            list(example.completion_token_ids)
            if example.completion_token_ids is not None
            else self.tokenizer(example.completion, add_special_tokens=False).input_ids
        )
        if self.tokenizer.eos_token_id is not None and (
            not completion_ids or completion_ids[-1] != self.tokenizer.eos_token_id
        ):
            completion_ids = completion_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        if len(input_ids) > self.max_length:
            if example.old_logprobs is not None:
                raise ValueError(
                    "GRPO sample exceeds trainer_max_length; increase --trainer-max-length "
                    "so old logprobs are compared under the original prompt context."
                )
            input_ids = input_ids[-self.max_length :]
            labels = labels[-self.max_length :]

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantage": float(example.advantage),
            "old_logprobs": align_old_logprobs(labels, example.old_logprobs),
        }


def load_rlvr_examples(records_path: Path) -> list[RlvrTrainingExample]:
    rows = []
    for line_no, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not trainable_rlvr_record(row):
            continue
        reward = record_rlvr_reward(row)
        prompt = row.get("prompt")
        completion = row.get("completion")
        if reward is None or prompt is None or completion is None:
            continue
        rows.append(
            {
                "prompt": str(prompt),
                "completion": str(completion),
                "reward": float(reward),
                "source": f"{records_path}:{line_no}",
            }
        )

    if not rows:
        raise ValueError(f"No successful RLVR records with verified rewards found in {records_path}")

    rewards = [row["reward"] for row in rows]
    mean_reward = sum(rewards) / len(rewards)
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(1, len(rewards) - 1)
    std_reward = math.sqrt(variance)
    denom = std_reward if std_reward > 1e-8 else 1.0

    examples = []
    for row in rows:
        advantage = (row["reward"] - mean_reward) / denom
        examples.append(
            RlvrTrainingExample(
                prompt=row["prompt"],
                completion=row["completion"],
                reward=row["reward"],
                advantage=float(max(-5.0, min(5.0, advantage))),
                source=row["source"],
                completion_token_ids=None,
                old_logprobs=None,
            )
        )
    return examples


def load_grpo_examples(records_path: Path, *, min_group_size: int = 2) -> list[RlvrTrainingExample]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for line_no, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not trainable_rlvr_record(row):
            continue
        if record_rlvr_reward(row) is None or row.get("prompt") is None or row.get("completion") is None:
            continue
        if row.get("completion_token_ids") is None or row.get("old_logprobs") is None:
            continue
        key = group_key(row)
        row["_source"] = f"{records_path}:{line_no}"
        groups.setdefault(key, []).append(row)

    examples: list[RlvrTrainingExample] = []
    for rows in groups.values():
        if len(rows) < min_group_size:
            continue
        rewards = [float(record_rlvr_reward(row)) for row in rows]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / max(1, len(rewards) - 1)
        std_reward = math.sqrt(variance)
        denom = std_reward if std_reward > 1e-8 else 1.0
        for row, reward in zip(rows, rewards, strict=True):
            advantage = max(-5.0, min(5.0, (reward - mean_reward) / denom))
            examples.append(
                RlvrTrainingExample(
                    prompt=str(row["prompt"]),
                    completion=str(row["completion"]),
                    reward=reward,
                    advantage=float(advantage),
                    source=row["_source"],
                    completion_token_ids=[int(token_id) for token_id in row["completion_token_ids"]],
                    old_logprobs=[float(logprob) for logprob in row["old_logprobs"]],
                )
            )
    if not examples:
        raise ValueError(
            f"No GRPO groups with at least {min_group_size} successful records and old logprobs found in {records_path}"
        )
    return examples


def record_rlvr_reward(row: dict[str, Any]) -> float | None:
    reward = row.get("rlvr_reward")
    if reward is None:
        reward = row.get("verified_reward")
    return None if reward is None else float(reward)


def trainable_rlvr_record(row: dict[str, Any]) -> bool:
    status = row.get("status")
    if status == "success":
        return True
    return status in {"failed", "invalid_completion"} and row.get("rlvr_reward") is not None


def group_key(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", ""))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{row.get('prompt_id', '')}:{row.get('generator_checkpoint', '')}:{prompt_hash}"


def align_old_logprobs(labels: list[int], old_logprobs: list[float] | None) -> list[float | None]:
    aligned: list[float | None] = [None] * len(labels)
    if old_logprobs is None:
        return aligned
    cursor = 0
    for index, label in enumerate(labels):
        if label == -100:
            continue
        if cursor < len(old_logprobs):
            aligned[index] = float(old_logprobs[cursor])
        cursor += 1
    return aligned


def collate_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = []
    attention_mask = []
    labels = []
    advantages = []
    old_logprobs = []
    for row in batch:
        pad_len = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(row["attention_mask"] + [0] * pad_len)
        labels.append(row["labels"] + [-100] * pad_len)
        advantages.append(row["advantage"])
        row_old_logprobs = row.get("old_logprobs", [None] * len(row["input_ids"]))
        old_logprobs.append(
            [
                -1.0e9 if value is None else float(value)
                for value in row_old_logprobs + [None] * pad_len
            ]
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "old_logprobs": torch.tensor(old_logprobs, dtype=torch.float32),
    }


def weighted_completion_loss(logits: torch.Tensor, labels: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    token_loss = torch.nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shifted_labels.shape)
    token_mask = (shifted_labels != -100).float()
    sequence_loss = (token_loss * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1.0)
    weights = torch.nn.functional.softplus(advantages.to(sequence_loss.device))
    return (sequence_loss * weights).mean()


def grpo_clipped_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    *,
    clip_epsilon: float,
    beta_kl: float,
) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_old_logprobs = old_logprobs[:, 1:].contiguous().to(shifted_logits.device)
    shifted_advantages = advantages.to(shifted_logits.device).view(-1, 1)
    token_mask = (shifted_labels != -100) & (shifted_old_logprobs > -1.0e8)

    safe_labels = shifted_labels.masked_fill(~token_mask, 0)
    new_logprobs = torch.log_softmax(shifted_logits.float(), dim=-1).gather(
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    log_ratio = (new_logprobs - shifted_old_logprobs).masked_fill(~token_mask, 0.0).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * shifted_advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratio * shifted_advantages
    policy_objective = torch.minimum(unclipped, clipped)
    approx_kl = log_ratio
    token_loss = -(policy_objective - beta_kl * approx_kl.square())
    masked_loss = token_loss.masked_fill(~token_mask, 0.0)
    return masked_loss.sum() / token_mask.float().sum().clamp_min(1.0)


def train_rlvr(config: RlvrTrainerConfig) -> dict[str, Any]:
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("RLVR training requires transformers, peft, accelerate, and bitsandbytes.") from exc

    output_dir = Path(config.output_dir)
    if config.checkpoint_retention not in {"all", "latest"}:
        raise ValueError("checkpoint_retention must be one of: all, latest")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_trainer_checkpoint(output_dir) if config.resume else None
    start_epoch = 0
    losses: list[float] = []
    effective_adapter_path = config.adapter_path
    if checkpoint is not None:
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        start_epoch = int(state.get("next_epoch", 0))
        losses = [float(loss) for loss in state.get("losses", [])]
        effective_adapter_path = checkpoint.as_posix()

    if config.algorithm == "grpo":
        examples = load_grpo_examples(Path(config.records_path))
    elif config.algorithm == "weighted_sft":
        examples = load_rlvr_examples(Path(config.records_path))
    else:
        raise ValueError(f"Unsupported RLVR trainer algorithm: {config.algorithm}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if effective_adapter_path:
        model = PeftModel.from_pretrained(model, effective_adapter_path, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
    model.train()

    dataset = RlvrDataset(examples, tokenizer, config.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    if start_epoch >= config.epochs:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        metrics = trainer_metrics(config, examples, losses, status="completed_from_checkpoint")
        atomic_write_text(output_dir / "trainer_metrics.json", json.dumps(metrics, indent=2))
        return metrics

    for epoch in range(start_epoch, config.epochs):
        for batch in dataloader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            if config.algorithm == "grpo":
                loss = grpo_clipped_loss(
                    outputs.logits,
                    batch["labels"],
                    batch["advantages"],
                    batch["old_logprobs"],
                    clip_epsilon=config.clip_epsilon,
                    beta_kl=config.beta_kl,
                )
            else:
                loss = weighted_completion_loss(outputs.logits, batch["labels"], batch["advantages"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
        write_jsonl(output_dir / "trainer_events.jsonl", [{"epoch": epoch, "loss": losses[-1]}], append=True)
        save_trainer_checkpoint(output_dir, model, tokenizer, config, epoch + 1, losses)
        if config.checkpoint_retention == "latest":
            prune_old_trainer_checkpoints(output_dir, keep_epoch=epoch + 1)
        if pause_requested(config.pause_path):
            return trainer_metrics(config, examples, losses, status="paused")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = trainer_metrics(config, examples, losses, status="completed")
    atomic_write_text(output_dir / "trainer_metrics.json", json.dumps(metrics, indent=2))
    return metrics


def trainer_metrics(
    config: RlvrTrainerConfig,
    examples: list[RlvrTrainingExample],
    losses: list[float],
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "config": asdict(config),
        "algorithm": config.algorithm,
        "examples": len(examples),
        "mean_reward": sum(example.reward for example in examples) / len(examples),
        "losses": losses,
        "final_loss": losses[-1] if losses else None,
        "status": status,
    }


def save_trainer_checkpoint(
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    config: RlvrTrainerConfig,
    next_epoch: int,
    losses: list[float],
) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"epoch_{next_epoch:04d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    state = {
        "config": asdict(config),
        "next_epoch": next_epoch,
        "losses": losses,
        "updated_at": time.time(),
    }
    atomic_write_text(checkpoint_dir / "trainer_state.json", json.dumps(state, indent=2))


def latest_trainer_checkpoint(output_dir: Path) -> Path | None:
    checkpoint_root = output_dir / "checkpoints"
    if not checkpoint_root.exists():
        return None
    checkpoints = [
        path for path in checkpoint_root.glob("epoch_*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.name.rsplit("_", 1)[-1]))


def prune_old_trainer_checkpoints(output_dir: Path, *, keep_epoch: int) -> None:
    checkpoint_root = output_dir / "checkpoints"
    keep_path = checkpoint_root / f"epoch_{keep_epoch:04d}"
    for path in checkpoint_root.glob("epoch_*"):
        if path.is_dir() and path != keep_path:
            shutil.rmtree(path)


def pause_requested(pause_path: str | None) -> bool:
    return pause_path is not None and Path(pause_path).exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter from EUREKA RLVR records.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--algorithm", choices=["weighted_sft", "grpo"], default="grpo")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--pause-path", default=None)
    parser.add_argument("--checkpoint-retention", choices=["all", "latest"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RlvrTrainerConfig(
        records_path=str(args.records),
        output_dir=str(args.output_dir),
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=not args.no_4bit,
        algorithm=args.algorithm,
        clip_epsilon=args.clip_epsilon,
        beta_kl=args.beta_kl,
        resume=not args.no_resume,
        pause_path=args.pause_path,
        checkpoint_retention=args.checkpoint_retention,
    )
    metrics = train_rlvr(config)
    print(json.dumps({"output_dir": args.output_dir.as_posix(), "final_loss": metrics["final_loss"]}, indent=2))


if __name__ == "__main__":
    main()
