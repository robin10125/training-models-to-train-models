from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from .adapters import get_adapter
from .rewards import RewardCandidate, RewardExpression


DEFAULT_HF_MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"


@dataclass(frozen=True)
class HfGeneratorConfig:
    model_id: str = DEFAULT_HF_MODEL_ID
    adapter_path: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    load_in_4bit: bool = True


class HfRewardGenerator:
    def __init__(self, config: HfGeneratorConfig) -> None:
        self.config = config
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "The HF generator requires transformers, accelerate, and bitsandbytes. "
                "Install the project dependencies again after updating requirements.txt."
            ) from exc

        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        if config.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Loading a generator adapter requires peft.") from exc
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path)
        self.model.eval()
        self.generator_checkpoint = (
            config.model_id if config.adapter_path is None else f"{config.model_id}+adapter:{config.adapter_path}"
        )

    def generate_population(
        self,
        *,
        task: str,
        population: int,
        generation: int,
        best_expression: str | None = None,
        best_score: float | None = None,
    ) -> list[RewardCandidate]:
        return [
            self.generate_candidate(
                task=task,
                index=index,
                generation=generation,
                best_expression=best_expression,
                best_score=best_score,
            )
            for index in range(population)
        ]

    def generate_candidate(
        self,
        *,
        task: str,
        index: int,
        generation: int,
        best_expression: str | None,
        best_score: float | None,
    ) -> RewardCandidate:
        adapter = get_adapter(task)
        prompt = ""
        last_error: Exception | None = None
        for attempt in range(3):
            prompt = build_reward_prompt(
                task=task,
                reward_variables=sorted(adapter.reward_variables),
                best_expression=best_expression,
                best_score=best_score,
                invalid_feedback=None if last_error is None else str(last_error),
            )
            text = self._chat_text(prompt)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = int(inputs.input_ids.shape[-1])
            completion_token_ids = output.sequences[0][prompt_len:].tolist()
            completion_text = self.tokenizer.decode(completion_token_ids, skip_special_tokens=True)
            try:
                expression = extract_reward_expression(completion_text)
                RewardExpression(expression, adapter.reward_variables)
            except ValueError as exc:
                last_error = exc
                continue

            old_logprobs = token_logprobs(output.scores, completion_token_ids)
            return RewardCandidate(
                name=f"gen{generation}_hf{index}",
                task=task,
                prompt_id=adapter.prompt_id,
                prompt=prompt,
                expression=expression,
                weights={},
                generation=generation,
                generator_type="hf",
                generator_checkpoint=self.generator_checkpoint,
                completion_token_ids=completion_token_ids,
                old_logprobs=old_logprobs,
            )

        raise ValueError(f"HF generator failed to produce a valid reward expression after 3 attempts: {last_error}")

    def _chat_text(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You generate concise Python reward expressions for reinforcement learning.",
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"User: {prompt}\nAssistant:"


def build_reward_prompt(
    *,
    task: str,
    reward_variables: list[str],
    best_expression: str | None,
    best_score: float | None,
    invalid_feedback: str | None = None,
) -> str:
    feedback = ""
    if best_expression is not None and best_score is not None:
        feedback = (
            "\nCurrent best reward expression:\n"
            f"{best_expression}\n"
            f"Current best true environment return: {best_score:.4f}\n"
            "Try to improve it while keeping the expression simple.\n"
        )
    retry = ""
    if invalid_feedback:
        retry = (
            "\nYour previous answer was rejected by the reward-expression validator:\n"
            f"{invalid_feedback}\n"
            "Return a simpler valid expression using only the allowed variables and helpers.\n"
        )

    return (
        f"Design one dense reward expression for {task}.\n"
        "Return only a single Python expression, not a function, markdown, comments, or explanation.\n"
        "The expression will be parsed with Python ast and may only use numeric operators, conditionals, "
        "comparisons, abs, min, max, sqrt, sin, cos, tanh, exp, and these variables:\n"
        f"{', '.join(reward_variables)}\n"
        "The expression trains PPO. It will be scored only by true environment return, not by its own value.\n"
        "Prefer forward progress, stable healthy locomotion, low lateral drift, and modest control cost.\n"
        f"{feedback}"
        f"{retry}"
    )


def extract_reward_expression(text: str) -> str:
    stripped = text.strip()
    code_fence = re.search(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if code_fence:
        stripped = code_fence.group(1).strip()

    for prefix in ("REWARD_EXPRESSION =", "reward =", "return "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        raise ValueError("HF generator returned an empty completion")

    expression = lines[0]
    if "=" in expression and not any(op in expression for op in ("==", "!=", "<=", ">=")):
        expression = expression.split("=", 1)[1].strip()
    return expression.strip().strip("`")


def token_logprobs(scores: tuple[Any, ...], token_ids: list[int]) -> list[float]:
    logprobs: list[float] = []
    for score, token_id in zip(scores, token_ids, strict=False):
        token_logprobs_tensor = torch.log_softmax(score[0].float(), dim=-1)
        logprobs.append(float(token_logprobs_tensor[int(token_id)].cpu()))
    return logprobs
