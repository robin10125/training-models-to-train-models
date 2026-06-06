from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from typing import Any

import torch
from gymnasium.envs.mujoco.ant_v5 import AntEnv

from .adapters import ANT_TASK, get_adapter
from .rewards import RewardCandidate, RewardExpression, total_expression_from_components, validate_component_expressions


DEFAULT_HF_MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
ANT_TASK_DESCRIPTION = "to make the ant run forward as fast as possible"
ANT_GENERATIVE_REWARD_VARIABLES = frozenset(
    {
        "x_velocity",
        "y_velocity",
        "torso_z",
        "action_l2",
        "healthy",
        "terminated",
        "truncated",
    }
)


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
        self._rejected_candidates: list[tuple[RewardCandidate, str]] = []

    def generate_population(
        self,
        *,
        task: str,
        population: int,
        generation: int,
        best_expression: str | None = None,
        best_score: float | None = None,
        elites: list[dict[str, Any]] | None = None,
        evolution_feedback: str | None = None,
    ) -> list[RewardCandidate]:
        candidates = []
        for index in range(population):
            try:
                candidates.append(
                    self.generate_candidate(
                        task=task,
                        index=index,
                        generation=generation,
                        best_expression=best_expression,
                        best_score=best_score,
                        elites=elites,
                        evolution_feedback=evolution_feedback,
                    )
                )
            except ValueError:
                continue
        return candidates

    def drain_rejected_candidates(self) -> list[tuple[RewardCandidate, str]]:
        rejected = self._rejected_candidates
        self._rejected_candidates = []
        return rejected

    def generate_candidate(
        self,
        *,
        task: str,
        index: int,
        generation: int,
        best_expression: str | None,
        best_score: float | None,
        elites: list[dict[str, Any]] | None = None,
        evolution_feedback: str | None = None,
    ) -> RewardCandidate:
        adapter = get_adapter(task)
        reward_variables = generation_reward_variables(task)
        prompt = ""
        last_error: Exception | None = None
        eureka_feedback = evolution_feedback or format_eureka_feedback(elites)
        for attempt in range(3):
            prompt = build_reward_prompt(
                task=task,
                reward_variables=sorted(reward_variables),
                best_expression=best_expression,
                best_score=best_score,
                elites=elites,
                evolution_feedback=eureka_feedback,
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
            old_logprobs = token_logprobs(output.scores, completion_token_ids)
            try:
                component_expressions = extract_reward_components(completion_text)
                expression = total_expression_from_components(component_expressions, "")
                validate_component_expressions(component_expressions, reward_variables)
                RewardExpression(expression, reward_variables)
            except ValueError as exc:
                last_error = exc
                self._rejected_candidates.append(
                    (
                        RewardCandidate(
                            name=f"gen{generation}_hf{index}_invalid{attempt}",
                            task=task,
                            prompt_id=adapter.prompt_id,
                            prompt=text,
                            expression=completion_text.strip(),
                            completion_text=completion_text.strip(),
                            weights={},
                            generation=generation,
                            generator_type="hf",
                            generator_checkpoint=self.generator_checkpoint,
                            completion_token_ids=completion_token_ids,
                            old_logprobs=old_logprobs,
                            eureka_role="invalid_initial" if not elites else "invalid_elite_refinement",
                            eureka_parent_names=[str(item.get("name")) for item in elites] if elites else None,
                            eureka_parent_expressions=[str(item.get("expression")) for item in elites] if elites else None,
                            eureka_parent_scores=[item.get("score") for item in elites] if elites else None,
                            eureka_elite_names=[str(item.get("name")) for item in elites] if elites else None,
                            eureka_elite_expressions=[str(item.get("expression")) for item in elites] if elites else None,
                            eureka_elite_scores=[item.get("score") for item in elites] if elites else None,
                            eureka_feedback=eureka_feedback,
                        ),
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            return RewardCandidate(
                name=f"gen{generation}_hf{index}",
                task=task,
                prompt_id=adapter.prompt_id,
                prompt=text,
                expression=expression,
                component_expressions=component_expressions,
                completion_text=completion_text.strip(),
                weights={},
                generation=generation,
                generator_type="hf",
                generator_checkpoint=self.generator_checkpoint,
                completion_token_ids=completion_token_ids,
                old_logprobs=old_logprobs,
                eureka_role="initial" if not elites else "elite_refinement",
                eureka_parent_names=[str(item.get("name")) for item in elites] if elites else None,
                eureka_parent_expressions=[str(item.get("expression")) for item in elites] if elites else None,
                eureka_parent_scores=[item.get("score") for item in elites] if elites else None,
                eureka_elite_names=[str(item.get("name")) for item in elites] if elites else None,
                eureka_elite_expressions=[str(item.get("expression")) for item in elites] if elites else None,
                eureka_elite_scores=[item.get("score") for item in elites] if elites else None,
                eureka_feedback=eureka_feedback,
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
    elites: list[dict[str, Any]] | None = None,
    evolution_feedback: str | None = None,
    invalid_feedback: str | None = None,
) -> str:
    feedback = ""
    if evolution_feedback:
        feedback = (
            "\nEUREKA evolutionary feedback from the previous generation:\n"
            f"{evolution_feedback}\n"
            "Use this feedback as evidence. Preserve reward terms that produced better verified return, "
            "remove terms that appear to reward the wrong behavior, and produce one new expression.\n"
        )
    elif elites:
        feedback = (
            "\nEUREKA elite archive from the previous generation, ranked by verified target-environment return:\n"
            f"{format_eureka_feedback(elites)}\n"
            "Use these as evidence. Preserve useful locomotion incentives, remove likely distractors, "
            "and produce one new reward expression that could outperform the ranked elites.\n"
        )
    elif best_expression is not None and best_score is not None:
        feedback = (
            "\nCurrent best reward expression:\n"
            f"{best_expression}\n"
            f"Current best verified target-environment return: {best_score:.4f}\n"
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
        "Return only a Python dictionary literal mapping reward component names to Python expression strings. "
        "Do not return a function, markdown, comments, or explanation. Example:\n"
        "{\"progress\": \"x_velocity\", \"stability\": \"1.0 if healthy else -1.0\", \"effort\": \"-0.01 * action_l2\"}\n"
        f"{task_prompt_context(task)}\n"
        "The expression will be parsed with Python ast and may only use numeric operators, conditionals, "
        "comparisons, abs, min, max, sqrt, sin, cos, tanh, exp, and these variables:\n"
        f"{', '.join(reward_variables)}\n"
        "The component expressions will be summed to train PPO. They will be scored only by verified target-environment return, not by their own value.\n"
        f"{feedback}"
        f"{retry}"
    )


def task_prompt_context(task: str) -> str:
    if task != ANT_TASK:
        raise ValueError(f"Unsupported task for prompt context: {task}")
    return (
        "Task context:\n"
        f"- Description: {ANT_TASK_DESCRIPTION}.\n"
        "- The agent is a MuJoCo Ant quadruped controlled by continuous joint torques.\n"
        "- Your generated expression trains the policy, but model reward is assigned from final verified target-environment performance.\n"
        f"{source_context()}"
    )


def source_context() -> str:
    gym_source = "\n".join(
        inspect.getsource(getattr(AntEnv, method_name))
        for method_name in ("step", "_get_obs", "reset_model")
    )
    return (
        "\nEnvironment source code excerpt:\n"
        "```python\n"
        f"{gym_source}\n"
        "```\n"
    )


def generation_reward_variables(task: str) -> frozenset[str]:
    if task != ANT_TASK:
        raise ValueError(f"Unsupported task for reward generation: {task}")
    return ANT_GENERATIVE_REWARD_VARIABLES


def format_eureka_feedback(elites: list[dict[str, Any]] | None) -> str | None:
    if not elites:
        return None
    rows = []
    for rank, item in enumerate(elites, start=1):
        score = item.get("score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        rows.append(
            f"{rank}. name={item.get('name')} true_return={score_text} expression={item.get('expression')}"
        )
    return "\n".join(rows)


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


def extract_reward_components(text: str) -> dict[str, str]:
    stripped = text.strip()
    code_fence = re.search(r"```(?:python|json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if code_fence:
        stripped = code_fence.group(1).strip()
    for prefix in ("REWARD_COMPONENTS =", "reward_components =", "components =", "return "):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        expression = extract_reward_expression(text)
        return {"total": expression}
    if isinstance(parsed, str):
        return {"total": parsed}
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("HF generator must return a non-empty dict of reward component expressions")
    components = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"Invalid reward component name: {key!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid reward component expression for {key!r}")
        components[key] = value.strip()
    return components


def token_logprobs(scores: tuple[Any, ...], token_ids: list[int]) -> list[float]:
    logprobs: list[float] = []
    for score, token_id in zip(scores, token_ids, strict=False):
        token_logprobs_tensor = torch.log_softmax(score[0].float(), dim=-1)
        logprobs.append(float(token_logprobs_tensor[int(token_id)].cpu()))
    return logprobs
