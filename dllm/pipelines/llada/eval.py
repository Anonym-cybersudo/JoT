"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/llada/eval.py \
    --tasks gsm8k_cot \
    --model llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512,cfg=0.0"
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import accelerate
import torch
import torch.nn.functional as F
from tqdm import tqdm
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype

import dllm
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig

eval_logger = logging.getLogger(__name__)


@dataclass
class LLaDAEvalConfig(MDLMSamplerConfig):
    # According to LLaDA's opencompass implementation: https://github.com/ML-GSAI/LLaDA/blob/main/opencompass/opencompass/models/dllm.py
    max_new_tokens: int = 1024
    max_length: int = 4096
    steps: int = 1024
    block_size: int = 1024

    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"
    efficiency_output_path: str | None = None  # Path to save efficiency metrics JSON
    # Adaptive early-exit parameters with spatial-temporal thresholding
    max_threshold: float = 3.0  # Maximum P1/P2 threshold
    min_threshold: float = 1.0  # Minimum P1/P2 threshold
    geo_decay: float = 0.5  # Geometric decay rate for spatial weights
    max_distance: int = 16  # Maximum distance for spatial context
    schedule_type: str = "exponential"  # "none", "exponential", "linear", "cosine"
    schedule_lambda: float = 8.0  # Lambda for exponential schedule


@register_model("llada")
class LLaDAEvalHarness(LM):
    @staticmethod
    def _parse_token_list(value):
        """Parse token list from string format like '[126081;126348]' or list."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]  # Remove brackets
            if not value:  # Empty string after removing brackets
                return []
            return [int(x.strip()) for x in value.split(";") if x.strip()]
        elif isinstance(value, list):
            return value
        elif value is None:
            return []
        return []

    def __init__(
        self,
        config: LLaDAEvalConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        if config is None:
            config = LLaDAEvalConfig()

        # Pull args from config, allow kwargs to override
        pretrained = kwargs.get("pretrained", config.pretrained)
        dtype = kwargs.get("dtype", config.dtype)
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        device = kwargs.get("device", config.device)
        cfg = kwargs.get("cfg", config.cfg_scale)
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = self._parse_token_list(
            kwargs.get("suppress_tokens", config.suppress_tokens)
        )
        begin_suppress_tokens = self._parse_token_list(
            kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        )
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        efficiency_output_path = kwargs.get(
            "efficiency_output_path", config.efficiency_output_path
        )
        # Adaptive early-exit parameters
        max_threshold = float(kwargs.get("max_threshold", config.max_threshold))
        min_threshold = float(kwargs.get("min_threshold", config.min_threshold))
        geo_decay = float(kwargs.get("geo_decay", config.geo_decay))
        max_distance = int(kwargs.get("max_distance", config.max_distance))
        schedule_type = kwargs.get("schedule_type", config.schedule_type)
        schedule_lambda = float(kwargs.get("schedule_lambda", config.schedule_lambda))
        accelerator = accelerate.Accelerator()

        # Get GLOBAL rank from torch.distributed (not accelerator)
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()  # ← GLOBAL rank (0-15)
            self._world_size = (
                torch.distributed.get_world_size()
            )  # ← GLOBAL world size (16)
        else:
            self._rank = 0
            self._world_size = 1

        # Use accelerator for device placement
        self.model = dllm.utils.get_model(
            SimpleNamespace(model_name_or_path=pretrained, dtype=get_dtype(dtype))
        )
        self.model.eval()

        if accelerator.num_processes > 1:
            # Let accelerator handle device placement
            self.model = accelerator.prepare(self.model)
            self.device = (
                accelerator.device
            )  # ← Accelerator figures out local device correctly
            self.accelerator = accelerator
        else:
            # Single GPU
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.tokenizer = dllm.utils.get_tokenizer(
            SimpleNamespace(model_name_or_path=pretrained, model=self.model)
        )

        # sampler params
        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.right_shift_logits = right_shift_logits

        # loglikelihood params
        self.mc_num = int(mc_num)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0

        # efficiency tracking
        self.efficiency_output_path = efficiency_output_path
        self.pretrained = pretrained
        
        # Adaptive early-exit parameters
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.geo_decay = geo_decay
        self.max_distance = max_distance
        self.schedule_type = schedule_type
        self.schedule_lambda = schedule_lambda

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, prompt_index.sum(), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> bool:
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """
        Move trailing spaces in the context to the beginning of the continuation
        and encode both pieces into token ids.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        out = []
        with torch.no_grad():
            for instance in tqdm(requests, desc="Computing likelihood..."):
                context, continuation = self._encode_pair(*instance.args)
                assert len(context) + len(continuation) <= self.max_length, (
                    f"Context + continuation length exceeds {self.max_length} tokens: "
                    f"{len(context)} + {len(continuation)}"
                )

                context = torch.tensor(context, device=self.device, dtype=torch.long)
                continuation = torch.tensor(
                    continuation, device=self.device, dtype=torch.long
                )

                logprob = self.get_loglikelihood(context, continuation)
                isgreedy = self.suffix_greedy_prediction(context, continuation)
                out.append((logprob, isgreedy))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate greedily until a stopping sequence

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, gen_kwargs).
            context: str
                Context string
            gen_kwargs: dict
                A dictionary of keyword arguments to pass to the generation function e.g. top_k, until, etc.
        :return: list[str]
            A list of model generated continuations.
            continuation: str
                The generated continuation.
        """
        out = []
        efficiency_samples = []  # Per-sample efficiency metrics
        sampler = MDLMSampler(model=self.model, tokenizer=self.tokenizer)

        for sample_idx, instance in enumerate(
            tqdm(requests, desc="Generating...", disable=(self.rank != 0))
        ):
            context, gen_kwargs = instance.args  # type: ignore
            prompt_ids = self.tokenizer(context)["input_ids"]
            prompt = [torch.tensor(prompt_ids, device=self.device, dtype=torch.long)]
            stop_tokens = gen_kwargs["until"]

            # Generation with timing
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            sample_out = sampler.sample(
                inputs=prompt,
                steps=self.steps,
                max_new_tokens=self.max_new_tokens,
                block_size=self.block_size,
                temperature=0.0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                suppress_tokens=self.suppress_tokens,
                begin_suppress_tokens=self.begin_suppress_tokens,
                right_shift_logits=self.right_shift_logits,
                max_threshold=self.max_threshold,
                min_threshold=self.min_threshold,
                geo_decay=self.geo_decay,
                max_distance=self.max_distance,
                schedule_type=self.schedule_type,
                schedule_lambda=self.schedule_lambda,
            )
            if isinstance(sample_out, tuple):
                generated_ids, actual_steps = sample_out
            else:
                generated_ids = sample_out
                actual_steps = self.steps

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed_time = time.perf_counter() - start_time

            configured_steps = self.steps

            generated_answer = self.tokenizer.decode(
                generated_ids[0][prompt[0].shape[0] :], skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )

            # Record efficiency metrics
            efficiency_samples.append(
                {
                    "sample_idx": sample_idx,
                    "input": context[:500] + "..." if len(context) > 500 else context,
                    "output": generated_answer,
                    "time_seconds": elapsed_time,
                    "configured_steps": configured_steps,
                    "actual_steps": actual_steps,
                    "step_ratio": configured_steps / actual_steps,
                }
            )

            out.append(generated_answer)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        # Save efficiency metrics if output path is specified
        if self.efficiency_output_path and self.rank == 0:
            self._save_efficiency_metrics(efficiency_samples)

        return out

    def _save_efficiency_metrics(self, samples: list[dict]) -> None:
        """Save per-sample and aggregated efficiency metrics to JSON."""
        if not samples:
            return

        total_time = sum(s["time_seconds"] for s in samples)
        avg_time = total_time / len(samples)
        avg_step_ratio = sum(s["step_ratio"] for s in samples) / len(samples)

        metrics = {
            "model": "llada",
            "pretrained": self.pretrained,
            "config": {
                "max_new_tokens": self.max_new_tokens,
                "steps": self.steps,
                "block_size": self.block_size,
                "cfg": self.cfg,
                "remasking": self.remasking,
                "max_threshold": self.max_threshold,
                "min_threshold": self.min_threshold,
                "geo_decay": self.geo_decay,
                "max_distance": self.max_distance,
                "schedule_type": self.schedule_type,
                "schedule_lambda": self.schedule_lambda,
            },
            "aggregated": {
                "total_samples": len(samples),
                "total_time_seconds": total_time,
                "avg_time_per_sample_seconds": avg_time,
                "avg_step_ratio": avg_step_ratio,  # configured_steps / actual_steps (speedup)
            },
            "samples": samples,
        }

        output_path = Path(self.efficiency_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        eval_logger.info(f"Saved efficiency metrics to {output_path}")


if __name__ == "__main__":
    cli_evaluate()
