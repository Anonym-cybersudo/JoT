"""
reference: https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, SamplerConfig, SamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


def compute_spatial_weights(
    mask_index: torch.Tensor,
    geo_decay: float = 0.5,
    max_distance: int = 16,
) -> torch.Tensor:
    """
    Compute spatial softening weights based on nearby unmasked tokens.
    
    Uses geometric distribution: weight at distance d = geo_decay^d
    Sums weights from both left and right unmasked neighbors.
    
    Args:
        mask_index: Boolean tensor [B, T] where True = masked position
        geo_decay: Decay rate for geometric distribution (e.g., 0.5)
        max_distance: Maximum distance to consider for neighbors
    
    Returns:
        weights: Tensor [B, T] with spatial softening weights (higher = more context)
    """
    B, T = mask_index.shape
    device = mask_index.device
    
    unmasked = ~mask_index  # [B, T]
    
    weights = torch.zeros(B, T, device=device, dtype=torch.float32)
    
    kernel_size = 2 * max_distance + 1
    kernel = torch.zeros(kernel_size, device=device)
    for d in range(1, max_distance + 1):
        kernel[max_distance - d] = geo_decay ** d  # left neighbors
        kernel[max_distance + d] = geo_decay ** d  # right neighbors
    kernel = kernel.view(1, 1, -1)  # [1, 1, kernel_size]
    
    unmasked_float = unmasked.float().unsqueeze(1)  # [B, 1, T]
    weights = F.conv1d(unmasked_float, kernel, padding=max_distance).squeeze(1)  # [B, T]
    
    return weights


@dataclass
class MDLMSamplerConfig(SamplerConfig):
    max_new_tokens: int = 128
    max_length: int = (
        None
    )
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False

    max_threshold: float = 3.0
    min_threshold: float = 1.0
    geo_decay: float = 0.5
    max_distance: int = 16



@dataclass
class MDLMSampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: MDLMSamplerConfig | None = None,
        **kwargs,
    ) -> SamplerOutput | torch.Tensor:
        """
        Generate text using masked diffusion language modeling.

        Iteratively unmasks tokens over multiple diffusion steps, starting from
        fully masked sequences appended to the input prompts.

        Args:
            inputs: List of input prompts (token tensors or lists of token IDs).
            config: Sampler configuration, or None to use defaults.
            **kwargs: Override specific config parameters.

        Returns:
            SamplerOutput with generated sequences, or (tensor, actual_steps) if
            return_dict=False.
        """
        if config is None:
            config = MDLMSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        assert 1 <= block_size
        assert 1 <= steps
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id


        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        prompt_lens = [p.shape[0] for p in inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        else:
            max_new_tokens = max_length - max(prompt_lens)

        B = len(inputs)
        T = max_length

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + max_new_tokens] = (
                mask_id
            )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, pl in enumerate(prompt_lens):
            valid_end = min(pl + max_new_tokens, T)
            attention_mask[i, :valid_end] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        actual_steps = 0
        finished = False

        for b in range(num_blocks):
            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=x.device
            )

            for j in range(B):
                start = prompt_lens[j] + b * block_size
                end = min(start + block_size, prompt_lens[j] + max_new_tokens, T)
                if start < end:
                    width = end - start
                    block_mask_index[j, :width] = (
                        x[j, start:end] == mask_id
                    )

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for i in range(effective_steps):
                mask_index = x == mask_id

                gen_mask_count = 0
                for j in range(B):
                    gen_start = prompt_lens[j]
                    gen_end = min(prompt_lens[j] + max_new_tokens, T)
                    gen_mask_count += mask_index[j, gen_start:gen_end].sum().item()
                if gen_mask_count == 0:
                    finished = True
                    break
                actual_steps += 1

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(
                    logits_with_noise, dim=-1
                )

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                p = F.softmax(logits, dim=-1)
                
                if remasking == "low_confidence":
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    x0_p[j, prompt_lens[j] + (b + 1) * block_size :] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(
                    mask_index, x0_p, -np.inf
                )

                max_thresh = kwargs.get("max_threshold", config.max_threshold)
                min_thresh = kwargs.get("min_threshold", config.min_threshold)
                geo_decay = kwargs.get("geo_decay", config.geo_decay)
                max_dist = kwargs.get("max_distance", config.max_distance)
                
                combined_mask = None
                if max_thresh > min_thresh:
                    eps = 1e-12
                    
                    global_step = b * steps + i
                    total_steps = num_blocks * steps
                    t_normalized = global_step / max(total_steps - 1, 1)
                    
                    spatial_weights = compute_spatial_weights(
                        mask_index, geo_decay, max_dist
                    )
                    
                    max_spatial = 2 * sum(geo_decay ** d for d in range(1, max_dist + 1))
                    spatial_softening = torch.clamp(spatial_weights / max_spatial, 0.0, 1.0)
                    
                    
                    threshold = max_thresh - (max_thresh - min_thresh) * spatial_softening
                    
                    top2_probs, _ = torch.topk(p, k=2, dim=-1)  # [B, T, 2]
                    top1_prob = top2_probs[:, :, 0]
                    top2_prob = top2_probs[:, :, 1]
                    ratio = top1_prob / (top2_prob + eps)
                    
                    combined_mask = mask_index & (ratio >= threshold)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                if combined_mask is not None:
                    transfer_index = transfer_index | combined_mask

                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())
            if finished:
                break

        if not return_dict:
            return x, actual_steps
        else:
            return SamplerOutput(
                sequences=x, histories=histories, actual_steps=actual_steps
            )

    @torch.no_grad()
    def infill(
        self, inputs: list[torch.Tensor | list], config, **kwargs
    ) -> SamplerOutput | torch.Tensor:
        """
        Fill in-place the <|mdm_mask|> tokens contained in `inputs`.
        The whole (padded) sequence is split into block windows of length
        `block_size`; within each window we progressively "unmask" positions
        according to the scheduler and chosen remasking strategy.

        Notes:
        - Right padding uses EOS.
        - CFG masks out *originally known* (non-mask, non-EOS) tokens in the
        unconditional branch, identical to `generate`.
        - Only masked positions are ever updated; non-mask tokens are left intact.
        """
        steps = kwargs.get("steps", config.steps)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(p, list) and len(p) == 0 else p for p in inputs
            ]

        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)

        if block_size is None:
            block_size = T

        assert 1 <= block_size
        assert 1 <= steps

        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t

        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if not (cfg_keep_tokens is None or len(cfg_keep_tokens) == 0):
            keep_mask = torch.isin(
                x, torch.as_tensor(cfg_keep_tokens, device=self.model.device)
            )
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)

            block_mask_index = torch.zeros(
                (B, block_size), dtype=torch.bool, device=self.model.device
            )
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask_index[j, :width] = x[j, start : start + width] == mask_id

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )

            effective_steps = num_transfer_tokens.size(1)

            for s in range(effective_steps):
                mask_index_full = x == mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self.model(
                        x_, attention_mask=attention_mask
                    ).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(
                        x, attention_mask=attention_mask
                    ).logits

                if suppress_tokens is not None and len(suppress_tokens) > 0:
                    for token_id in suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, T]

                if begin_suppress_tokens is not None and len(begin_suppress_tokens) > 0:
                    for token_id in begin_suppress_tokens:
                        logits[:, :, token_id] = -torch.inf

                p = F.softmax(logits, dim=-1)
                
                if remasking == "low_confidence":
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
                        -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((B, T), device=self.model.device)
                else:
                    raise NotImplementedError(remasking)

                for j in range(B):
                    end_j = start + widths[j]
                    x0_p[j, :start] = -np.inf
                    x0_p[j, end_j:] = -np.inf

                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -np.inf)

                max_thresh = kwargs.get("max_threshold", config.max_threshold)
                min_thresh = kwargs.get("min_threshold", config.min_threshold)
                geo_decay = kwargs.get("geo_decay", config.geo_decay)
                max_dist = kwargs.get("max_distance", config.max_distance)
                schedule_type = kwargs.get("schedule_type", config.schedule_type)
                schedule_lambda = kwargs.get("schedule_lambda", config.schedule_lambda)
                
                combined_mask = None
                if max_thresh > min_thresh:
                    eps = 1e-12
                    
                    global_step = b * steps_per_block + s
                    total_steps = num_blocks * steps_per_block
                    t_normalized = global_step / max(total_steps - 1, 1)
                    
                    spatial_weights = compute_spatial_weights(
                        mask_index_full, geo_decay, max_dist
                    )
                    max_spatial = 2 * sum(geo_decay ** d for d in range(1, max_dist + 1))
                    spatial_softening = torch.clamp(spatial_weights / max_spatial, 0.0, 1.0)
                    

                    threshold = max_thresh - (max_thresh - min_thresh) * spatial_softening
                    
                    top2_probs, _ = torch.topk(p, k=2, dim=-1)
                    top1_prob = top2_probs[:, :, 0]
                    top2_prob = top2_probs[:, :, 1]
                    ratio = top1_prob / (top2_prob + eps)
                    
                    combined_mask = mask_index_full & (ratio >= threshold)

                transfer_index = torch.zeros_like(x, dtype=torch.bool)
                for j in range(B):
                    k = int(num_transfer_tokens[j, s].item())
                    if k > 0:
                        _, select_idx = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_idx] = True

                if combined_mask is not None:
                    transfer_index = transfer_index | combined_mask

                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        if not return_dict:
            return x
        else:
            return SamplerOutput(sequences=x, histories=histories)
