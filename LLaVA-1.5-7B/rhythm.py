import os
import json
import re
import random
import torch
from PIL import Image
from datasets import Dataset
from transformers import LlavaForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

from utils import *


model_path = "..."
output_dir = "..."
data_dir = "datasets/..."

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    attn_implementation="eager"
)


data = load_train(data_dir)


def compute_gamma(attentions, response_positions, strategy="coupled", q=0.4, gamma_amp=2, lookback_k=3):
    d_lh = compute_head_avg_backward_distance(attentions, response_positions)
    local_heads, global_heads = group_heads_by_span(d_lh, local_ratio=0.3, global_ratio=0.3)

    A_loc = torch.stack([attentions[l][:, h] for l, h in local_heads]).mean(dim=0)
    A_glob = torch.stack([attentions[l][:, h] for l, h in global_heads]).mean(dim=0)

    waad = compute_waad(A_loc, response_positions)
    fai = compute_fai(A_glob, response_positions)

    gamma = torch.ones(len(response_positions), device=waad.device)

    delta = torch.abs(torch.diff(waad, append=waad[-1:]))

    thresh_delta = torch.quantile(delta.float(), 1 - q)
    T_loc = delta >= thresh_delta

    thresh_fai = torch.quantile(fai.float(), 1 - q)
    T_glob = fai >= thresh_fai

    if strategy == "local":
        gamma[T_loc] = gamma_amp

    elif strategy == "global":
        gamma[T_glob] = gamma_amp

    elif strategy == "coupled":
        gamma[T_glob] = gamma_amp
        thresh_waad_low = torch.quantile(waad.float(), q)
        anchor_indices = torch.where(T_glob)[0]
        alpha = 0.5
        bonus = gamma_amp - 1.0

        for t in anchor_indices:
            if waad[t] > thresh_waad_low:
                continue

            start_search = max(0, t - lookback_k)
            window_preplans = torch.where(T_loc[start_search:t])[0]

            if len(window_preplans) > 0:
                local_idx = window_preplans[-1]
                intro_t = start_search + local_idx
                gamma[t] = 1.0 + (1.0 - alpha) * bonus
                gamma[intro_t] += alpha * bonus

    return gamma.detach()


class RhythmAwareGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, strategy="coupled", loss_type="grpo", **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.loss_type = loss_type

    def _generate_and_score_completions(self, inputs):
        outputs = super()._generate_and_score_completions(inputs)

        device = self.accelerator.device
        batch_size = outputs["completion_ids"].shape[0]

        pad_token_id = 32001

        gamma_t_list = []

        for i in range(batch_size):
            prompt_ids = outputs["prompt_ids"][i:i + 1]
            completion_ids = outputs["completion_ids"][i:i + 1]
            full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

            attention_mask = (full_ids != pad_token_id).long()

            extra_kwargs = {}
            if "pixel_values" in outputs:
                extra_kwargs["pixel_values"] = outputs["pixel_values"][i:i + 1]
            if "image_sizes" in outputs:
                extra_kwargs["image_sizes"] = outputs["image_sizes"][i:i + 1]
            if "image_grid_thw" in outputs:
                extra_kwargs["image_grid_thw"] = outputs["image_grid_thw"][i:i + 1]
            if "pixel_attention_mask" in outputs:
                extra_kwargs["pixel_attention_mask"] = outputs["pixel_attention_mask"][i:i + 1]

            with torch.no_grad():
                attn_outputs = self.model(
                    input_ids=full_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    output_attentions=True,
                    return_dict=True,
                    **extra_kwargs
                )

            prompt_physical_len = prompt_ids.shape[1]
            valid_completion_len = (completion_ids != pad_token_id).sum().item()
            response_positions = list(range(prompt_physical_len, prompt_physical_len + valid_completion_len))

            gamma_t = compute_gamma(
                attentions=attn_outputs.attentions,
                response_positions=response_positions,
                strategy=self.strategy
            )

            current_physical_len = completion_ids.shape[1]
            if gamma_t.shape[0] < current_physical_len:
                pad_len = current_physical_len - gamma_t.shape[0]
                gamma_t = torch.cat([
                    gamma_t,
                    torch.ones(pad_len, device=gamma_t.device)
                ])
            gamma_t = gamma_t[:current_physical_len]
            gamma_t_list.append(gamma_t)

        gamma_t_batch = torch.stack(gamma_t_list)
        outputs["gamma_t"] = gamma_t_batch

        return outputs

    def _compute_loss(self, model, inputs):
        advantages = inputs["advantages"]
        gamma_t = inputs["gamma_t"]

        gamma_t = torch.where(advantages.unsqueeze(1) > 0, gamma_t, torch.ones_like(gamma_t))
      
        token_advantages = advantages.unsqueeze(1) * gamma_t

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        log_ratio = per_token_logps - old_per_token_logps

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * token_advantages
        per_token_loss2 = coef_2 * token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (token_advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (token_advantages > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())

        return loss


training_args = GRPOConfig(
    output_dir=output_dir,
    num_generations=4,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    max_prompt_length=None,
    max_completion_length=512,
    temperature=1.0,
    logging_steps=50,
    bf16=True,
    tf32=True
)

trainer = RhythmAwareGRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    strategy="coupled",
    reward_funcs=[sqa_reward_func, reason_reward_func]
)

print("Starting training...")
trainer.train()
trainer.save_model(output_dir)
print("Training completed.")
