#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train N LoRA adapters that share one value head for reward modelling,
with per-0.01-epoch validation of each adapter’s accuracy,
using the Trainer’s built-in evaluation.
"""

import os, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedModel,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from torch.utils.data import DataLoader
from grm_utils import AutoModelForCausalLMWithValueHead,AutoModelForCausalLMWithMultiValueHead
# PEFT / TRL
from peft import LoraConfig, TaskType, get_peft_model
from trl.trainer.utils import compute_accuracy

# project helpers
from reward_trainer import RewardTrainer, RewardDataCollatorWithPadding
from load_datasets import load_train_eval_dataset
from load_eval_datasets import load_eval_dataset
from utils import print_trainable_parameters, compute_metrics, freeze_trainable_parameters


# -----------------------------------------------------------------------------
# 1 · CLI / script arguments
# -----------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    per_device_train_batch_size: int   = field(default=8) #8
    gradient_accumulation_steps: int   = field(default=16)
    learning_rate:            float    = field(default=1e-5)
    num_train_epochs:         int      = field(default=2)
    optim:                    str      = field(default="adamw_torch")
    lr_scheduler_type:        str      = field(default="cosine")
    max_length:               int      = field(default=1024)
    gradient_checkpointing:   bool     = field(default=True)
    bf16:                     bool     = field(default=True)
    attn_implementation:      str      = field(default="flash_attention_2")

    dataset:      str  = field(default="llm-blender/Unified-Feedback")
    dataset_mode: str  = field(default="40k")
    debug:        bool = field(default=False)
    output_tag:   str  = field(default="")
    random_seed:  int  = field(default=32)
    
    use_lora:            bool        = field(default=True)
    lora_target_modules: List[str]   = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_r:              int         = field(default=32)
    lora_alpha:          int         = field(default=64)
    lora_dropout:        float       = field(default=0.05)
    num_adapters:        int         = field(default=2)
    diversity_lambda:    float       = field(default=0.1)
    diversity_type:      str         = field(default="dncc", metadata={"help": "one of: 'none', 'ncl', 'hsic', 'dpp'"})

    per_device_eval_batch_size: int  = field(default=16) #16
    evaluation_strategy:        str  = field(default="no")
    report_to:                  str  = field(default="none")
    log_dir:                    str  = field(default="./reward_models_train")
    resume_from_dir:            str  = field(default=None)
    wandb_name:                 str  = field(default="multi-lora")
    save_strategy:              str  = field(default="epoch")
    save_steps:                 int  = field(default=1000)

    base_model:        str   = field(default="google/gemma-2b-it")
    loss_type:         str   = field(default="bt")
    weight_ratio:      float = field(default=0.1)
    freeze_pretrained: bool  = field(default=True)
    layer_type: Optional[str] = field(default='mlp') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)
def _set_active_adapter(wrapped, name: str):
    if hasattr(wrapped, "module"):
        wrapped.module.set_adapter(name)
    else:
        wrapped.set_adapter(name)
        
def idx_to_name(model, idxs):
    names = list(dict(model.named_parameters()).keys())
    return {i: names[i] for i in idxs}

def load_adapters_and_head(base_model, ckpt_dir: Path, device):
    # load *one* adapter (the default) first
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(
        base_model,
        ckpt_dir / "adapter_0",
        is_trainable=True,
        device_map={"": device},  
        torch_dtype=torch.bfloat16            #   keep dtype consistent
    )
    for sub in ckpt_dir.glob("adapter_*"):
        if sub.name not in peft_model.peft_config:
            peft_model.load_adapter(
                sub,
                adapter_name=sub.name,
                is_trainable=True,           
                device_map={"": device},   
                torch_dtype=torch.bfloat16
            )

    # score_path = ckpt_dir / "score_head.bin"
    # if score_path.exists():
    #     peft_model.score.load_state_dict(
    #         torch.load(score_path, map_location=device), strict=True)
    
    return peft_model.to(device)

def save_all_components(accelerator, trainer, tokenizer, save_dir: str):
    model_to_save = accelerator.unwrap_model(trainer.model)
    tokenizer.save_pretrained(save_dir)

    # 1️⃣  save each value head alongside its adapter
    for i in range(trainer.num_adapters):
        subdir = os.path.join(save_dir, f"adapter_{i}")
        os.makedirs(subdir, exist_ok=True)
        # LoRA weights
        model_to_save.save_pretrained(
            subdir, safe_serialization=True,
            selected_adapters=[f"adapter_{i}"]
        )
        # Value-head weights
        torch.save(model_to_save.v_heads[i].state_dict(),
                   os.path.join(subdir, "v_head.bin"))

    # 2️⃣  still useful to save the best average checkpoint
    torch.save(
        {k: v.cpu() for k, v in model_to_save.state_dict().items()
         if not k.startswith("v_heads.") and "adapter_" not in k},
        os.path.join(save_dir, "backbone.bin"),
    )

def adapter_vec_with_vhead(model, i: int, out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    device = next(model.parameters()).device
    vecs   = []

    # ───────────────────────────────────────────────────────────────
    # 1) LoRA ΔW = B @ A  (same as before)
    # ───────────────────────────────────────────────────────────────
    sufA = f".lora_A.adapter_{i}.weight"
    params = dict(model.named_parameters())          # one pass lookup
    for name, A in params.items():
        if name.endswith(sufA):
            B = params[name.replace(".lora_A.", ".lora_B.")]
            vecs.append((B @ A).reshape(-1).to(out_dtype))

    # ───────────────────────────────────────────────────────────────
    # 2) Value-head parameters for head i
    #    names look like "…v_heads.{i}.summary.*"
    # ───────────────────────────────────────────────────────────────
    prefix = f"v_heads.{i}."
    for name, p in params.items():
        if prefix in name:
            vecs.append(p.reshape(-1).to(out_dtype))

    # ───────────────────────────────────────────────────────────────
    if not vecs:                                    # safety guard
        return torch.zeros(1, device=device, dtype=out_dtype)

    return torch.cat(vecs).to(device)

def dncc_bregman_penalty(rew_pos: torch.Tensor,
                         rew_neg: torch.Tensor,
                         temperature: float = 1.0,
                         eps: float = 1e-8) -> torch.Tensor:
    """
    DNCC diversity for scalar rewards (B × N).  Approximates the
    non-negative Bregman information with an average KL-divergence
    between per-adapter posteriors and the ensemble posterior.
    """
    # 1) Turn raw scores → probabilities (like a tiny soft-max)
    p_pos = torch.sigmoid(rew_pos / temperature)          # [B, N]
    p_neg = torch.sigmoid(rew_neg / temperature)          # [B, N]

    # 2) Ensemble posteriors
    mean_pos = p_pos.mean(dim=1, keepdim=True)            # [B, 1]
    mean_neg = p_neg.mean(dim=1, keepdim=True)            # [B, 1]

    # 3) Symmetric KL between each learner and the mean
    kl_pos = (p_pos * (p_pos.add(eps).log() -
                       mean_pos.add(eps).log())).mean()
    kl_neg = (p_neg * (p_neg.add(eps).log() -
                       mean_neg.add(eps).log())).mean()

    # 4) DNCC wants to **maximise** disagreement ⇒ add the term
    return kl_pos + kl_neg        # already ≥ 0 by construction

def hybrid_ncl_penalty(rew: torch.Tensor,
                       alpha_tau: float = 2.0) -> torch.Tensor:
    """
    Hybrid NCL:  add a soft-selection weight α_i = softmax(|Δ_i|·τ)
    so that learners with distinctive outputs contribute more.
    """
    B, N = rew.shape
    centred = rew - rew.mean(dim=1, keepdim=True)     # [B, N]
    
    # 1) Magnitude of each learner’s deviation
    dev = centred.abs().mean(dim=0)                   # [N]
    alpha = torch.softmax(dev * alpha_tau, dim=0)     # Σ α = 1
    
    # 2) Weighted negative covariance  ( ≤ 0 if learners correlate )
    cov = (alpha * centred).sum(dim=1)                # [B]
    return -cov.pow(2).mean()                         # maximise |cov|

def _rbf_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Fast scalar RBF on the last dim (vectorised)."""
    x2 = (x * x).sum(1, keepdim=True)
    dist = x2 + x2.t() - 2 * x @ x.t()
    return torch.exp(-dist / (2 * sigma ** 2))

def hsic_penalty(rew: torch.Tensor,
                     sigma: float = 1.0) -> torch.Tensor:

    B, N = rew.shape
    # Draw two different adapters without replacement each forward pass
    j, k = torch.randperm(N)[:2]
    x, y = rew[:, j:j+1], rew[:, k:k+1]               # [B, 1]

    K = _rbf_kernel(x, sigma)
    L = _rbf_kernel(y, sigma)

    # Unbiased HSIC – Janzing & Gretton formulation
    Bsize = float(B)
    H = torch.eye(B, device=rew.device) - 1.0 / Bsize
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = (Kc * Lc).sum() / (Bsize - 1) ** 2

    return hsic        # ≥ 0; add with +λ to maximise diversity

# -----------------------------------------------------------------------------
# 2 · Custom Callback for per-0.01-epoch eval using Trainer.evaluate()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class MultiAdapterEvalCallback(TrainerCallback):
    def __init__(self,
                 trainer,
                 eval_steps: int,
                 num_adapters: int,
                 eval_dataset: Dataset,
                 eval_loader: DataLoader,
                 accelerator: Accelerator):
        super().__init__()
        self.trainer       = trainer
        self.eval_steps    = eval_steps
        self.num_adapters  = num_adapters
        self.eval_dataset  = eval_dataset
        self.best_accuracy = 0.0 # Consider initializing this per-adapter if needed
        self.eval_loader   = eval_loader
        self.accelerator   = accelerator

    def on_step_end(self,
                    args,
                    state: TrainerState,
                    control: TrainerControl,
                    **kwargs):

        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n===== Validation @ step {state.global_step} =====")

            lora_accuracy = []
            for i in range(self.num_adapters):
                adapter_name = f"adapter_{i}"
                correct_loc = 0
                total_loc   = 0
                if self.trainer.use_lora:
                    _set_active_adapter(self.trainer.model, adapter_name)

                full_correct_flags = []          # <‑‑ tiny payload

                pbar = tqdm(
                    total=len(self.eval_dataset)//(
                        self.trainer.args.per_device_eval_batch_size *
                        self.accelerator.num_processes),
                    desc=f"Evaluating {adapter_name}",
                    disable=not self.accelerator.is_local_main_process
                )

                self.trainer.model.eval()
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    for batch in self.eval_loader:
                        # move to device
                        # print("eval batch", batch)
                        c = {k: v.to(self.accelerator.device)
                             for k, v in batch.items()
                             if k in ["input_ids", "attention_mask_chosen"]}
                        r = {k: v.to(self.accelerator.device)
                             for k, v in batch.items()
                             if k in ["input_ids_rejected", "attention_mask_rejected"]}

                        rc = self.trainer.model(c["input_ids"], attention_mask=c.get("attention_mask_chosen"), active_head=i)[-1].squeeze()
                        rr = self.trainer.model(r["input_ids_rejected"], attention_mask=r.get("attention_mask_rejected"), active_head=i)[-1].squeeze()

                        comp = rc > rr
                        correct_loc += int(comp.sum().item())
                        total_loc   += comp.numel()

                        # Free GPU
                        del rc, rr, comp
                        torch.cuda.empty_cache()

                        pbar.update(1)
                    pbar.close()

                # ‑‑‑ gather only booleans across processes ‑‑‑
                self.accelerator.wait_for_everyone()
                # all_correct = self.accelerator.gather_for_metrics(
                #                   torch.stack(full_correct_flags))  # 0/1 tensor
                ct_tensor = torch.tensor([correct_loc, total_loc], 
                                          device=self.accelerator.device)
                gathered = self.accelerator.gather_for_metrics(ct_tensor)
                # metric on main process only
                if self.accelerator.is_main_process:
                    # accuracy = all_correct.float().mean().item()
                    ws = self.accelerator.num_processes
                    ct = gathered.view(ws, 2)
                    correct_sum = int(ct[:, 0].sum().item())
                    total_sum   = int(ct[:, 1].sum().item())
                    accuracy    = correct_sum / total_sum
                    print(f"  • {adapter_name:9s} accuracy = {accuracy:.4f}")
                    lora_accuracy.append(accuracy)

                torch.cuda.empty_cache()

            # --- checkpoint logic unchanged ---
            if (self.accelerator.is_main_process and
                    sum(lora_accuracy)/self.num_adapters > self.best_accuracy):
                self.best_accuracy = sum(lora_accuracy)/self.num_adapters
                save_dir = (Path(self.trainer.args.output_dir) /
                            f"best_step_{state.global_step}_acc={self.best_accuracy:.4f}")

                save_all_components(self.accelerator, self.trainer, self.trainer.tokenizer, save_dir)
                print(f"🚀 New best model saved at {save_dir} with accuracy "
                      f"{self.best_accuracy:.4f}")

            if self.trainer.use_lora:
                _set_active_adapter(self.trainer.model, "adapter_0")

            self.trainer.model.train()
            print("==========================================\n")
            torch.cuda.empty_cache()

        return control
# -----------------------------------------------------------------------------
# 3 · Multi-adapter RewardTrainer (unchanged)
# -----------------------------------------------------------------------------
class MultiAdapterRewardTrainer(RewardTrainer):
    def __init__(self,
                 diversity_lambda: float = 0.0,
                 num_adapters: int = 1,
                 max_length: int = 1024,
                 use_lora: bool = True,
                 **kwargs):
        self.diversity_lambda = diversity_lambda
        self.num_adapters     = num_adapters
        self.max_length       = max_length
        self.use_lora         = use_lora
        self.best_accuracy    = 0.0 
        self.diversity_type   = "utd"
        super().__init__(**kwargs)

    def _adapter_update_vector(self, model, adapter_idx: int):
        """
        Collects and concatenates (B @ A) for every LoRA module
        in adapter_{adapter_idx}. If none are found, returns a
        zero‐vector to avoid torch.cat(empty).
        """
        params = dict(model.named_parameters())
        vecs = []
        # suffix we expect at the end of each lora_A weight name
        suffix = f".lora_A.adapter_{adapter_idx}.weight"
        for name, A in params.items():
            # look only at the exact LoRA-A weights for this adapter
            if name.endswith(suffix):
                # construct the corresponding lora_B name
                B_name = name.replace(".lora_A.", ".lora_B.")
                B = params.get(B_name)
                if B is None:
                    # either warn or skip if B not found
                    continue
                # compute B @ A and flatten
                vecs.append((B @ A).view(-1))

        if not vecs:
            # no LoRA parameters found for this adapter:
            # return a singleton zero‐tensor on the correct device
            device = next(model.parameters()).device
            return torch.zeros(1, device=device)

        # concatenate into one long vector
        return torch.cat(vecs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inp_ids = inputs["input_ids"]
        attn    = inputs["attention_mask"]
        # print("inputs", inputs)
        chosen, rejected = [], []
        for i in range(self.num_adapters):
            if self.use_lora:
                _set_active_adapter(model, f"adapter_{i}")
            # --- AFTER you have added/initialised every adapter + value head ---
            for n, p in model.named_parameters():                       # <-- ①
                if "adapter_" in n or "v_heads." in n:                       # plural!
                    p.requires_grad_(True)                                   # trainable
                else:
                    p.requires_grad_(False)                                  # frozen
            # -----------------------------------------------------------------

            rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], active_head=i)[-1]

            bsz = rewards.size(0)
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1
            rewards_j = rewards[jidx]
            rewards_k = rewards[kidx]
            
            chosen.append(rewards_j)
            rejected.append(rewards_k)
            del rewards                       # drop activations of this adapter
            torch.cuda.empty_cache()
        for i in range(self.num_adapters):
            # turn on gradients for every LoRA module in this adapter
            for name, param in model.named_parameters():
                if f"adapter_{i}" in name:
                    param.requires_grad_(True)
                    # print(name, param.requires_grad)
                    
        pref_loss = sum(
            -nn.functional.logsigmoid(rc - rr).mean()
            for rc, rr in zip(chosen, rejected)
        ).mean() / self.num_adapters

        if self.diversity_lambda > 0:
            rewards = torch.stack([rc.squeeze(-1) for rc in chosen], dim=1)  # [B, N]
            rewards_rejected = torch.stack([rc.squeeze(-1) for rc in rejected], dim=1)  # [B, N]
            reward_margin = torch.stack([(rc-rj).squeeze(-1) for rc, rj in zip(chosen,rejected)], dim=1) 
            if self.diversity_type == "dncc":
                div_term = dncc_bregman_penalty(rewards, rewards_rejected)
            elif self.diversity_type == "hybrid":
                div_term = hybrid_ncl_penalty(rewards)
            elif self.diversity_type == "hsic":
                div_term = utd_hsic_penalty(reward_margin)
            else:            # 'none'
                div_term = 0.0
            div_loss =  self.diversity_lambda * div_term
            print("peft_loss", pref_loss, "diversity_loss", self.diversity_lambda*div_term)
        else:
            div_loss = 0.0
        
        loss = pref_loss + div_loss

        if return_outputs:
            return loss, {
                "rewards_chosen": chosen[0],
                "rewards_rejected": rejected[0]
            }
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return loss, None, None

        chosen = logits_dict["rewards_chosen"].detach()
        rejected = logits_dict["rewards_rejected"].detach()
        logits = torch.stack([chosen, rejected], dim=1)
        probs = torch.softmax(logits, dim=1)
        labels = torch.zeros(probs.size(0), dtype=torch.long, device=probs.device)

        return loss.detach(), probs.cpu().numpy(), labels.cpu().numpy()

def init_lora_adapter(peft_model: nn.Module,
                      adapter_idx: int,
                      base_seed: int = 1234) -> None:
    """
    Re-initialise all LoRA weights that belong to adapter_{idx}.
    Uses A ~ Kaiming-Uniform, B = 0, with an adapter-specific RNG seed.
    """
    torch.manual_seed(base_seed + adapter_idx)

    A_suffix = f".lora_A.adapter_{adapter_idx}.weight"
    B_suffix = f".lora_B.adapter_{adapter_idx}.weight"

    for name, param in peft_model.named_parameters():
        if name.endswith(A_suffix):
            # fan_in mode is the default for Linear layers
            tmp = param.data.to(torch.float32)
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5), mode="fan_in", nonlinearity="linear")
            param.data.copy_(tmp.to(param.dtype))

        elif name.endswith(B_suffix):
            # zero so that ΔW = 0 at t = 0
            param.data.zero_()
# -----------------------------------------------------------------------------
# 4 · Main
# -----------------------------------------------------------------------------
def main():
    # parse CLI
    parser      = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # set up Accelerator & Tokenizer
    accel         = Accelerator()
    device        = accel.device

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
    tokenizer.max_length = script_args.max_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load data
    train_ds, _ = load_train_eval_dataset(
        script_args.dataset, tokenizer,
        mode=script_args.dataset_mode,
        size=100 if script_args.debug else None,
        random_seed = script_args.random_seed
    )
    eval_ds = load_eval_dataset(script_args.dataset, tokenizer)
    print('size of train dataset: ', len(train_ds))

    print('size of test dataset: ', len(eval_ds))

    collator    = DataCollatorWithPadding(
        tokenizer,
        True,
        script_args.max_length
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=script_args.per_device_eval_batch_size,
        collate_fn=collator,
        drop_last=False
    )
         
    steps_per_epoch = max(1, len(train_ds) // (
        script_args.per_device_train_batch_size *
        script_args.gradient_accumulation_steps *
        accel.num_processes
    ))
    eval_steps = max(1, int(steps_per_epoch * 0.1))
    print(f"→ evaluating every {eval_steps} steps (~0.01 epoch)")

    model_params = {
        'vhead_layer_type': script_args.layer_type,
        'vhead_num_neurons': 1024,
        'vhead_num_layers': script_args.num_layers,
    }

    model = AutoModelForCausalLMWithMultiValueHead.from_pretrained(
        script_args.base_model, device_map=device, 
        torch_dtype=torch.bfloat16,
        num_value_heads=script_args.num_adapters,
        **model_params,
    )

    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if script_args.resume_from_dir:            # ← NEW
        resume_dir = Path(script_args.resume_from_dir)
        assert resume_dir.exists(), f"{resume_dir} does not exist"
        peft_model = load_adapters_and_head(model, resume_dir, device)
        # count how many adapters we just loaded
        print(f"✓ Loaded {script_args.num_adapters} adapters from {resume_dir}")
    else:

        # prepare LoRA
        lora_cfg = LoraConfig(
            r=script_args.lora_r,
            fan_in_fan_out=True,
            lora_alpha=script_args.lora_alpha,
            target_modules=script_args.lora_target_modules,
            lora_dropout=script_args.lora_dropout,
            task_type=TaskType.SEQ_CLS,
            # modules_to_save=["score"],
        )
        peft_model = get_peft_model(model, lora_cfg, adapter_name="adapter_0", mixed=False)
        # torch.manual_seed(0 )
        init_lora_adapter(peft_model, 0, base_seed=42)
        
        for i in range(1, script_args.num_adapters):
            adapter_name = f"adapter_{i}"
            peft_model.add_adapter(
                adapter_name=f"adapter_{i}",
                peft_config=lora_cfg,
                # is_trainable=True          # <── key line
                )
            init_lora_adapter(peft_model, i, base_seed=42+i)



    print_trainable_parameters(peft_model)
    for name, param in peft_model.named_parameters():
        if "adapter_" in name or "v_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for head in peft_model.v_heads:
        if hasattr(head, "dropout"):
            head.dropout.p = 0.1              # deterministic forward

    output_name = os.path.join(script_args.log_dir,
                               f"{script_args.base_model.split('/')[-1]}_{script_args.wandb_name}"
                               f"_{'multilora'+str(script_args.num_adapters) if script_args.use_lora else 'full'}")
    training_args = TrainingArguments(
        output_dir=os.path.join(output_name, f"logs{script_args.output_tag}_{script_args.num_adapters}adps"),
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        learning_rate=script_args.learning_rate,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        bf16=script_args.bf16,
        logging_steps=1,
        warmup_ratio=0.01, #0.03
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        eval_strategy=script_args.evaluation_strategy,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        run_name=script_args.wandb_name,
        report_to=script_args.report_to,
        remove_unused_columns=False,
        gradient_checkpointing=script_args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=5.0,
    )

    # instantiate our custom trainer
    trainer = MultiAdapterRewardTrainer(
        model=peft_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=RewardDataCollatorWithPadding(tokenizer, True, script_args.max_length),
        compute_metrics=compute_accuracy,
        diversity_lambda=script_args.diversity_lambda,
        num_adapters=script_args.num_adapters,
        max_length=script_args.max_length,
        use_lora=script_args.use_lora,
    )

    # add our improved eval callback
    eval_cb = MultiAdapterEvalCallback(
        trainer=trainer,
        eval_steps=eval_steps,
        num_adapters=script_args.num_adapters,
        eval_dataset=eval_ds,
        eval_loader=eval_loader,
        accelerator=accel,
    )
    trainer.add_callback(eval_cb)
    trainer.optimizer = None
    trainer.create_optimizer() 
    trainer.train()


if __name__ == "__main__":
    main()
