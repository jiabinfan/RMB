#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train N jointly optimized LoRA reward models with independent value heads.
The Bradley-Terry loss is regularized by normalized HSIC on preference margins.
"""

import os, math
from contextlib import contextmanager, nullcontext
from functools import partial
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import torch.nn as nn
from pathlib import Path
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    set_seed,
)
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from grm_utils import AutoModelForCausalLMWithMultiValueHead
# PEFT / TRL
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file
from trl.trainer.utils import compute_accuracy

# project helpers
from reward_trainer import RewardTrainer, RewardDataCollatorWithPadding
from load_datasets import load_train_eval_dataset
from load_eval_datasets import load_eval_dataset
from utils import print_trainable_parameters

# -----------------------------------------------------------------------------
# 1 · CLI / script arguments
# -----------------------------------------------------------------------------
@dataclass
class ScriptArguments:
    per_device_train_batch_size: int   = field(default=2) #8
    gradient_accumulation_steps: int   = field(default=8)
    learning_rate:            float    = field(default=1e-5)
    num_train_epochs:         int      = field(default=2)
    max_steps:                int      = field(default=-1)
    optim:                    str      = field(default="adamw_torch")
    lr_scheduler_type:        str      = field(default="cosine")
    max_length:               int      = field(default=1024)
    gradient_checkpointing:   bool     = field(default=True)
    bf16:                     bool     = field(default=True)
    attn_implementation:      str      = field(default="sdpa")

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
    num_adapters:        int         = field(default=3)
    diversity_lambda:    float       = field(default=0.1)
    diversity_type:      str         = field(default="hsic", metadata={"help": "one of: 'none', 'hsic'"})

    per_device_eval_batch_size: int  = field(default=8) #16
    eval_tasks:                 str  = field(default="unified",
                                              metadata={"help": "comma-sep validation sets: unified,hhh,mt,rewardbench,skywork"})
    eval_size:                  int  = field(default=1000)
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
def _unwrap_model(wrapped):
    """Return the PEFT model underneath DDP/Accelerate wrappers."""
    return wrapped.module if hasattr(wrapped, "module") else wrapped


def _lora_parameters(wrapped):
    """Cache all LoRA tensors once; PEFT's set_adapter() otherwise freezes peers."""
    model = _unwrap_model(wrapped)
    cached = getattr(model, "_joint_lora_parameters", None)
    if cached is None:
        cached = tuple(
            param
            for name, param in model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        )
        model._joint_lora_parameters = cached
    return cached


def _restore_all_lora_trainable(wrapped):
    for param in _lora_parameters(wrapped):
        param.requires_grad_(True)


def _set_active_adapter(wrapped, name: str, *, joint_training: bool = False):
    """Select one adapter, optionally preserving trainability of every adapter.

    PEFT deliberately makes only the selected adapter trainable. That default is
    correct for single-adapter training, but not for a loss whose graph contains
    forwards from several adapters and is backpropagated only after the loop.
    """
    model = _unwrap_model(wrapped)
    model.set_adapter(name)
    if joint_training:
        _restore_all_lora_trainable(model)


def _is_active_adapter(model, name: str) -> bool:
    active = getattr(model, "active_adapter", None)
    return active == name or active == [name] or active == (name,)


@contextmanager
def _adapter_recompute_context(wrapped, name: str):
    """Bind checkpoint recomputation to the adapter used by the original forward."""
    model = _unwrap_model(wrapped)
    if not _is_active_adapter(model, name):
        model.set_adapter(name)
        _restore_all_lora_trainable(model)
    yield


def _configure_checkpoint_for_adapter(wrapped, name: str):
    """Attach an adapter-specific context to non-reentrant checkpoint calls.

    Transformers stores the checkpoint callable on modules. Each forward captures
    the callable currently installed, so its backward recomputation cannot
    accidentally use whichever adapter happened to be selected last in the loop.
    """
    model = _unwrap_model(wrapped)

    def context_fn():
        return nullcontext(), _adapter_recompute_context(model, name)

    checkpoint_func = partial(
        torch_checkpoint,
        use_reentrant=False,
        context_fn=context_fn,
    )
    checkpoint_modules = getattr(model, "_adapter_checkpoint_modules", None)
    if checkpoint_modules is None:
        checkpoint_modules = tuple(
            module
            for module in model.modules()
            if getattr(module, "gradient_checkpointing", False)
        )
        if checkpoint_modules:
            model._adapter_checkpoint_modules = checkpoint_modules
    for module in checkpoint_modules:
        module._gradient_checkpointing_func = checkpoint_func
    if not checkpoint_modules:
        # Harmless when gradient checkpointing is disabled; useful for diagnosing
        # an unexpected Transformers implementation when it was requested.
        return False
    return True

def _fix_lora_keys(state, dtype):
    fixed = {}
    for key, value in state.items():
        if ".pretrained_model." not in key:
            key = key.replace(
                "base_model.model.model.",
                "base_model.model.pretrained_model.model.",
                1,
            )
        fixed[key] = value.to(dtype)
    return fixed


def _adapter_payload(outer: Path) -> Path:
    name = outer.name
    for candidate in (outer / name, outer):
        if (
            (candidate / "adapter_config.json").is_file()
            and (candidate / "adapter_model.safetensors").is_file()
        ):
            return candidate
    raise FileNotFoundError(f"No complete LoRA payload found in {outer}")


def load_adapters_and_head(base_model, ckpt_dir: Path, device):
    root = Path(ckpt_dir).expanduser().resolve()
    adapter_dirs = sorted(
        (path for path in root.glob("adapter_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("adapter_")),
    )
    indices = [int(path.name.removeprefix("adapter_")) for path in adapter_dirs]
    if not adapter_dirs:
        raise FileNotFoundError(f"No adapter_N directories found in {root}")
    if indices != list(range(len(adapter_dirs))):
        raise ValueError(
            f"Adapters must be contiguous and zero-based; found {indices}"
        )

    payloads = [_adapter_payload(path) for path in adapter_dirs]
    first_config = LoraConfig.from_pretrained(payloads[0])
    peft_model = get_peft_model(
        base_model, first_config, adapter_name="adapter_0", mixed=False
    )

    for index, (outer, payload) in enumerate(zip(adapter_dirs, payloads)):
        name = f"adapter_{index}"
        if name not in peft_model.peft_config:
            peft_model.add_adapter(name, LoraConfig.from_pretrained(payload))
        state = load_file(payload / "adapter_model.safetensors", device="cpu")
        set_peft_model_state_dict(
            peft_model,
            _fix_lora_keys(state, torch.bfloat16),
            adapter_name=name,
        )

        head_path = next(
            (
                path
                for path in (outer / "v_head.bin", payload / "v_head.bin")
                if path.is_file()
            ),
            None,
        )
        if head_path is None:
            raise FileNotFoundError(f"No v_head.bin found for {name}")
        head_state = torch.load(head_path, map_location="cpu", weights_only=True)
        peft_model.v_heads[index].load_state_dict(head_state, strict=True)

    peft_model.set_adapter("adapter_0")
    return peft_model.to(device)

def save_all_components(accelerator, trainer, tokenizer, save_dir: str):
    model_to_save = accelerator.unwrap_model(trainer.model)
    tokenizer.save_pretrained(save_dir)

    # Save each value head alongside its adapter
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

    # NOTE: the FROZEN backbone is intentionally NOT saved here. It is byte-identical
    # to the base model and was previously re-written (~5.7GB) into every best_step_*
    # dir -> tens of GB of redundant copies per run. The LoRA adapters + value heads
    # saved above are sufficient to rebuild each reward head on top of `base_model`.




# Normalized HSIC is scale-stable, uses a median RBF bandwidth, averages all
# adapter pairs, and operates on shift-invariant preference margins. The caller
# adds this non-negative dependence measure to the minimized training objective.
def _rbf_gram_v2(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    x2 = (x * x).sum(1, keepdim=True)
    d2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
    return torch.exp(-d2 / (2.0 * sigma * sigma + 1e-12))

def _median_bandwidth_v2(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        x2 = (x * x).sum(1, keepdim=True)
        d2 = (x2 + x2.t() - 2.0 * (x @ x.t())).clamp_min(0.0)
        m = x.shape[0]
        iu = torch.triu_indices(m, m, offset=1, device=x.device)
        dvals = d2[iu[0], iu[1]].sqrt()
        return torch.clamp(torch.median(dvals), min=1e-3)

def _normalized_hsic_v2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = x.shape[0]
    K = _rbf_gram_v2(x, _median_bandwidth_v2(x))
    L = _rbf_gram_v2(y, _median_bandwidth_v2(y))
    H = torch.eye(m, device=x.device, dtype=x.dtype) - 1.0 / m
    Kc = H @ K @ H
    Lc = H @ L @ H
    hxy = (Kc * Lc).sum()
    hxx = (Kc * Kc).sum()
    hyy = (Lc * Lc).sum()
    return hxy / (torch.sqrt(hxx * hyy) + 1e-12)

def hsic_diversity_penalty(margins: torch.Tensor) -> torch.Tensor:
    """
    margins: [M, N] per-pair margins for N adapters over M samples.
    Returns the mean pairwise normalized-HSIC (in [0, 1]); MINIMIZE to promote
    diversity. Standardizes each adapter column first (scale-invariance).
    """
    margins = margins.float()
    M, N = margins.shape
    if M < 4 or N < 2:                       # too few samples -> no-op (keep graph)
        return margins.sum() * 0.0
    mu = margins.mean(0, keepdim=True)
    sd = margins.std(0, keepdim=True) + 1e-6
    Z = (margins - mu) / sd
    total = Z.new_zeros(())
    cnt = 0
    for i in range(N):
        for j in range(i + 1, N):
            total = total + _normalized_hsic_v2(Z[:, i:i + 1], Z[:, j:j + 1])
            cnt += 1
    return total / max(cnt, 1)        # minimize dependence to promote diversity

# -----------------------------------------------------------------------------
# 2 · Custom Callback for per-0.01-epoch eval using Trainer.evaluate()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class MultiAdapterEvalCallback(TrainerCallback):
    """Validate every `eval_steps` on a LIST of eval sets (Unified-Feedback,
    HHH-Align, MT-Bench, RewardBench, ...). Reports per-dataset, per-adapter
    pairwise accuracy and selects the best checkpoint on the mean over datasets
    of the mean-per-adapter accuracy.

    Each process evaluates the full (subsampled) set independently in eval/no_grad
    mode -> identical results across ranks -> no cross-process collective needed
    (this avoids the gather_for_metrics / group-alignment hazards)."""

    def __init__(self, trainer, eval_steps, num_adapters, eval_sets, accelerator):
        super().__init__()
        self.trainer       = trainer
        self.eval_steps    = eval_steps
        self.num_adapters  = num_adapters
        self.eval_sets     = eval_sets            # list of (name, dataset, loader)
        self.accelerator   = accelerator
        self.best_accuracy = 0.0

    def _adapter_acc(self, loader, i):
        # Shard eval across GPUs: each rank handles every ws-th batch, then all-reduce
        # the counts (4x fewer forwards/rank vs the old full-set-per-rank scheme).
        ws   = self.accelerator.num_processes
        rank = self.accelerator.process_index
        correct, total = 0, 0
        for bi, batch in enumerate(loader):
            if bi % ws != rank:
                continue
            c_ids  = batch["input_ids"].to(self.accelerator.device)
            c_mask = batch["attention_mask_chosen"].to(self.accelerator.device)
            r_ids  = batch["input_ids_rejected"].to(self.accelerator.device)
            r_mask = batch["attention_mask_rejected"].to(self.accelerator.device)
            rc = self.trainer.model(c_ids, attention_mask=c_mask, active_head=i)[-1].reshape(-1)
            rr = self.trainer.model(r_ids, attention_mask=r_mask, active_head=i)[-1].reshape(-1)
            comp = rc > rr
            correct += int(comp.sum().item())
            total   += int(comp.numel())
            del rc, rr, comp
        ct = self.accelerator.reduce(
            torch.tensor([correct, total], device=self.accelerator.device, dtype=torch.long),
            reduction="sum")
        return int(ct[0].item()), max(int(ct[1].item()), 1)

    def _validate(self, step, save=True):
        is_main = self.accelerator.is_main_process
        if is_main:
            print(f"\n===== Validation @ step {step} =====")
        self.trainer.model.eval()
        per_dataset_mean = []
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for name, ds, loader in self.eval_sets:
                accs = []
                for i in range(self.num_adapters):
                    if self.trainer.use_lora:
                        _set_active_adapter(self.trainer.model, f"adapter_{i}")
                    c, t = self._adapter_acc(loader, i)
                    acc = c / t
                    accs.append(acc)
                    if is_main:
                        print(f"  [{name:16s}] adapter_{i} acc = {acc:.4f}")
                if accs:
                    m = sum(accs) / len(accs)
                    per_dataset_mean.append(m)
                    if is_main:
                        print(f"  [{name:16s}] mean-adapter   = {m:.4f}")
        if is_main and per_dataset_mean:
            sel = sum(per_dataset_mean) / len(per_dataset_mean)
            print(f"  >> selection score (mean over {len(per_dataset_mean)} sets) = {sel:.4f}")
            if save and sel > self.best_accuracy:
                self.best_accuracy = sel
                save_dir = (Path(self.trainer.args.output_dir) /
                            f"best_step_{step}_acc={self.best_accuracy:.4f}")
                save_all_components(self.accelerator, self.trainer, self.trainer.tokenizer, save_dir)
                print(f"new best saved at {save_dir} (sel {self.best_accuracy:.4f})")
        if self.trainer.use_lora:
            _set_active_adapter(self.trainer.model, "adapter_0")
        self.trainer.model.train()
        if is_main:
            print("==========================================\n")

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # STEP-0 SANITY VALIDATION: exercise the full eval path (sharded forward +
        # all-reduce across ranks) BEFORE any training step, so an eval bug fails fast
        # in the first minutes instead of after eval_steps (e.g. the FA2/pad crash that
        # took 82 steps to surface). No checkpoint saved for the untrained model.
        if self.accelerator.is_main_process:
            print("[step-0] sanity validation to verify the eval pipeline works ...")
        self._validate(0, save=False)
        return control

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            self._validate(state.global_step, save=True)
        return control


class AdapterGradientSanityCallback(TrainerCallback):
    """Fail fast unless every adapter receives a real LoRA gradient."""

    def __init__(self, num_adapters: int):
        self.num_adapters = num_adapters
        self.checked = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.checked or model is None:
            return control

        core = _unwrap_model(model)
        norms = {}
        missing = []
        for i in range(self.num_adapters):
            marker = f".lora_B.adapter_{i}."
            grads = [
                param.grad
                for name, param in core.named_parameters()
                if marker in name
            ]
            finite_grads = [
                grad for grad in grads
                if grad is not None and torch.isfinite(grad).all()
            ]
            norm = sum(float(grad.float().norm().item()) for grad in finite_grads)
            norms[f"adapter_{i}"] = norm
            if not grads or len(finite_grads) != len(grads) or norm == 0.0:
                missing.append(f"adapter_{i}")

        if missing:
            raise RuntimeError(
                "Joint-adapter gradient check failed before the first optimizer "
                f"step; missing/non-finite LoRA-B gradients for {missing}. "
                f"Observed norms: {norms}"
            )
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[gradient-check] all adapters have finite LoRA-B gradients: {norms}")
        self.checked = True
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
        # honor the CLI/constructor value instead of silently hardcoding it
        self.diversity_type   = kwargs.pop("diversity_type", "hsic")
        if self.diversity_type not in {"none", "hsic"}:
            raise ValueError(f"Unsupported diversity_type: {self.diversity_type}")
        from collections import deque
        # rolling buffer of detached per-pair margins so the HSIC kernel estimate
        # is meaningful despite the tiny per-step micro-batch (B=2). maxlen*B+B samples.
        self._margin_buf      = deque(maxlen=31)
        super().__init__(**kwargs)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # print("inputs", inputs)
        chosen, rejected = [], []
        for i in range(self.num_adapters):
            if self.use_lora:
                adapter_name = f"adapter_{i}"
                # PEFT set_adapter() freezes every inactive adapter. Restore the
                # joint trainable partition because backward happens only after all
                # adapter forwards have contributed to the combined loss.
                _set_active_adapter(
                    model,
                    adapter_name,
                    joint_training=model.training,
                )
                if model.training:
                    # Non-reentrant checkpointing reruns layers during backward.
                    # Capture this adapter now instead of reading the mutable
                    # active_adapter value left by the final forward.
                    _configure_checkpoint_for_adapter(model, adapter_name)

            rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], active_head=i)[-1]
            # if "0" in str(rewards.device):
            #     # print("alist", alist)
            #     print("rewards", rewards)
            bsz = rewards.size(0)
            if bsz % 2:
                raise ValueError(
                    "Reward batches must contain interleaved chosen/rejected rows"
                )
            jidx = torch.arange(0, bsz, 2)
            kidx = jidx + 1
            rewards_j = rewards[jidx]
            rewards_k = rewards[kidx]

            # out = model(input_ids=inp_ids, attention_mask=attn, return_dict=True)
            # logits = out.logits
            # chosen.append(logits[0::2])
            # rejected.append(logits[1::2])

            chosen.append(rewards_j)
            rejected.append(rewards_k)
            del rewards                       # drop the per-adapter reward tensor
            # (no torch.cuda.empty_cache() here: it forces a device sync every
            #  micro-step and fights PYTORCH_CUDA_ALLOC_CONF=expandable_segments)

        target = inputs.get("preference_target")
        if target is None:
            # Keep the pre-RRM objective byte-for-byte equivalent for all old
            # datasets rather than routing it through the soft-label loss.
            pref_loss = sum(
                -nn.functional.logsigmoid(rc - rr).mean()
                for rc, rr in zip(chosen, rejected)
            ) / self.num_adapters
            directional_mask = torch.ones(
                chosen[0].shape[0], device=chosen[0].device, dtype=torch.bool
            )
        else:
            target = target.to(
                device=chosen[0].device, dtype=chosen[0].dtype
            ).reshape_as(chosen[0])
            pref_loss = sum(
                nn.functional.binary_cross_entropy_with_logits(rc - rr, target)
                for rc, rr in zip(chosen, rejected)
            ) / self.num_adapters
            directional_mask = target.reshape(-1) > 0.5

        if self.diversity_lambda > 0:
            rewards = torch.stack([rc.squeeze(-1) for rc in chosen], dim=1)  # [B, N]
            rewards_rejected = torch.stack([rc.squeeze(-1) for rc in rejected], dim=1)  # [B, N]

            margin_cur = rewards - rewards_rejected          # [B, N] per-pair margins
            # RRM tie pairs enforce prompt-invariant calibration and should not
            # simultaneously be pushed toward cross-adapter disagreement. Learn
            # ensemble diversity only from directional preference pairs.
            margin_directional = margin_cur[directional_mask]
            if self.diversity_type == "hsic":
                # corrected, normalized HSIC on margins, with a rolling detached buffer
                if len(self._margin_buf) > 0:
                    hist = torch.cat(list(self._margin_buf), dim=0).to(margin_cur.dtype)
                    margin_all = torch.cat([hist, margin_directional], dim=0)
                else:
                    margin_all = margin_directional
                if margin_directional.shape[0] == 0:
                    div_term = margin_cur.sum() * 0.0
                else:
                    div_term = hsic_diversity_penalty(margin_all)
                    self._margin_buf.append(margin_directional.detach())
            else:            # 'none'
                div_term = rewards.sum() * 0.0
            div_loss =  self.diversity_lambda * div_term
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
    set_seed(script_args.random_seed)

    # set up Accelerator & Tokenizer
    accel         = Accelerator()
    device        = accel.device

    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=True)
    tokenizer.max_length = script_args.max_length
    # Use right padding consistently for training and validation collation.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Some base models (e.g. Mistral-7B-v0.3) ship NO chat template, but the data
    # pipeline calls apply_chat_template -> install a minimal one if missing.
    if getattr(tokenizer, "chat_template", None) is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + eos_token }}"
            "{% else %}{{ message['content'] }}{% endif %}"
            "{% endfor %}"
        )

    # load data
    train_ds, _ = load_train_eval_dataset(
        script_args.dataset, tokenizer,
        mode=script_args.dataset_mode,
        size=100 if script_args.debug else None,
        random_seed=script_args.random_seed,
        load_eval=False,
    )
    # Use in-distribution validation for selection unless eval_tasks is overridden.
    _EVAL_TASK_MAP = {
        "unified": "llm-blender/Unified-Feedback", "hhh": "hhh", "mt": "mt",
        "rewardbench": "rewardbench_overall", "skywork": "skywork",
    }
    eval_sets = []
    for _key in [t.strip() for t in script_args.eval_tasks.split(",") if t.strip()]:
        _task = _EVAL_TASK_MAP.get(_key, _key)
        try:
            _ds = load_eval_dataset(
                _task, tokenizer, size=script_args.eval_size or None
            )
            if script_args.eval_size and len(_ds) > script_args.eval_size:
                _ds = _ds.select(range(script_args.eval_size))
            _loader = DataLoader(_ds, batch_size=script_args.per_device_eval_batch_size,
                                 shuffle=False, drop_last=False)
            eval_sets.append((_key, _ds, _loader))
            print(f"  eval set '{_key}' ({_task}): {len(_ds)} pairs")
        except Exception as _e:
            print(f"  could not load eval set '{_key}' ({_task}): {_e}")
    if not eval_sets:
        raise RuntimeError("no validation sets could be loaded")
    eval_ds = eval_sets[0][1]   # primary set (back-compat for trainer arg)
    print('size of train dataset: ', len(train_ds))

    print('size of test dataset: ', len(eval_ds))



    steps_per_epoch = max(1, len(train_ds) // (
        script_args.per_device_train_batch_size *
        script_args.gradient_accumulation_steps *
        accel.num_processes
    ))
    eval_steps = max(1, int(steps_per_epoch * 0.10))   # ~10% epoch; sharded eval keeps it cheap
    print(f"Evaluating every {eval_steps} steps (~0.10 epoch) on {len(eval_sets)} validation sets")

    model_params = {
        'vhead_layer_type': script_args.layer_type,
        'vhead_num_neurons': script_args.num_neurons,
        'vhead_num_layers': script_args.num_layers,
    }
    # SDPA is the right-padding-safe default used by the reproducible scripts.
    _attn = script_args.attn_implementation
    model = AutoModelForCausalLMWithMultiValueHead.from_pretrained(
        script_args.base_model, device_map=device,
        torch_dtype=torch.bfloat16,
        num_value_heads=script_args.num_adapters,
        attn_implementation=_attn,
        **model_params,
    )
    print(f"[attn] attention implementation = {_attn}")
    # if script_args.freeze_pretrained:

    #     mlp = nn.Sequential(
    #         nn.Linear(model.config.hidden_size, 1024, dtype=torch.bfloat16),
    #         nn.ReLU(),
    #         nn.Linear(1024, 1, dtype=torch.bfloat16)
    #     ).to(device)
    #     model.score = mlp

    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    # print_trainable_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    # print_trainable_parameters(model, print_trainable_name=True)

    if script_args.resume_from_dir:            # ← NEW
        resume_dir = Path(script_args.resume_from_dir)
        assert resume_dir.exists(), f"{resume_dir} does not exist"
        peft_model = load_adapters_and_head(model, resume_dir, device)
        # count how many adapters we just loaded
        # script_args.num_adapters = len(peft_model.peft_config)    # updates CLI value
        print(f"Loaded {script_args.num_adapters} adapters from {resume_dir}")
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
            peft_model.add_adapter(
                adapter_name=f"adapter_{i}",
                peft_config=lora_cfg,
                # is_trainable=True          # <── key line
                )
            # init_lora_adapter itself adds adapter_idx to base_seed.
            init_lora_adapter(peft_model, i, base_seed=42)

    # PEFT creates the first adapter in FP32 but later adapters can inherit BF16
    # from the backbone. Keep optimizer precision identical across ensemble members.
    for name, param in peft_model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.data = param.data.float()

    for name, param in peft_model.named_parameters():
        if "adapter_" in name or "v_head" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    print_trainable_parameters(peft_model)
    for head in peft_model.v_heads:
        if hasattr(head, "dropout"):
            head.dropout.p = 0.1              # paper dropout setting

    # training args
    output_name = os.path.join(script_args.log_dir,
                               f"{script_args.base_model.split('/')[-1]}_{script_args.wandb_name}"
                               f"_{'multilora'+str(script_args.num_adapters) if script_args.use_lora else 'full'}")
    training_args = TrainingArguments(
        output_dir=os.path.join(output_name, f"logs{script_args.output_tag}_{script_args.num_adapters}adps"),
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        learning_rate=script_args.learning_rate,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        bf16=script_args.bf16,
        logging_steps=1,
        warmup_ratio=0.03,
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
        diversity_type=script_args.diversity_type,
        num_adapters=script_args.num_adapters,
        max_length=script_args.max_length,
        use_lora=script_args.use_lora,
    )

    # add our improved eval callback
    eval_cb = MultiAdapterEvalCallback(
        trainer=trainer,
        eval_steps=eval_steps,
        num_adapters=script_args.num_adapters,
        eval_sets=eval_sets,
        accelerator=accel,
    )
    trainer.add_callback(eval_cb)
    trainer.add_callback(AdapterGradientSanityCallback(script_args.num_adapters))
    trainer.optimizer = None
    trainer.create_optimizer()
    trainer.train()


if __name__ == "__main__":
    main()
