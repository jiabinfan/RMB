#!/usr/bin/env python
# -*- coding: utf-8 -*-
from safetensors.torch import load_file
from peft import AutoPeftModelForCausalLM, AutoPeftModelForSequenceClassification
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from peft import LoraConfig, TaskType, get_peft_model
import os, glob, json, math, numpy as np, torch, xgboost as xgb
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn
from peft import PeftModel
from accelerate import utils
from grm_utils import AutoModelForCausalLMWithValueHead, AutoModelForCausalLMWithMultiValueHead
from transformers import DataCollatorWithPadding
from utils import print_trainable_parameters, compute_metrics, freeze_trainable_parameters
from load_eval_datasets import load_eval_dataset


# ──────────────────────────────────────────────────────────────────────────
# NEW IMPORTS (put with your other imports)
# ──────────────────────────────────────────────────────────────────────────
import time, threading, psutil, platform
from contextlib import contextmanager

try:
    import pynvml  # optional, for GPU energy
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False

# ──────────────────────────────────────────────────────────────────────────
# NEW: small GPU power sampler (approx. Joules via NVML)
# ──────────────────────────────────────────────────────────────────────────
class GPUPowerSampler:
    """Background sampler that integrates GPU power (Watts) into Joules."""
    def __init__(self, device_index=0, hz=50):
        self.enabled = _HAS_NVML and torch.cuda.is_available()
        self.hz = hz
        self.dt = 1.0 / float(hz)
        self._lock = threading.Lock()
        self._running = False
        self._joules = 0.0
        if self.enabled:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))

    def _loop(self):
        t_prev = time.perf_counter()
        while self._running:
            try:
                p_watts = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW → W
            except Exception:
                p_watts = 0.0
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            with self._lock:
                self._joules += p_watts * dt
            time.sleep(self.dt)

    def start(self):
        if not self.enabled or self._running: return
        self._running = True
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def stop(self):
        if not self.enabled or not self._running: return
        self._running = False
        self._thr.join()

    def read_j(self) -> float:
        if not self.enabled: return 0.0
        with self._lock:
            return float(self._joules)

# ──────────────────────────────────────────────────────────────────────────
# NEW: tiny helpers
# ──────────────────────────────────────────────────────────────────────────
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _percentiles(xs):
    xs = np.array(xs, dtype=np.float64)
    if xs.size == 0: return {"mean": 0, "p50": 0, "p90": 0, "p99": 0}
    return {
        "mean": float(xs.mean()),
        "p50": float(np.percentile(xs, 50)),
        "p90": float(np.percentile(xs, 90)),
        "p99": float(np.percentile(xs, 99)),
    }

import re
# ──────────────────────────────────────────────────────────────────────────
# 1 · CLI arguments
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArgs:
    base_model: str = field(default="google/gemma-2b-it")
    checkpoint_dir: Optional[str] = field(default=None,
                                          metadata={"help": "Root folder that *contains* adapter_* dirs"})

    dataset: str = field(default="hhh", metadata={"help": "hhh | mt_bench | reward_bench"})
    dataset_mode: str = field(default="40k")
    batch_size: int = field(default=32)
    max_length: int = field(default=1024)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    tree_method: str = field(default="gpu_hist")
    layer_type: Optional[str] = field(default='mlp') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)
    num_adapters: Optional[int] = field(default=3)
    adapter_glob: str = field(default="./**/adapter_*",
                              metadata={"help": "Glob that resolves to adapter_* folders"})

    booster_path: str = field(default="multi_lora_booster.xgb")

    measure_energy: bool = field(default=True, metadata={"help": "Use NVML to estimate GPU Joules"})
    power_hz: int = field(default=50, metadata={"help": "GPU power sample rate (Hz)"})
    hourly_cost_usd: float = field(default=0.0, metadata={"help": "Optional infra cost per hour to compute $/1k samples"})
    metrics_out: Optional[str] = field(default="metrics.json", metadata={"help": "Where to write JSON metrics"})
# ──────────────────────────────────────────────────────────────────────────
# 2 · Collate & helpers
# ──────────────────────────────────────────────────────────────────────────
from pathlib import Path
import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file

def _fix_keys(sd, dtype):
    new_sd = {}
    for k, v in sd.items():
        if ".pretrained_model." not in k:       # ← old checkpoints
            k = k.replace(
                "base_model.model.model.",
                "base_model.model.pretrained_model.model.", 1
            )
        new_sd[k] = v.to(dtype)
    return new_sd

def load_adapters_and_head(base_model, ckpt_root, device, dtype=torch.bfloat16):
    root = Path(ckpt_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    # ── collect adapter_* directories in numeric order ───────────────
    adirs = sorted(root.glob("adapter_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    if not adirs:
        raise RuntimeError(f"No adapter_* dirs found in {root}")

    # ── 1️⃣ wrap base model with the *first* adapter (empty weights) ──
    model = PeftModel.from_pretrained(
        base_model,
        adirs[0] / "adapter_0",                    # supplies the LoRA config
        adapter_name=adirs[0].name,
        load_weights=False,          # ← we’ll load them manually
        is_trainable=False,
        device_map={"": device},
        torch_dtype=dtype,
    )

    # ── 2️⃣ register + load every adapter’s weights ──────────────────
    for adir in adirs:
        tmp_adp_name = "adapter_" +adir.name[-1] 
        adir = adir /  tmp_adp_name 
        name = adir.name              # e.g. adapter_1
        if name not in model.peft_config:     # first one is already there
            cfg = LoraConfig.from_pretrained(adir)
            model.add_adapter(name, cfg)

        sd = load_file(adir / "adapter_model.safetensors", device="cpu")
        #print("\n".join(list(sd.keys())[:10]))  # show the first 10 keys
        set_peft_model_state_dict(model, _fix_keys(sd, dtype), adapter_name=name)

    # ── 3️⃣ load the shared value head AFTER the PEFT wrapping ───────
    for adir in adirs:
        idx = int(adir.name.split("_")[1])
        v_path = adir / "v_head.bin"
        if v_path.exists():
            v_sd = torch.load(v_path, map_location="cpu")
            model.v_heads[idx].load_state_dict(v_sd, strict=True)
        else:
            raise FileNotFoundError(v_path)


    model.set_adapter(adirs[0].name)      # default active
    return model.to(device)


from safetensors.torch import load_file
import torch

def verify_lora_adapters(peft_model, checkpoint_dir):
    """
    Verifies that the loaded LoRA adapter weights in the PeftModel match the
    weights saved in the .safetensors files on disk.

    Args:
        peft_model (PeftModel): The model with loaded adapters.
        checkpoint_dir (str): The root directory containing the adapter folders.
    """
    print("\n--- Verifying LoRA Adapter Weights ---")
    live_state_dict = peft_model.state_dict()
    root = Path(checkpoint_dir).expanduser().resolve()
    
    # Iterate over each adapter configured in the live model
    for adapter_name in peft_model.peft_config.keys():
        # print(f"Verifying '{adapter_name}'...")
        
        adapter_file = root / adapter_name / "adapter_model.safetensors"
        if not adapter_file.exists():
            print(f"  └── ❌ ERROR: Cannot find file at {adapter_file}")
            continue

        # Load the state dict from the .safetensors file
        disk_state_dict = load_file(adapter_file, device="cpu")
        
        mismatched_params = 0
        # For each parameter in the saved file, check it against the live model
        for param_key, disk_tensor in disk_state_dict.items():
            if param_key not in live_state_dict:
                print(f"  └── ❓ WARNING: Parameter '{param_key}' found on disk but not in live model.")
                continue

            live_tensor = live_state_dict[param_key]
            
            # Compare the tensors
            if not torch.allclose(live_tensor.cpu(), disk_tensor):
                print(f"  └── ❌ MISMATCH found for parameter: {param_key}")
                mismatched_params += 1
        
        if mismatched_params == 0:
            print(f"  └── ✅ OK: All parameters for '{adapter_name}' match the file on disk.")
        else:
            print(f"  └── ❌ FAILED: {mismatched_params} parameters for '{adapter_name}' did not match.")

def verify_value_head(peft_model, checkpoint_dir):
    """
    Verifies that the loaded value head weights in the model match the
    weights saved in the v_head.bin file on disk.

    Args:
        peft_model (PeftModel): The model containing the value head.
        checkpoint_dir (str): The root directory containing v_head.bin.
    """
    print("\n--- Verifying Value Head Weights ---")
    root = Path(checkpoint_dir).expanduser().resolve()
    v_head_file = root / "adapter_0" / "v_head.bin"

    if not v_head_file.exists():
        print(f"  └── ❌ ERROR: Cannot find file at {v_head_file}")
        return

    # Load from .bin file
    disk_state_dict = torch.load(v_head_file, map_location="cpu")
    # Get live state dict from the model's value head
    live_state_dict = peft_model.v_head.state_dict()

    mismatched_params = 0
    for param_key, disk_tensor in disk_state_dict.items():
        print(param_key)
        if param_key not in live_state_dict:
            print(f"  └── ❓ WARNING: Parameter '{param_key}' found on disk but not in live model's v_head.")
            continue
        
        live_tensor = live_state_dict[param_key]
        if not torch.allclose(live_tensor.cpu().float(), disk_tensor):
            print(f"  └── ❌ MISMATCH found for v_head parameter: {param_key}")
            mismatched_params += 1
            
    if mismatched_params == 0:
        print("  └── ✅ OK: All value head parameters match the file on disk.")
    else:
        print(f"  └── ❌ FAILED: {mismatched_params} value head parameters did not match.")
# ──────────────────────────────────────────────────────────────────────────
# 3 · Dataset wrappers
# ──────────────────────────────────────────────────────────────────────────


def _set_active_adapter(wrapped, name: str):
    if hasattr(wrapped, "module"):
        wrapped.module.set_adapter(name)
    else:
        wrapped.set_adapter(name)

import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_distributions(feats: np.ndarray,
                               feature_names: list[str],
                               bins: int = 30,
                               kde: bool = True,
                               figsize: tuple[int,int] = (12,8)):
    """
    feats: shape (n_samples, n_features)
    feature_names: list of length n_features
    """
    # 1) Build DataFrame
    df = pd.DataFrame(feats, columns=feature_names)  # create tabular view :contentReference[oaicite:2]{index=2}

    # 2) Grid layout
    n = len(feature_names)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)  # multi‐subplot layout :contentReference[oaicite:3]{index=3}

    axes = axes.flatten() if n > 1 else [axes]
    for ax, feat in zip(axes, feature_names):
        sns.histplot(df[feat].dropna(),
                     bins=bins,
                     kde=kde,
                     stat="count",
                     edgecolor="white",
                     ax=ax)  # histogram + optional KDE :contentReference[oaicite:4]{index=4}
        ax.set_title(f"Distribution of {feat}")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("feature_distributions.png"); plt.close()


def plot_feature_correlation(
        feats: np.ndarray,
        feature_names: list[str],
        cmap: str = "coolwarm",
        annot: bool = True,
        figsize: tuple[int, int] = (10, 8),
        annot_fs: int = 20,   
        title_fs: int = 20,
        xlabel_fs: int = 20,
        ylabel_fs: int = 20,
        xlabel="Features", ylabel="Features",
):
    """
    feats: shape (n_samples, n_features)
    """
    df = pd.DataFrame(feats, columns=feature_names)
    corr = df.corr(method='pearson')

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        corr,
        cmap=cmap,
        vmin=0, vmax=1,
        annot=annot,
        annot_kws={"size": annot_fs},
        fmt=".2f",
        square=True,
        linewidths=0.5
    )
    
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title("HSIC Diversity", fontsize=title_fs)
    #ax.set_title("Random Seed Diversity", fontsize=title_fs)
    # ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    # ax.set_ylabel(ylabel, fontsize=ylabel_fs)

    plt.tight_layout()
    plt.savefig("feature_correlation_HSIC.pdf")
    plt.savefig("feature_correlation_0HSIC.png")
    plt.close()
    return corr
def get_feature_importance(
        booster,
        feature_names,
        importance_types = None,
        normalize = True) -> pd.DataFrame:
    # Resolve which metrics to fetch
    if importance_types is None or importance_types == "all":
        importance_types = ['gain', 'weight', 'cover',
                            'total_gain', 'total_cover']
    elif isinstance(importance_types, str):
        importance_types = [importance_types]

    # Pull raw scores once per metric
    raw = {
        m: booster.get_score(importance_type=m)     # XGBoost API :contentReference[oaicite:0]{index=0}
        for m in importance_types
    }

    # Build a row per feature
    records = []
    for idx, fname in enumerate(feature_names):
        row = {"feature": fname}
        for m in importance_types:
            row[m] = raw[m].get(f"f{idx}", 0.0)      # absent → 0
        records.append(row)

    df = pd.DataFrame(records)

    # Optional column-wise normalisation
    if normalize:
        for m in importance_types:
            s = df[m].sum()
            if s:
                df[m] /= s

    # Sort by the first metric requested
    df = df.sort_values(importance_types[3], ascending=False).reset_index(drop=True)
    return df

# ──────────────────────────────────────────────────────────────────────────
# 4 · Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = HfArgumentParser(ScriptArgs).parse_args_into_dataclasses()[0]
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process


    # >>> METRICS: init
    t_eval_start = time.perf_counter()

    # Per-batch timings (seconds)
    base_times, mean_overheads, xgb_overheads = [], [], []
    e2e_mean_times, e2e_xgb_times = [], []

    # Per-batch energy (Joules) for each stage (optional GPU-only)
    base_joules, mean_joules, xgb_joules = [], [], []

    # Sample counters
    local_total_samples = 0

    # Memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    else:
        gpu_name = "cpu"

    # Optional GPU energy meter
    gpu_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    pm = GPUPowerSampler(device_index=gpu_index, hz=args.power_hz)
    if args.measure_energy: pm.start()



    # 4.2  – tokenizer & base LM
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model_params = {
        'vhead_layer_type': args.layer_type,
        'vhead_num_neurons': 1024,
        'vhead_num_layers': args.num_layers,
    }

    adapter_root = Path(args.adapter_glob or ".")
    n_adapters = len(list(adapter_root.glob("adapter_*")))
    base = AutoModelForCausalLMWithMultiValueHead.from_pretrained(
        args.base_model, device_map=device, 
        torch_dtype=torch.bfloat16,
        num_value_heads=n_adapters,
        **model_params,
    )

    base.pretrained_model.resize_token_embeddings(len(tokenizer))
    # print_trainable_parameters(base)
    base.config.pad_token_id = tokenizer.pad_token_id
    

    resume_dir = args.adapter_glob
    model = load_adapters_and_head(base, resume_dir, device)

    n_adapters = len(model.peft_config)
    if is_main: print(f"✓ loaded {n_adapters} LoRA adapters")

    booster = xgb.Booster()
    booster.load_model(args.booster_path)
    if is_main: print(f"✓ XGBoost restored from {args.booster_path}")
    model.eval()
    # 4.3  – dataset & dataloader   

    eval_ds = load_eval_dataset(args.dataset, tokenizer)


    collator    = DataCollatorWithPadding(
        tokenizer,
        True,
        args.max_length
    )
    dl_eval = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        collate_fn=collator,
        drop_last=False
    )
         
    dl = accelerator.prepare(dl_eval)

    debug_path = "debug.csv"
    # Only have main process write header, to avoid races
    if is_main:
        import csv
        with open(debug_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            header = ["chosen", "rejected"]
            # interleave per-adapter scores: chosen, rejected
            for j in range(n_adapters):
                header += [f"adapter_{j}_chosen", f"adapter_{j}_rejected"]
            writer.writerow(header)
            
    # 4.4  – feature extraction & inference
    adapter_correct = torch.zeros(n_adapters, device=device)   # per-head win counter
    adapter_total   = 0                                        # #pairs seen (same for every head)
    sample_counter = 0
    pair_preds, pair_golds = [], []                            # for booster accuracy
    all_pos = []  # will hold arrays of shape (batch_size, n_adapters)
    all_neg = []
    pair_preds_mean = []

    for batch in tqdm(dl, disable=not is_main, desc="⟨GPU⟩ inference"):
        with torch.inference_mode():
            B = batch["input_ids"].size(0)
            local_total_samples += B
            # --------------------------------------------------
            # ❶ Collect raw logits from every adapter
            # --------------------------------------------------
            _sync(); t0 = time.perf_counter(); j0 = pm.read_j()
            pos_list, neg_list = [], []                        # each element shape = (B,)
            for a in range(n_adapters):
                adapter_name = f"adapter_{a}"
                _set_active_adapter(model, adapter_name)
                # print(batch)
                r_pos = model(batch["input_ids"], attention_mask=batch.get("attention_mask_chosen"), active_head=a)[-1].squeeze()
                r_neg = model(batch["input_ids_rejected"], attention_mask=batch.get("attention_mask_rejected"),active_head=a )[-1].squeeze()
                pos_list.append(r_pos)
                neg_list.append(r_neg)

            # shape: (B, N) after stack & transpose
            pos_mat = torch.stack(pos_list, dim=0).T           # chosen features
            neg_mat = torch.stack(neg_list, dim=0).T           # rejected features
            _sync(); t1 = time.perf_counter(); j1 = pm.read_j()

            base_dt = t1 - t0
            base_times.append(base_dt)
            base_joules.append(max(0.0, j1 - j0))
            # --------------------------------------------------
            # ❷ Per-adapter accuracy  (point-wise > comparison)
            # --------------------------------------------------
            wins = (pos_mat > neg_mat)                         # Boolean matrix (B, N) :contentReference[oaicite:2]{index=2}
            adapter_correct += wins.sum(dim=0)
            adapter_total   += pos_mat.size(0)

            # --------------------------------------------------
            # MEAN COMBINER (overhead only)
            # --------------------------------------------------
            _sync(); tm0 = time.perf_counter(); jm0 = pm.read_j()
            avg_pos = pos_mat.mean(dim=1)              # (B,)
            avg_neg = neg_mat.mean(dim=1)              # (B,)
            pair_pred_mean = (avg_pos > avg_neg)       # bool (B,)
            _sync(); tm1 = time.perf_counter(); jm1 = pm.read_j()

            mean_dt = tm1 - tm0
            mean_overheads.append(mean_dt)
            mean_joules.append(max(0.0, jm1 - jm0))
            e2e_mean_times.append(base_dt + mean_dt)
            
            # --------------------------------------------------
            # ❸ Build booster feature rows   [r₀ … r_{N-1}]
            #     – Row A = chosen, label 1
            #     – Row B = rejected, label 0
            # --------------------------------------------------
            _sync(); tx0 = time.perf_counter(); jx0 = pm.read_j()
            feats_pos = pos_mat.float().cpu().numpy()          # (B, N)
            feats_neg = neg_mat.float().cpu().numpy()

            all_pos.append(feats_pos)
            all_neg.append(feats_neg)
            # print(feats_pos.shape, feats_pos)
            p_pos = booster.inplace_predict(feats_pos)         # probability y=1 :contentReference[oaicite:3]{index=3}
            p_neg = booster.inplace_predict(feats_neg)

            # reshape to (B,) for easy comparison
            p_pos = torch.tensor(p_pos, device=device)
            p_neg = torch.tensor(p_neg, device=device)

            avg_pos = pos_mat.mean(dim=1)              # (B,)
            avg_neg = neg_mat.mean(dim=1)              # (B,)
            pair_pred_mean = (avg_pos > avg_neg)       # boolean (B,)
            pair_preds_mean.append(pair_pred_mean)
            
            pair_pred = (p_pos > p_neg)                        # ensemble vote
            pair_preds.append(pair_pred)
            pair_golds.append(torch.ones_like(pair_pred))      # gold = 1 (chosen should win)
            
            _sync(); tx1 = time.perf_counter(); jx1 = pm.read_j()
            xgb_dt = tx1 - tx0
            xgb_overheads.append(xgb_dt)
            xgb_joules.append(max(0.0, jx1 - jx0))
            e2e_xgb_times.append(base_dt + xgb_dt)
            

            # --------------------------------------------------
            # ORIGINAL bookkeeping for ensemble acc + debug
            # --------------------------------------------------
            feats_pos = feats_pos  # already built above
            feats_neg = feats_neg
            pair_preds_mean.append(pair_pred_mean)
            pair_preds.append(pair_pred)
            pair_golds.append(torch.ones_like(pair_pred))

        if is_main:
            import csv
            # helper: split prompt from a chosen/rejected pair by longest common prefix
            def _split_prompt_from_pair(chosen_text: str, rejected_text: str):
                l = 0
                for a, b in zip(chosen_text, rejected_text):
                    if a != b:
                        break
                    l += 1
                prompt = chosen_text[:l].rstrip()
                chosen_resp = chosen_text[l:].lstrip()
                rejected_resp = rejected_text[l:].lstrip()
                return prompt, chosen_resp, rejected_resp

            with open(debug_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                B = pos_mat.size(0)
                for i in range(B):
                    # decode full strings
                    chosen_full = tokenizer.decode(
                        batch["input_ids"][i],
                        skip_special_tokens=True
                    ).replace("\n", " ")
                    rejected_full = tokenizer.decode(
                        batch["input_ids_rejected"][i],
                        skip_special_tokens=True
                    ).replace("\n", " ")
                    lora_cols = []
                    for j in range(n_adapters):
                        lora_cols.append(f"{pos_mat[i, j].item():.6f}")  # adapter j, chosen
                        lora_cols.append(f"{neg_mat[i, j].item():.6f}")  # adapter j, rejected

                    row = [chosen_full, rejected_full] + lora_cols
                    writer.writerow(row)

            sample_counter += pos_mat.size(0)
            torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────
    # ❹  Gather & final metrics
    # ──────────────────────────────────────────────────────────
    # ensemble accuracy
    y_pred = accelerator.gather_for_metrics(torch.cat(pair_preds))
    y_true = accelerator.gather_for_metrics(torch.cat(pair_golds))
    ensemble_acc = (y_pred == y_true).float().mean().item()

    y_pred_mean = accelerator.gather_for_metrics(torch.cat(pair_preds_mean))
    ensemble_mean_acc = (y_pred_mean == y_true).float().mean().item()
    
    # ❷ Sum adapter_correct across all processes:
    adapter_correct_sum = utils.reduce(adapter_correct, reduction="sum")
    adapter_total_tensor = torch.tensor(adapter_total, device=device, dtype=torch.float32)
    adapter_total_sum = utils.reduce(adapter_total_tensor, reduction="sum")

    # ❸ Compute global accuracies (length = n_adapters)
    adapter_acc = (adapter_correct_sum / adapter_total_sum).cpu().numpy()
    
    if is_main:
        print(f"\n=== {args.dataset.upper()} RESULTS ===")
        print(f"Ensemble (mean of LoRA heads) accuracy : {ensemble_mean_acc*100:5.2f}%")
        print(f"Ensemble (XGB) accuracy : {ensemble_acc*100:5.2f}%")
        for i, acc in enumerate(adapter_acc):
            print(f"Adapter {i:<2d} accuracy : {acc*100:5.2f}%")
            
        pos_feats = np.vstack(all_pos)
        neg_feats = np.vstack(all_neg)
        combined_feats = np.vstack([pos_feats, neg_feats])

        # 2. Feature names
        feature_names = [f"RM {i}" for i in range(combined_feats.shape[1])]

        # 3. Distributions
        plot_feature_distributions(combined_feats, feature_names,
                                bins=40, kde=True, figsize=(12,10))

        # 4. Correlation
        corr_matrix = plot_feature_correlation(combined_feats, feature_names)

  
        imp = get_feature_importance(booster, feature_names)   # default = all
        print(imp.head(10)[['feature','gain','weight','cover','total_gain','total_cover']])



    # >>> METRICS: aggregate
    t_eval_end = time.perf_counter()
    if args.measure_energy: pm.stop()

    # Gather per-batch timings across processes
    def _gather(lst):
        if len(lst) == 0:
            ten = torch.zeros(0, device=device, dtype=torch.float32)
        else:
            ten = torch.tensor(lst, device=device, dtype=torch.float32)
        return accelerator.gather_for_metrics(ten).cpu().numpy().tolist()

    base_all   = _gather(base_times)
    mean_all   = _gather(mean_overheads)
    xgb_all    = _gather(xgb_overheads)
    e2e_mean_all = _gather(e2e_mean_times)
    e2e_xgb_all  = _gather(e2e_xgb_times)

    # Gather energy (sum over processes)
    base_j_all = sum(_gather(base_joules))
    mean_j_all = sum(_gather(mean_joules))
    xgb_j_all  = sum(_gather(xgb_joules))

    # Samples and wall time
    tot_samples = accelerator.gather_for_metrics(
        torch.tensor([local_total_samples], device=device, dtype=torch.long)
    ).sum().item()
    wall_s = t_eval_end - t_eval_start

    # Memory (peak GPU, max across workers)
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated()
        peak_mem = accelerator.reduce(
            torch.tensor(peak_mem, device=device, dtype=torch.long), reduction="max"
        ).item()
    else:
        peak_mem = psutil.Process().memory_info().rss

    if is_main:
        # Stats
        stats = {
            "latency": {
                "base": _percentiles(base_all),
                "mean_overhead": _percentiles(mean_all),
                "xgb_overhead": _percentiles(xgb_all),
                "e2e_mean": _percentiles(e2e_mean_all),
                "e2e_xgb": _percentiles(e2e_xgb_all),
                "units": "seconds/sample (batch-normalized)",
            },
            "throughput": {
                "overall_wall": {
                    "samples_per_s": (tot_samples / wall_s) if wall_s > 0 else 0.0,
                    "samples": tot_samples,
                    "wall_time_s": wall_s,
                },
                "from_latency": {
                    "mean_e2e_mean": (1.0 / max(1e-12, _percentiles(e2e_mean_all)["mean"])),
                    "mean_e2e_xgb":  (1.0 / max(1e-12, _percentiles(e2e_xgb_all)["mean"])),
                }
            },
            "overhead_vs_base": {
                "mean/ms":  1e3 * _percentiles(mean_all)["mean"],
                "xgb/ms":   1e3 * _percentiles(xgb_all)["mean"],
            },
            "memory": {
                "peak_bytes": int(peak_mem),
                "device": gpu_name,
            },
            "energy_gpu": {
                "base_J": float(base_j_all),
                "mean_J": float(mean_j_all),
                "xgb_J":  float(xgb_j_all),
                "e2e_mean_J": float(base_j_all + mean_j_all),
                "e2e_xgb_J":  float(base_j_all + xgb_j_all),
                "J_per_sample_e2e_mean": float((base_j_all + mean_j_all) / max(1, tot_samples)),
                "J_per_sample_e2e_xgb":  float((base_j_all + xgb_j_all) / max(1, tot_samples)),
                "enabled": bool(args.measure_energy and _HAS_NVML and torch.cuda.is_available()),
            },
            "cost": {
                "hourly_cost_usd": float(args.hourly_cost_usd),
                "total_cost_usd": float(args.hourly_cost_usd * (wall_s / 3600.0)),
                "usd_per_1k_samples_e2e_mean": float(
                    (args.hourly_cost_usd * (wall_s / 3600.0)) / max(1e-9, tot_samples) * 1000.0
                ),
                "usd_per_1k_samples_e2e_xgb": float(
                    (args.hourly_cost_usd * (wall_s / 3600.0)) / max(1e-9, tot_samples) * 1000.0
                ),
            },
            "env": {
                "hostname": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "bf16": bool(args.bf16),
                "fp16": bool(args.fp16),
                "batch_size": int(args.batch_size),
                "max_length": int(args.max_length),
                "n_adapters": int(n_adapters),
            }
        }

        # Pretty print a quick table
        def _ms(x): return 1e3 * x
        lat_e2e_mean = stats["latency"]["e2e_mean"]; lat_e2e_xgb = stats["latency"]["e2e_xgb"]
        print("\n=== INFERENCE EFFICIENCY ===")
        print(f"Samples: {tot_samples} | Wall: {wall_s:.3f}s | Device: {gpu_name}")
        print(f"Throughput (wall): {stats['throughput']['overall_wall']['samples_per_s']:.2f} samples/s")
        print(f"E2E(mean) p50/p90/p99: "
              f"{_ms(lat_e2e_mean['p50']):.2f}/{_ms(lat_e2e_mean['p90']):.2f}/{_ms(lat_e2e_mean['p99']):.2f} ms")
        print(f"E2E(XGB ) p50/p90/p99: "
              f"{_ms(lat_e2e_xgb['p50']):.2f}/{_ms(lat_e2e_xgb['p90']):.2f}/{_ms(lat_e2e_xgb['p99']):.2f} ms")
        print(f"Overhead mean vs base: {_ms(stats['overhead_vs_base']['mean/ms']):.2f} ms")
        print(f"Overhead XGB  vs base: {_ms(stats['overhead_vs_base']['xgb/ms']):.2f} ms")
        if stats["energy_gpu"]["enabled"]:
            print(f"Energy (J/sample) E2E mean/XGB: "
                  f"{stats['energy_gpu']['J_per_sample_e2e_mean']:.4f} / {stats['energy_gpu']['J_per_sample_e2e_xgb']:.4f}")

        # Optional: write JSON
        if args.metrics_out:
            import json, os
            os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
            with open(args.metrics_out, "w") as f:
                json.dump(stats, f, indent=2)

        # plot_feature_importance(imp_df, top_n=10)
if __name__ == "__main__":
    main()
