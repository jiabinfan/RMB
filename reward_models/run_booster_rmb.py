#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, json, math, numpy as np, torch, xgboost as xgb
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from accelerate import Accelerator
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, HfArgumentParser
)
from torch.utils.data import DataLoader
import torch.nn as nn
from peft import PeftModel
from grm_utils import AutoModelForCausalLMWithMultiValueHead
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from peft import LoraConfig, TaskType, get_peft_model
# ──────────────────────────────────────────────────────────────────────────
# 1 · CLI / script arguments
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArguments:
    # LoRA / checkpoint selection
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Folder that contains adapter_0/, adapter_1/, …"}
    )
    adapter_glob: str = field(
        default="./reward_models_train/**/best_step_*_acc*/",
        metadata={"help": "Fallback glob to locate checkpoints if --checkpoint_dir is not given"}
    )

    # base model + runtime
    base_model: str = field(default="google/gemma-2b-it")
    max_length: int = field(default=1024)
    batch_size: int = field(default=8)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2")
    freeze_pretrained: bool = field(default=True)

    # XGBoost parameters
    tree_method: str = field(default="gpu_hist", metadata={"help": "'gpu_hist' or 'hist'"})
    num_round: int = field(default=300)
    early_stopping: int = field(default=20)
    booster_out: str = field(default="multi_lora_booster.xgb")
    xgb_max_depth: int = field(default=3)
    xgb_eta: float = field(default=0.05)
    
    layer_type: Optional[str] = field(default='mlp') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)
    # dataset
    dataset: str = field(default="llm-blender/Unified-Feedback")
    eval_dataset: str = field(default="llm-blender/Unified-Feedback")
    dataset_mode: str = field(default="40k")
    
class PairAccCB(xgb.callback.TrainingCallback):
    """Compute pair-wise accuracy on `dvalid` each round; store in self.curr_acc."""
    def __init__(self, dvalid):
        self.dvalid = dvalid
        self.gptr   = dvalid.get_uint_info("group_ptr")
        self.curr_acc = 0.0

    def after_iteration(self, model, epoch, evals_log):
        pred = model.predict(self.dvalid)
        correct = sum(pred[a] > pred[a+1]                   # row order = [chosen, rejected]
                      for a in self.gptr[:-1])              # one a per group
        self.curr_acc = correct / (len(self.gptr) - 1)
        print(f"[{epoch}]  val-pair_acc: {self.curr_acc:.4f}")
        return False                                        # keep training


class SaveBestAccCB(xgb.callback.TrainingCallback):
    """Save the booster whenever PairAccCB achieves a new high score."""
    def __init__(self, pair_cb: PairAccCB, ckpt_path: str, rank_zero: bool = True):
        self.pair_cb   = pair_cb
        self.best_acc  = -1.0
        self.ckpt_path = ckpt_path
        self.rank_zero = rank_zero

    def after_iteration(self, model, epoch, evals_log):
        acc = self.pair_cb.curr_acc
        if acc > self.best_acc + 1e-7:          # small tolerance
            self.best_acc = acc
            if self.rank_zero:
                model.save_model(self.ckpt_path)
                print(f"[{epoch}]  ▲ new best acc = {acc:.4f} ➜ saved to {self.ckpt_path}")
        return False

from pathlib import Path
import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file

PREFIX_WRONG = "base_model.model.pretrained_model.model."
PREFIX_RIGHT = "base_model.model.pretrained_model.model."
# def _fix_keys(sd, dtype):
#     """strip extra prefix + cast"""
#     return {k.replace(PREFIX_WRONG, PREFIX_RIGHT, 1): v.to(dtype)
#             for k, v in sd.items()}
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
        print("\n".join(list(sd.keys())[:10]))  # show the first 10 keys
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
        print(f"Verifying '{adapter_name}'...")
        
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
    v_head_file = root / "v_head.bin"

    if not v_head_file.exists():
        print(f"  └── ❌ ERROR: Cannot find file at {v_head_file}")
        return

    # Load from .bin file
    disk_state_dict = torch.load(v_head_file, map_location="cpu")
    # Get live state dict from the model's value head
    live_state_dict = peft_model.v_head.state_dict()

    mismatched_params = 0
    for param_key, disk_tensor in disk_state_dict.items():
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
from datasets import concatenate_datasets          # new import
from load_datasets import load_train_eval_dataset 
from load_eval_datasets import load_eval_dataset

def _set_active_adapter(wrapped, name: str):
    if hasattr(wrapped, "module"):
        wrapped.module.set_adapter(name)
    else:
        wrapped.set_adapter(name)
def lora_stats(model, adapter_name="adapter_0", n=5):
    keys = [k for k,_ in model.named_parameters() if f".{adapter_name}." in k]
    print("found", len(keys), "LoRA tensors for", adapter_name)
    for k in keys[:n]:
        t = dict(model.named_parameters())[k]
        print(f"{k}  mean|abs| = {t.float().cpu().double().abs().mean() :.4e}")

# ──────────────────────────────────────────────────────────────────────────
# 2 · Helper functions
# ──────────────────────────────────────────────────────────────────────────
def collate_fn(tokenizer, batch, max_len):
    pad_id = tokenizer.pad_token_id
    keys = ["input_ids_chosen", "attention_mask_chosen",
            "input_ids_rejected", "attention_mask_rejected"]
    out = {}
    for k in keys:
        out[k] = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ex[k][:max_len]) for ex in batch],
            batch_first=True, padding_value=pad_id
        )
    return out


# ──────────────────────────────────────────────────────────────────────────
# 3 · Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    # 4.2  – tokenizer & base LM
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    tokenizer.max_length = args.max_length
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model_params = {
        'vhead_layer_type': args.layer_type,
        'vhead_num_neurons': 1024,
        'vhead_num_layers': args.num_layers,
    }

    adapter_root = Path(args.adapter_glob or ".")
    n_adapters = len(list(adapter_root.glob("adapter_*")))
    print("num heads", n_adapters)
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
    print("Loaded adapters:", list(model.peft_config.keys()))

    # lora_stats(model, "adapter_0")
    # lora_stats(model, "adapter_1")
    # lora_stats(model, "adapter_2")
    n_adapters = len(model.peft_config)
    if is_main: print(f"✓ loaded {n_adapters} LoRA adapters")

    # -------------------------------------------------------------
    # 3.5 Load dataset
    # -------------------------------------------------------------
    from load_datasets import load_train_eval_dataset  # project helper
    train_ds, _ = load_train_eval_dataset(
        args.dataset, tokenizer, mode=args.dataset_mode
    )

    _, eval_ds = load_train_eval_dataset(
        args.eval_dataset, tokenizer, mode=args.dataset_mode
    )
    
    dl_train = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length)
    )
    dl_eval = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(tokenizer, b, args.max_length)
    )
    dl_train, dl_eval = accelerator.prepare(dl_train, dl_eval)

    # -------------------------------------------------------------
    # 3.6 Feature extraction
    # -------------------------------------------------------------
    def gather_pairwise_features(loader):
        """
        Build proper pair-wise data:
            Row-0 = chosen  (label 1)
            Row-1 = rejected(label 0)
        Order is interleaved so each query-group consists of 2 consecutive rows.
        """
        feat_rows, labels, group_sizes = [], [], []
        model.eval()

        for batch in tqdm(loader, disable=not is_main, desc="⟨GPU⟩ extract"):
            with torch.inference_mode():
                pos_cols, neg_cols = [], []
                for a in range(n_adapters):
                    adapter_name = f"adapter_{a}"
                    _set_active_adapter(model, adapter_name)
                    r_pos = model(batch["input_ids_chosen"],
                                attention_mask=batch["attention_mask_chosen"], active_head=a)[-1].squeeze()
                    r_neg = model(batch["input_ids_rejected"],
                                attention_mask=batch["attention_mask_rejected"], active_head=a)[-1].squeeze()
                    pos_cols.append(r_pos)
                    neg_cols.append(r_neg)

                # (B, N) matrices
                # print(pos_cols)
                pos_mat = torch.stack(pos_cols, dim=1)   # chosen
                neg_mat = torch.stack(neg_cols, dim=1)   # rejected

                # ── interleave:  (B, 2, N)  →  (2B, N) ────────────────────────
                pair_feat = torch.stack([pos_mat, neg_mat], dim=1) \
                                .reshape(-1, pos_mat.size(1))     # (2B, N)
                pair_lab  = torch.tensor([1, 0], device=pos_mat.device) \
                                .repeat(pos_mat.size(0))          # (2B,)

                feat_rows.append(pair_feat)
                labels.append(pair_lab)
                group_sizes.extend([2] * pos_mat.size(0))           # B groups this batch

            torch.cuda.empty_cache()

        # gather all processes
        X = accelerator.gather_for_metrics(torch.cat(feat_rows))      
        y = accelerator.gather_for_metrics(torch.cat(labels))
        g = accelerator.gather_for_metrics(torch.tensor(group_sizes, device=device))

        return X.float().cpu().numpy(), y.cpu().numpy(), g.cpu().numpy()

    if is_main: print("→ Building X_train ...")
    X_train, y_train, g_train = gather_pairwise_features(dl_train)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtrain.set_group(g_train)

    X_val, y_val, g_val = gather_pairwise_features(dl_eval)
    dval = xgb.DMatrix(X_val, label=y_val)
    dval.set_group(g_val)


    # -------------------------------------------------------------
    # 3.7 Fit XGBoost booster (rank 0 only)
    # -------------------------------------------------------------
    if is_main:
        params = dict(
            objective="rank:pairwise",
            # eval_metric="auc",
            eval_metric="ndcg@2",
            disable_default_eval_metric=1,
            tree_method=args.tree_method,
            max_depth=args.xgb_max_depth,
            min_child_weight=30,   # force larger leaves
            gamma=2.0,             # minimum loss drop to split
            lambda_=2.0,           # L2 penalty
            alpha=0.1,             # L1 penalty
            # --- add randomness ---
            subsample=0.8,         # bag rows
            colsample_bytree=0.8,  # bag features per tree
            colsample_bylevel=0.8,
            # --- slower learning ---
            eta=0.03,    
        )
        print("→ Training XGBoost …")
        pair_cb = PairAccCB(dval)
        save_cb = SaveBestAccCB(pair_cb, args.booster_out, rank_zero=is_main)
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=args.num_round,
            evals=[(dval, "val")],
            callbacks=[pair_cb, save_cb],          # <-- order matters
            early_stopping_rounds=args.early_stopping,
            verbose_eval=1,
        )
        # booster.save_model(args.booster_out)
        # print(f"✓ Booster saved ➜ {args.booster_out}")

    accelerator.wait_for_everyone()

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

