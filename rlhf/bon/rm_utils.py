import os
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import HfArgumentParser, AutoModelForSequenceClassification, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np
import pandas as pd          
tqdm.pandas()
from peft import LoraConfig, PeftModel
import matplotlib.pyplot as plt
from model_utils import load_model_withhead
from grm_utils import AutoModelForCausalLMWithMultiValueHead
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from peft import LoraConfig, TaskType, get_peft_model
import os, glob, json, math, numpy as np, torch, xgboost as xgb

def load_reward_model(script_args, gpu_id, rm_type=None):
    ### here we use device map to put large reward models to empty gpus to avoid memory error
    if '7B' in script_args.reward_peft_path:
        rm_gpu_id = {
            0: 4,
            1: 4,
            2: 5,
            3: 5,
        }[gpu_id]
    else:
        rm_gpu_id = gpu_id

    rm_load_params = {
        "num_labels": 1,
        "device_map": rm_gpu_id,
        "torch_dtype": torch.bfloat16,
    }

    if len(script_args.attn_implementation):
        rm_load_params["attn_implementation"] = script_args.attn_implementation

    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_base_model, use_fast = False)
    rm_tokenizer.model_max_length = script_args.max_length
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    if rm_type == 'grm':
        reward_model = load_model_withhead(script_args.reward_base_model, script_args.reward_peft_path, \
                         rm_tokenizer, rm_gpu_id, layer_type=script_args.layer_type, num_layers=script_args.num_layers)
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_base_model,
            **rm_load_params
            )

        if os.path.exists(script_args.reward_peft_path):
            reward_model = PeftModel.from_pretrained(reward_model, script_args.reward_peft_path)
        if hasattr(reward_model, 'merge_and_unload'):
            reward_model = reward_model.merge_and_unload()
        reward_model.config.pad_token_id = rm_tokenizer.pad_token_id

    return reward_model, rm_tokenizer, rm_gpu_id

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
        # print("\n".join(list(sd.keys())[:10]))  # show the first 10 keys
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


def load_boost_reward_model(script_args, device, is_main, rm_type=None):
    ### here we use device map to put large reward models to empty gpus to avoid memory error
  
    # rm_gpu_id = gpu_id

    rm_load_params = {
        # "num_labels": 1,
        "device_map": device,
        "torch_dtype": torch.bfloat16,
    }

    if len(script_args.attn_implementation):
        rm_load_params["attn_implementation"] = script_args.attn_implementation

    rm_tokenizer = AutoTokenizer.from_pretrained(script_args.reward_base_model, use_fast = False)
    rm_tokenizer.model_max_length = script_args.max_length
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
    
    model_params = {
        'vhead_layer_type': script_args.layer_type,
        'vhead_num_neurons': 1024,
        'vhead_num_layers': script_args.num_layers,
    }
    

    adapter_root = Path(script_args.adapter_glob or ".")
    n_adapters = len(list(adapter_root.glob("adapter_*")))
    print("num heads", n_adapters)
    base = AutoModelForCausalLMWithMultiValueHead.from_pretrained(
        script_args.base_model_name, device_map=device, 
        torch_dtype=torch.bfloat16,
        num_value_heads=n_adapters,
        **model_params,
    )

    base.pretrained_model.resize_token_embeddings(len(rm_tokenizer))
    # print_trainable_parameters(base)
    base.config.pad_token_id = rm_tokenizer.pad_token_id
    

    resume_dir = script_args.adapter_glob
    peft_model = load_adapters_and_head(base, resume_dir, device)
    # print("Loaded adapters:", list(peft_model.peft_config.keys()))
    # print("params loaded:", len([k for k,_ in peft_model.named_parameters() if "lora_A.adapter_0" in k]))


    n_adapters = len(peft_model.peft_config)
    if is_main: print(f"✓ loaded {n_adapters} LoRA adapters")

    booster = xgb.Booster()
    booster.load_model(script_args.booster_path)
    if is_main: print(f"✓ XGBoost restored from {script_args.booster_path}")
    

    return peft_model, booster, n_adapters, rm_tokenizer, device

class RMEnsemble():
    def __init__(self, ensemble_method='avg', base_model_name='', peft_path_list=[]):
        self.ensemble_method = ensemble_method
        self.base_model_name = base_model_name
        self.peft_path_list = peft_path_list
        self.reward_models = []
        self.gpu_ids = []
        self.rm_tokenizers = []

    def load_reward_models(self, script_args, gpu_id):
        for _ in range(len(self.peft_path_list)):
            reward_model, rm_tokenizer, rm_gpu_id = load_reward_model(script_args, gpu_id)
            self.reward_models.append(reward_model)
            self.gpu_ids.append(rm_gpu_id)
            self.rm_tokenizers.append(rm_tokenizer)

        
    def forward(self, encoded_prompt_response):
        results = []
        with torch.no_grad():
            for i in range(len(self.peft_path_list)):
                # reward_tensors = [self.reward_models[i](x['input_ids'].to(self.gpu_ids[i])).logits[0] for x in encoded_prompt_response] 
                # results.append(torch.concat(reward_tensors).view(-1, 1))
                # print("len(self.peft_path_list)",len(self.peft_path_list))
                # print("encoded_prompt_response", encoded_prompt_response.shape)
                
                reward_tensors = self.reward_models[i](encoded_prompt_response.to(self.gpu_ids[i])).logits[:]
                # print(reward_tensors.shape, reward_tensors)
                results.append(reward_tensors)
        if self.ensemble_method == 'avg':
            reward_tensors = torch.concat(results, dim=-1).mean(dim=-1)
        elif self.ensemble_method == 'min':
            reward_tensors = torch.concat(results, dim=-1).min(dim=-1)[0]
        else:
            raise NotImplementedError
        return reward_tensors

class RMBoost():
    def __init__(self):
  
        self.booster = None
        self.peft_model = None
        self.rm_tokenizer = None
        self.rm_tokenizers = None
        self.gpu_id = None
        self.num_adapters = 0
        
    def load_reward_models(self, script_args, gpu_id, is_main):
        
        peft_model, booster, num_adapters, rm_tokenizer, rm_gpu_id = load_boost_reward_model(script_args, gpu_id, is_main)
        self.booster = booster
        self.peft_model = peft_model
        self.rm_tokenizer = rm_tokenizer
        self.gpu_id = gpu_id
        self.num_adapters = num_adapters
        self.rm_tokenizers = [self.rm_tokenizer]
    def forward(self, encoded_prompt_response):
        lora_rewards = []
        with torch.no_grad():
            for i in range(self.num_adapters):
                reward_tensors = [self.peft_model(x['input_ids'].to(self.gpu_id), x['attention_mask'].to(self.gpu_id), active_head=i)[-1].squeeze()  for x in encoded_prompt_response] 
                lora_rewards.append(torch.stack(reward_tensors, dim=0))
                
        boost_featuers = torch.stack(lora_rewards, dim=0).T     
        reward_tensors = self.booster.inplace_predict(boost_featuers.float().cpu().numpy())
        print("reward_tensors", reward_tensors)
        return torch.tensor(reward_tensors)
    