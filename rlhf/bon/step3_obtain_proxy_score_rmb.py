from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import os
import sys
import torch
import numpy as np
import pandas as pd
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import argparse
from load_datasets import build_datasets_inference, prepare_data_loader
from utils import create_output_directory, adversarial_perturb
from collections import defaultdict
import os, glob, json, math, numpy as np, torch, xgboost as xgb
from pathlib import Path
from safetensors.torch import load_file

# Add the `./reward_models` path to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reward_models')))
from grm_utils import load_model_withhead, model_withhead_forward, AutoModelForCausalLMWithMultiValueHead



@dataclass
class ScriptArguments:
    per_device_batch_size: Optional[int] = field(default=64, metadata={"help": "The batch size per device during evaluation."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "The maximum sequence length."})
    data_path: Optional[str] = field(default="./step3_generate_samples/generated_samples_unified", metadata={"help": "Path to the data file."})
    model_type: Optional[str] = field(default="grm", metadata={'help': "use 'grm', 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    base_model: Optional[str] = field(default="google/gemma-2b-it", metadata={"help": "Path to the pre-trained model."})
    peft_name: Optional[str] = field(default="./step2_train_proxy_reward_model/gemma-2b-it", metadata={"help": "PEFT model name or directory if using PEFT."})
    save_path: Optional[str] = field(default='./step4_obtain_proxy_score/gemma-2b-it', metadata={"help": "Directory to save results."})
    # Only for GRM
    layer_type: Optional[str] = field(default='linear') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    debug: Optional[bool] = field(default=False)
    # only for lora-boost
    attn_implementation: str = field(default="flash_attention_2")
    tree_method: str = field(default="gpu_hist")
    layer_type: Optional[str] = field(default='mlp') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)
    num_adapters: Optional[int] = field(default=3)
    adapter_glob: str = field(default="./**/adapter_*",
                              metadata={"help": "Glob that resolves to adapter_* folders"})

    booster_path: str = field(default="multi_lora_booster.xgb")

    perturb: Optional[bool] = field(default=False)
    flips_per_seq: Optional[int] = field(default=1)
    
def parse_args() -> ScriptArguments:
    parser = argparse.ArgumentParser(description="Set parameters for model training & evaluation.")
    for field_name, field_def in ScriptArguments.__dataclass_fields__.items():
        parser.add_argument(
            f"--{field_name}",
            type=type(field_def.default),
            default=field_def.default,
            help=field_def.metadata.get("help", "")
        )
    args = parser.parse_args()
    return ScriptArguments(**vars(args))


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

def _set_active_adapter(wrapped, name: str):
    if hasattr(wrapped, "module"):
        wrapped.module.set_adapter(name)
    else:
        wrapped.set_adapter(name)



# Main execution logic
def obtain_proxy_score():
    # Parse arguments
    script_args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.model_type)
    
    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process

    # 4.2  – tokenizer & base LM
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.pad_token = tokenizer.eos_token

    model_params = {
        'vhead_layer_type': script_args.layer_type,
        'vhead_num_neurons': 1024,
        'vhead_num_layers': script_args.num_layers,
    }

    adapter_root = Path(script_args.adapter_glob or ".")
    n_adapters = len(list(adapter_root.glob("adapter_*")))
    print("num heads", n_adapters)
    base = AutoModelForCausalLMWithMultiValueHead.from_pretrained(
        script_args.base_model, device_map=device, 
        torch_dtype=torch.bfloat16,
        num_value_heads=n_adapters,
        **model_params,
    )

    base.pretrained_model.resize_token_embeddings(len(tokenizer))
    # print_trainable_parameters(base)
    base.config.pad_token_id = tokenizer.pad_token_id
    

    resume_dir = script_args.adapter_glob
    model = load_adapters_and_head(base, resume_dir, device)
    print("Loaded adapters:", list(model.peft_config.keys()))
    print("params loaded:", len([k for k,_ in model.named_parameters() if "lora_A.adapter_0" in k]))

    n_adapters = len(model.peft_config)
    if is_main: print(f"✓ loaded {n_adapters} LoRA adapters")

    booster = xgb.Booster()
    booster.load_model(script_args.booster_path)
    if is_main: print(f"✓ XGBoost restored from {script_args.booster_path}")
    model.eval()
    



    # Prepare dataset and DataLoader
    dataset = build_datasets_inference(script_args.data_path, tokenizer, split='test', max_length=script_args.max_length)
    if script_args.debug:
        dataset = dataset.select(range(0,5000))
    print('Size of Dataset: %s'%(len(dataset)))
        
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.local_process_index, shuffle=False)
    data_loader = prepare_data_loader(dataset, tokenizer, script_args.per_device_batch_size, sampler=sampler, collate_fn_type='custom')
    # data_loader = accelerator.prepare(data_loader)

    # Load model
    # if script_args.model_type == 'grm':
    #     model = load_model_withhead(script_args.base_model, script_args.peft_name, tokenizer, device=accelerator.local_process_index, layer_type=script_args.layer_type, num_layers=script_args.num_layers)
    # elif script_args.model_type in ['bt', 'margin', 'labelsmooth', 'pos_reg']:
    #     model = AutoModelForSequenceClassification.from_pretrained(script_args.base_model, num_labels=1, device_map=accelerator.local_process_index, torch_dtype=torch.bfloat16)
    #     # model.resize_token_embeddings(len(tokenizer))
    #     # model.config.pad_token_id = tokenizer.pad_token_id
    #     if os.path.exists(script_args.peft_name):
    #         model = PeftModel.from_pretrained(model, script_args.peft_name)
    #     if hasattr(model, 'merge_and_unload'):
    #         model = model.merge_and_unload()


    # Run evaluation and gather results
    full_prompts, full_rewards, full_source_ids, full_id_ids = [], [], [], []
    pbar = tqdm(total=len(data_loader) * script_args.per_device_batch_size // accelerator.num_processes)
    device = accelerator.local_process_index
    

    with torch.no_grad():
        for batch in data_loader:
            if script_args.perturb  :
                pert_ids, pert_mask = adversarial_perturb(
                    batch["input_ids"],
                    batch["attention_mask"],
                    tokenizer=tokenizer,
                    device=device,
                    max_length=script_args.max_length,
                    flips_per_seq=script_args.flips_per_seq,
                    base_seed="42",
                )
            else:
                pert_ids, pert_mask = batch["input_ids"],batch["attention_mask"]

            pos_list =  []                        # each element shape = (B,)
            for a in range(n_adapters):
                adapter_name = f"adapter_{a}"
                _set_active_adapter(model, adapter_name)
                # print(batch)
                r_pos = model(pert_ids.to(device), attention_mask=pert_mask.to(device), active_head=a)[-1].squeeze()

                #r_pos = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), active_head=a)[-1].squeeze()
                pos_list.append(r_pos)

            # shape: (B, N) after stack & transpose
            pos_mat = torch.stack(pos_list, dim=0).T           # chosen features
            feats_pos = pos_mat.float().cpu().numpy()          # (B, N)
            p_pos = booster.inplace_predict(feats_pos)         # probability y=1 :contentReference[oaicite:3]{index=3}

            # reshape to (B,) for easy comparison
            p_pos = torch.tensor(p_pos, device=device)
            
            full_rewards.extend(p_pos)
            #full_prompts.extend(batch['input_ids'])
            full_prompts.extend(pert_ids)
            full_source_ids.extend(batch['source'])
            full_id_ids.extend(batch['id'])
            pbar.update(1)
            
    full_prompts = [x.rstrip(tokenizer.pad_token) for x in tokenizer.batch_decode(full_prompts)]
    full_rewards = [float(x) for x in full_rewards]
    # full_source_ids = full_source_ids
    # full_id_ids = full_id_ids
    
    accelerator.wait_for_everyone()
    # Gather results from all processes
    all_prompts = accelerator.gather_for_metrics(full_prompts)
    all_rewards = accelerator.gather_for_metrics(full_rewards)
    all_source_ids = accelerator.gather_for_metrics(full_source_ids)
    all_id_ids = accelerator.gather_for_metrics(full_id_ids)

    id_to_count = defaultdict(int)
    all_group_index = []
    for sid in all_id_ids:
        all_group_index.append(id_to_count[sid])  # 0-based index within each id group
        id_to_count[sid] += 1
        
    if accelerator.is_main_process:
        all_results = {
            'prompts': all_prompts,
            'rewards': all_rewards,
            'source_ids': all_source_ids,
            'id_ids': all_id_ids,
            'idx_in_id_group': all_group_index,  # <- new column

        }
        df = pd.DataFrame(all_results)
        df.to_csv(os.path.join(output_dir, 'proxy_score.csv'), index=False)


if __name__ == "__main__":
    obtain_proxy_score()
