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
from peft import PeftModel
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
from rm_utils import load_reward_model, RMEnsemble

# Add the `./reward_models` path to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reward_models')))
from grm_utils import load_model_withhead, model_withhead_forward

from huggingface_hub import login
login(token="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")

@dataclass
class ScriptArguments:
    per_device_batch_size: Optional[int] = field(default=64, metadata={"help": "The batch size per device during evaluation."})
    max_length: Optional[int] = field(default=1024, metadata={"help": "The maximum sequence length."})
    data_path: Optional[str] = field(default="./step3_generate_samples/generated_samples_unified", metadata={"help": "Path to the data file."})
    model_type: Optional[str] = field(default="grm", metadata={'help': "use 'grm', 'bt', 'margin', 'labelsmooth', and 'pos_reg'."})
    reward_base_model: Optional[str] = field(default="google/gemma-2b-it", metadata={"help": "Path to the pre-trained model."})
    peft_name: Optional[str] = field(default="./step2_train_proxy_reward_model/gemma-2b-it", metadata={"help": "PEFT model name or directory if using PEFT."})
    save_path: Optional[str] = field(default='./step4_obtain_proxy_score/gemma-2b-it', metadata={"help": "Directory to save results."})
    ensemble_method: Optional[str] = field(default='avg')
    reward_peft_path: Optional[str] = field(default='', metadata={'help': 'Separate each path with a comma'})
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={'help': "use '' if you don't want attention acceleration or meet error here."})

    # Only for GRM
    layer_type: Optional[str] = field(default='linear') # mlp, linear
    num_layers: Optional[int] = field(default=1)
    debug: Optional[bool] = field(default=False)
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



# Main execution logic
def obtain_proxy_score():
    # Parse arguments
    script_args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    gpu_id= Accelerator().local_process_index 

    # Create output directory
    output_dir = create_output_directory(script_args.save_path, script_args.model_type)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.reward_base_model, use_fast=False, use_auth_token=True)
    tokenizer.model_max_length = script_args.max_length
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset and DataLoader
    dataset = build_datasets_inference(script_args.data_path, tokenizer, split='test', max_length=script_args.max_length)
    if script_args.debug:
        print(script_args.debug)
        dataset = dataset.select(range(0,5000))
    print('Size of Dataset: %s'%(len(dataset)))
        
    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.local_process_index, shuffle=False)
    data_loader = prepare_data_loader(dataset, tokenizer, script_args.per_device_batch_size, sampler=sampler, collate_fn_type='custom')
    # data_loader = accelerator.prepare(data_loader)

    # Load model
    if script_args.model_type == 'grm':
        print("gpu device", accelerator.local_process_index)
        model = load_model_withhead(script_args.reward_base_model, script_args.peft_name, tokenizer, device=accelerator.local_process_index, layer_type=script_args.layer_type, num_layers=script_args.num_layers)
    elif script_args.model_type in ['bt', 'margin', 'labelsmooth', 'pos_reg']:
        model = AutoModelForSequenceClassification.from_pretrained(script_args.reward_base_model, num_labels=1, device_map=accelerator.local_process_index, torch_dtype=torch.bfloat16)
        # model.resize_token_embeddings(len(tokenizer))
        # model.config.pad_token_id = tokenizer.pad_token_id
        if os.path.exists(script_args.peft_name):
            model = PeftModel.from_pretrained(model, script_args.peft_name)
        if hasattr(model, 'merge_and_unload'):
            model = model.merge_and_unload()
    else:
        reward_peft_model_paths = script_args.reward_peft_path.split(',')
        reward_models = RMEnsemble(script_args.ensemble_method, base_model_name=script_args.reward_base_model, peft_path_list=reward_peft_model_paths)
        reward_models.load_reward_models(script_args, gpu_id)


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
            if script_args.model_type == 'grm':
                reward_tensors = model_withhead_forward(model, batch["input_ids"], batch["attention_mask"], device, forward_type='reward') 
            elif script_args.model_type == 'avg' or "avg" in script_args.model_type:
                reward_tensors = reward_models.forward(pert_ids.to(device))
                #reward_tensors = reward_models.forward(batch["input_ids"])
            else:
                reward_tensors = model(batch["input_ids"].to(device))
            
            full_rewards.extend(reward_tensors)
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
