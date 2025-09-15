import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
import torch
from accelerate import Accelerator
import evaluate
import numpy as np
import os
from collections import OrderedDict
import torch
import torch.nn as nn
accuracy = evaluate.load('accuracy')


import hashlib, random
from typing import Tuple

def _stable_int_seed_from_tensor(t: torch.Tensor, prefix_seed: str = "42") -> int:
    # Hash the raw ids (on CPU) + a fixed prefix to get a repeatable seed per batch
    lst = t.detach().cpu().numpy().tolist()
    h = hashlib.sha256((prefix_seed + str(lst)).encode()).hexdigest()
    return int(h[:8], 16)

_QWERTY = {
    'q':'was', 'w':'qes', 'e':'wrd', 'r':'etf', 't':'ryg', 'y':'tuh', 'u':'yij', 'i':'uoj', 'o':'ipk', 'p':'ol',
    'a':'qwsz', 's':'awedxz', 'd':'serfcx', 'f':'drtgcv', 'g':'ftyhbv', 'h':'gyujnb', 'j':'huikmn', 'k':'jiolm', 'l':'kop',
    'z':'asx', 'x':'zsdc', 'c':'xdfv', 'v':'cfgb', 'b':'vghn', 'n':'bhjm', 'm':'njk'
}

def _apply_keyboard_typos(text: str, rng: random.Random, rate: float = 0.03, max_changes: int = 1) -> str:
    """DeepWordBug-like single-character neighbor substitutions (deterministic via rng)."""
    chars = list(text)
    idxs = [i for i,c in enumerate(chars) if c.isalpha()]
    rng.shuffle(idxs)
    changes = 0
    for i in idxs:
        if changes >= max_changes: break
        if rng.random() <= rate:
            c = chars[i]
            low = c.lower()
            if low in _QWERTY and _QWERTY[low]:
                repl = rng.choice(_QWERTY[low])
                chars[i] = repl.upper() if c.isupper() else repl
                changes += 1
    return ''.join(chars)

def adversarial_perturb(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    device,
    max_length: int,
    method: str = "typo",     # "hotflip" or "typo"
    flips_per_seq: int = 1,
    base_seed: str = "42",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministically perturb a batch.
    - For non-GRM: HotFlip-like one gradient step projected back to nearest vocab token.
    - For GRM (or when inputs_embeds path is unavailable): keyboard-typo attack on decoded text.
    """
    # Make local copies on device (don’t mutate the dataloader’s tensors)
    dev = device if isinstance(device, torch.device) else torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    ids = input_ids.clone().to(dev)
    mask = attention_mask.clone().to(dev)

    # Build a stable per-batch seed
    seed = _stable_int_seed_from_tensor(ids, base_seed)
    rng = random.Random(seed)

    # Special tokens to avoid flipping into/out of
    special_id_list = [getattr(tokenizer, name) for name in (
        "pad_token_id","eos_token_id","bos_token_id","cls_token_id","sep_token_id"
    )]
    special_ids = torch.tensor([i for i in special_id_list if i is not None], device=dev, dtype=torch.long)

    # --- Typo attack (GRM-friendly & deterministic) ---
    # Decode -> apply typos -> re-tokenize to the same shape
    ids_cpu = ids.detach().cpu().tolist()
    new_ids, new_mask = [], []
    for row in ids_cpu:
        # strip pads before decoding to avoid trailing pads in text
        trimmed = [t for t in row if (special_id_list[0] is None or t != tokenizer.pad_token_id)]
        text = tokenizer.decode(trimmed, skip_special_tokens=True)
        pert_text = _apply_keyboard_typos(text, rng, rate=0.03, max_changes=max(1, flips_per_seq))
        enc = tokenizer(
            pert_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        new_ids.append(enc["input_ids"][0])
        new_mask.append(enc["attention_mask"][0])

    ids_out = torch.stack(new_ids).to(dev)
    mask_out = torch.stack(new_mask).to(dev)
    return ids_out, mask_out


def is_lora_model(model):
    for key in model.state_dict().keys():
        if 'lora' in key:
            return True
    return False

def get_trainable_weights(model):
    save_dict = OrderedDict()
    state_dict = model.state_dict()
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'pretrained_model.' in key:
                key = key.replace('pretrained_model.', '')
            save_dict[key] = state_dict[key]
    return save_dict

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    predictions = np.argmax(predictions, axis=1)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


def grm_compute_metrics(eval_pred):
    rewards = eval_pred.label_ids
    reward_accuracy = (rewards[:, 0] > rewards[:, 1]).mean()
    
    predictions = eval_pred.predictions
    accuracy = (predictions[:, 0] > predictions[:, 1]).mean()
    return {
        'dpo_accuracy': accuracy,
        'reward_accuracy': reward_accuracy
    }


def print_trainable_parameters(model, print_trainable_name=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if print_trainable_name:
                print(name)
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def freeze_trainable_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def create_output_directory(log_dir: str, wandb_name: str):
    output_path = os.path.join(log_dir, wandb_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path


# Function to save results as Parquet files
def save_results_in_parquet_splits(results, num_splits, save_path, mode='test'):
    results_df = pd.DataFrame(results)
    dataset_with_results = Dataset.from_pandas(results_df)
    
    split_size = len(dataset_with_results) // num_splits
    for i in range(num_splits):
        start = i * split_size
        end = start + split_size if i < num_splits - 1 else len(dataset_with_results)
        split = dataset_with_results.select(range(start, end))
        file_path = f"{save_path}/{mode}-0000{i}-of-0000{num_splits}.parquet"
        split.to_parquet(file_path)


# Define the KL equation
def kl_equation(N):
    return np.log(N) - (N - 1) / N


# Calculate and filter KL values
def calculate_kl_values(N_values, kl_min=0, kl_max=5):
    kl_values = [kl_equation(N) for N in N_values]
    results = pd.DataFrame({'N': N_values, 'kl': kl_values})
    return results[(results['kl'] >= kl_min) & (results['kl'] <= kl_max)]


# Define function to get highest rewards within N items per group
def get_highest_within_n(group, n):
    return group.head(n).nlargest(1, 'rewards')