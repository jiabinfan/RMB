import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
from datasets import DatasetDict, load_dataset, load_from_disk, concatenate_datasets
import random


def _dataset_num_proc(dataset, cap=8):
    """Bound Arrow workers by the CPUs allocated to the Slurm task."""
    try:
        allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    except ValueError:
        allocated = 1
    return max(1, min(cap, allocated, len(dataset)))


def build_rrm_local_dataset(data_path, tokenizer, size=None):
    """Load a materialized RRM corpus with hard and neutral soft-label pairs."""
    path = Path(data_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_dir():
        ds = load_from_disk(str(path))
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
    else:
        ds = load_dataset("parquet", data_files=str(path), split="train")

    required = {"chosen", "rejected", "preference_target"}
    missing = required.difference(ds.column_names)
    if missing:
        raise ValueError(
            f"RRM dataset {path} is missing required columns: {sorted(missing)}"
        )
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def formatting_func(example):
        kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": tokenizer.max_length,
            "return_tensors": "pt",
        }
        chosen_text = tokenizer.apply_chat_template(
            example["chosen"], tokenize=False
        )
        rejected_text = tokenizer.apply_chat_template(
            example["rejected"], tokenize=False
        )
        tokens_chosen = tokenizer.encode_plus(chosen_text, **kwargs)
        tokens_rejected = tokenizer.encode_plus(rejected_text, **kwargs)
        return {
            "input_ids_chosen": tokens_chosen["input_ids"][0],
            "attention_mask_chosen": tokens_chosen["attention_mask"][0],
            "input_ids_rejected": tokens_rejected["input_ids"][0],
            "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            "preference_target": float(example["preference_target"]),
        }

    ds = ds.map(
        formatting_func, batched=False, num_proc=_dataset_num_proc(ds)
    )
    keep = {
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
        "preference_target",
    }
    ds = ds.remove_columns([col for col in ds.column_names if col not in keep])
    ds.set_format(type="torch")
    return ds
# for vanilla chosen and reject style dataset, such as dendrydong/preference_700K
def build_dataset(data_path, tokenizer, split='train', size=None, model_name=''):
    if 'lmsys/mt_bench_human_judgments' in data_path:
        raw = load_dataset('lmsys/mt_bench_human_judgments')
        a = raw['human'].add_column('source_id', [0] * len(raw['human']))
        b = raw['gpt4_pair'].add_column('source_id', [1] * len(raw['gpt4_pair']))
        ds = concatenate_datasets([a, b])
    else:
        ds = load_dataset(data_path, split=split)

    # size = 16
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        if 'lmsys/mt_bench_human_judgments' in data_path:
            if example['winner'] == 'model_a':
                chosen_messages, rejected_messages = example['conversation_a'], example['conversation_b']
            else:
                chosen_messages, rejected_messages  = example['conversation_b'], example['conversation_a']
        else:
            chosen_messages = example['chosen']
            rejected_messages = example['rejected']
        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)

        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = example['chosen'][:-1]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(
        formatting_func, batched=False, num_proc=_dataset_num_proc(ds)
    )
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for UnifiedFeedback
def build_dataset_UF(data_path, tokenizer, split='train', size=None, mode='', model_name='', random_seed = 42):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    # if mode == "none":
    #     ds = load_dataset(data_path, split=split)
    # else:
    #     ds = load_dataset(data_path, 'all', split=split)
    # filter data with the same rating
    ds = ds.filter(
        lambda example: example['conv_A_rating'] != example['conv_B_rating'],
        num_proc=_dataset_num_proc(ds),
    )

    if len(mode) and split=='train':
        if mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20))
        elif mode.lower().replace("_", "-") == "40k-heldout":
            # Stage 1's 40K split uses every twentieth post-filter index
            # (0, 20, 40, ...). Select an odd-offset stride here so the booster
            # sees the same source distribution with zero Stage-1 overlap.
            ds = ds.select(range(1, len(ds), 20))
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2))
        elif mode == '50k' or mode == '50K':
            # shuffle then take first 100 000
            print(f"Shuffling with seed = {random_seed}")   # optional, for debugging
            ds = ds.shuffle(seed=random_seed)
            n = min(50_000, len(ds))             # in case ds has <100k
            ds = ds.select(range(n))
        elif mode == '200k' or mode == '200K':
            # shuffle then take first 100 000
            print(f"Shuffling with seed = {random_seed}")   # optional, for debugging
            ds = ds.shuffle(seed=random_seed)
            n = min(200_000, len(ds))             # in case ds has <100k
            ds = ds.select(range(n))
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    source_dict = {
        'argilla/ultrafeedback-binarized-preferences-cleaned': 0,
        'Anthropic/hh-rlhf': 1,
        'flan_v2_flan2021': 2,
        'ultrachat': 3,
        'evol_instruct': 4,
        'false_qa': 5,
        'Dahoas/synthetic-instruct-gptj-pairwise': 6,
        'flan_v2_cot': 7,
        'flan_v2_p3': 8,
        'truthful_qa': 9,
        'lmsys/chatbot_arena_conversations': 10,
        'openai/summarize_from_feedback(comparisons)': 11,
        'sharegpt': 12,
        'flan_v2_niv2': 13,
        'berkeley-nest/Nectar': 14,
        'openai/webgpt_comparisons': 15,}
    def formatting_func0(example):
        example['source_id'] = source_dict[example['source']]
        # pick better chain
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen, rejected = example['conv_A'], example['conv_B']
        else:
            chosen, rejected = example['conv_B'], example['conv_A']

        # special prompt for summarization sources
        if 'summarize' in example['source']:
            prefix = 'Generate one-sentence summary for the following post: '
            chosen[0]['content']  = prefix + chosen[0]['content'].strip()
            rejected[0]['content'] = prefix + rejected[0]['content'].strip()

        # render chat history → single prompt
        p_ch = tokenizer.apply_chat_template(chosen,  tokenize=False)
        p_rj = tokenizer.apply_chat_template(rejected, tokenize=False)

        # tokenize → lists
        tokenizer.model_max_length = 1024
        tokens_c = tokenizer.encode_plus(
            p_ch,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,   # <-- yields Python lists
        )
        tokens_r = tokenizer.encode_plus(
            p_rj,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        # print("train:", tokenizer.decode(tokens_c["input_ids"],
        #                                     skip_special_tokens=True))
        return {
            'input_ids_chosen':              tokens_c['input_ids'],
            'attention_mask_chosen':  tokens_c['attention_mask'],
            'input_ids_rejected':     tokens_r['input_ids'],
            'attention_mask_rejected':tokens_r['attention_mask'],
        }


    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        # if random.random() < -0.1 and split == "6666":
        #     if example['conv_A_rating'] < example['conv_B_rating']:
        #         chosen_messages = example['conv_A']
        #         rejected_messages = example['conv_B']
        #         margin = example['conv_A_rating'] - example['conv_B_rating']
        #     else:
        #         chosen_messages = example['conv_B']
        #         rejected_messages = example['conv_A']
        #         margin = example['conv_B_rating'] - example['conv_A_rating']

        # else:
        if example['conv_A_rating'] > example['conv_B_rating']:
            chosen_messages = example['conv_A']
            rejected_messages = example['conv_B']
            margin = example['conv_A_rating'] - example['conv_B_rating']
        else:
            chosen_messages = example['conv_B']
            rejected_messages = example['conv_A']
            margin = example['conv_B_rating'] - example['conv_A_rating']

        if 'summarize' in example['source']:
            chosen_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + chosen_messages[0]['content'].strip()
            rejected_messages[0]['content'] = 'Generate one-sentence summary for the following post: ' + rejected_messages[0]['content'].strip()

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo training
            prompt = [example['conv_A'][0]]
            prompt_template = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected,
                # "margin": margin, # GRM does not need this
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "margin": margin,
            }

    ds = ds.map(
        formatting_func, batched=False, num_proc=_dataset_num_proc(ds)
    )
    # ds = ds.filter(lambda x: len(x["input_ids_chosen"]) <= script_args.max_length and len(x["input_ids_rejected"]) <= script_args.max_length, num_proc=30)
    remove_columns = []
    for col in ds.column_names:
        if 'input' not in col and 'attention' not in col and 'margin' not in col and 'label' not in col:
            remove_columns.append(col)
    ds = ds.remove_columns(remove_columns)

    ds.set_format(type="torch")
    return ds


# for Skywork Reward Preference 80K
def build_dataset_SK(data_path, tokenizer, split='train', size=None, model_name=''):
    ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    def formatting_func(example):
        kwargs = {"padding": True, "truncation": True, "max_length": tokenizer.max_length, "return_tensors": "pt"}
        prompt = example['chosen'][0]['content']

        chosen_messages = example['chosen']
        rejected_messages = example['rejected']

        prompt_plus_chosen_response = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        prompt_plus_rejected_response = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
        tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
        if 'GRM' in model_name:
            # add label mask for sft and dpo
            prompt_template = tokenizer.apply_chat_template([{"content": prompt, "role": "user" }], tokenize=False, add_generation_prompt=True)
            tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)['input_ids'][0]
            label_chosen = tokens_chosen["input_ids"][0].clone()
            label_chosen[:len(tokens_prompt)] = -100
            label_rejected = tokens_rejected["input_ids"][0].clone()
            label_rejected[:len(tokens_prompt)] = -100
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                "label_chosen": label_chosen,  'label_rejected': label_rejected
            }
        else:
            return {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0],
            }

    ds = ds.map(
        formatting_func, batched=False, num_proc=_dataset_num_proc(ds)
    )
    ds.set_format(type="torch")
    return ds



RB_V1_CATEGORIES = {
    "chat": {
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        # Some pages say "mt-bench-medium"; the rows often use "mt-bench-med".
        "mt-bench-medium",
        "mt-bench-med",
    },
    "chat-hard": {
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    },
    "safety": {
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        # The subset in the dataset card is "do not answer"; keep a robust alias too.
        "do not answer",
        "donotanswer",
    },
    "reasoning": {
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    },
}

def build_rewardbench_v1_dataset(tokenizer, category="overall", split="filtered", size=None):
    """
    Build a RewardBench v1 eval dataset with the same shape/columns as your other builders.
    category: one of {"chat","chat-hard","safety","reasoning","overall"}.
              "overall" keeps all subsets.
    split: "filtered" (recommended) or "raw" (as exposed on HF).
    """
    # 1) Load
    try:
        ds = load_dataset("allenai/reward-bench", split=split)
    except Exception:
        # Fallback in case the environment exposes a single split
        ds_all = load_dataset("allenai/reward-bench")
        split_name = next(iter(ds_all))  # e.g., "filtered" or "raw"
        ds = ds_all[split_name]

    # 2) Filter to category (if not overall)
    if category and category.lower() != "overall":
        wanted = set(RB_V1_CATEGORIES[category.lower()])
        # Be tolerant to whitespace/underscore/hyphen oddities
        def _norm(x): return x.lower().replace("_", "-").strip()
        wanted_norm = {_norm(x) for x in wanted}

        ds = ds.filter(
            lambda ex: (_norm(ex["subset"]) in wanted_norm),
            num_proc=_dataset_num_proc(ds),
        )

    # Optional size cap
    if size is not None:
        ds = ds.select(range(min(size, len(ds))))

    # 3) Build subset→id map (stable within this run)
    #    Using the actual values present after filtering to avoid KeyErrors.
    present_subsets = sorted(set(ds["subset"]))
    subset_to_id = {name: i for i, name in enumerate(present_subsets)}

    # 4) Formatter: prompt + {chosen,rejected} → two-turn chats → tokenize
    def formatting_func(example):
        prompt = example["prompt"]
        chosen_text = example["chosen"]
        rejected_text = example["rejected"]

        conv_ch = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen_text},
        ]
        conv_rj = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected_text},
        ]

        p_ch = tokenizer.apply_chat_template(conv_ch, tokenize=False)
        p_rj = tokenizer.apply_chat_template(conv_rj, tokenize=False)

        tokenizer.model_max_length = 1024
        tokens_c = tokenizer.encode_plus(
            p_ch,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        tokens_r = tokenizer.encode_plus(
            p_rj,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )

        return {
            "input_ids_chosen":                tokens_c["input_ids"],
            "attention_mask_chosen":    tokens_c["attention_mask"],
            "input_ids_rejected":       tokens_r["input_ids"],
            "attention_mask_rejected":  tokens_r["attention_mask"],
            "source_id":                subset_to_id.get(example["subset"], -1),
        }

    ds = ds.map(
        formatting_func,
        batched=False,
        num_proc=_dataset_num_proc(ds),
        load_from_cache_file=False,
    )


    # 5) Keep the same columns as your other builders
    keep = {
        "input_ids_chosen",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
        "source_id",
    }
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    # 6) Length filter
    ds = ds.filter(
        lambda ex: len(ex["input_ids_chosen"]) <= tokenizer.model_max_length
                   and len(ex["input_ids_rejected"]) <= tokenizer.model_max_length,
        num_proc=_dataset_num_proc(ds),
    )

    ds.set_format(type="torch")
    return ds


def load_train_eval_dataset(
    data_path,
    tokenizer,
    size=None,
    mode="",
    model_name="",
    random_seed=42,
    load_eval=True,
    load_train=True,
):
    if not load_train and not load_eval:
        raise ValueError("At least one of load_train or load_eval must be true")

    # RRM is opt-in. Do not reinterpret arbitrary local paths used by legacy
    # experiments as an RRM soft-label corpus.
    local_path = Path(data_path).expanduser()
    rrm_marker = local_path.is_dir() and (
        local_path / "rrm_metadata.json"
    ).is_file()
    if mode.lower() == "rrm" or rrm_marker:
        if not local_path.exists():
            raise FileNotFoundError(
                f"RRM mode requires a local dataset path, got {data_path!r}"
            )
        dataset = build_rrm_local_dataset(local_path, tokenizer, size=size)
        return (
            dataset if load_train else None,
            dataset if load_eval else None,
        )

    tl = data_path.lower()

    if "unified" in tl:
        # mode is only used for loading training data
        # Honor the public ``size`` argument. The previous hardcoded None made
        # debug/smoke runs tokenize the entire Unified-Feedback corpus first.
        ss = size
        train_dataset = (
            build_dataset_UF(
                data_path,
                tokenizer,
                split="train",
                size=ss,
                mode=mode,
                model_name=model_name,
                random_seed=random_seed,
            )
            if load_train
            else None
        )
        eval_dataset = (
            build_dataset_UF(
                data_path,
                tokenizer,
                split="val",
                size=ss,
                model_name=model_name,
            )
            if load_eval
            else None
        )
    elif 'Skywork' in data_path:
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.3) # 0.005
        train_dataset = dataset_split['train'] if load_train else None
        eval_dataset = dataset_split['test'] if load_eval else None


    elif (
        ("rewardbench" in tl)
        or ("rb-v1" in tl)
        or ("rb1" in tl)
        or ("rbv1" in tl)
    ):
        # Accept suffixes like:
        #   "rewardbench:chat", "rewardbench chat-hard", "rb-v1-safety",
        #   "rb1_reasoning", "rewardbench overall", etc.
        mode = "overall"
        for cand in ("chat-hard", "chat", "safety", "reasoning", "overall"):
            if cand in tl:
                mode = cand
                break
        # Prefer the filtered split by default
        dataset = build_rewardbench_v1_dataset(
            tokenizer, category=mode, split="filtered", size=size
        )
        return (
            dataset if load_train else None,
            dataset if load_eval else None,
        )


    else:
        dataset = build_dataset(data_path, tokenizer, split='train', size=size, model_name=model_name)
        #dataset_split = dataset.train_test_split(test_size=0.01)
        #train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
        train_dataset = dataset if load_train else None
        eval_dataset = dataset if load_eval else None
    return train_dataset, eval_dataset
