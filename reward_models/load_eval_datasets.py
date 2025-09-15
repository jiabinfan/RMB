from tqdm import tqdm
import numpy as np
from datasets import load_dataset, concatenate_datasets

def build_unified_eval_dataset(data_path, tokenizer, split='val', size=None):
    try:
        ds = load_dataset(data_path, 'all', split=split)
    except:
        ds = load_dataset(data_path, split=split)
    # drop examples where A and B have same rating
    ds = ds.filter(lambda ex: ex['conv_A_rating'] != ex['conv_B_rating'], num_proc=30)

    if size is not None:
        ds = ds.select(range(size))

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
        'openai/webgpt_comparisons': 15,
    }

    def formatting_func(example):
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
        # print("valid:", tokenizer.decode(tokens_c["input_ids"],
        #                                     skip_special_tokens=True))
        return {
            'input_ids':              tokens_c['input_ids'],
            'attention_mask_chosen':  tokens_c['attention_mask'],
            'input_ids_rejected':     tokens_r['input_ids'],
            'attention_mask_rejected':tokens_r['attention_mask'],
            'source_id':              example['source_id'],
        }

    ds = ds.map(
        formatting_func,
        batched=False,
        num_proc=10,
        load_from_cache_file=False,  # ensure we rerun even if a cache exists
    )

    # drop everything except our four arrays + source_id
    keep = {
        'input_ids',
        'attention_mask_chosen',
        'input_ids_rejected',
        'attention_mask_rejected',
        'source_id',
    }
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    # filter out sequences that are too long
    ds = ds.filter(
        lambda ex: len(ex['input_ids']) <= tokenizer.model_max_length
                   and len(ex['input_ids_rejected']) <= tokenizer.model_max_length,
        num_proc=10,
    )

    # finally, tell HF to convert lists → torch.Tensor at training time
    ds.set_format(type='torch')
    return ds


def build_ood_eval_dataset(data_path, tokenizer, split='test', size=None):
    # load and merge sub‐datasets
    if 'HuggingFaceH4/hhh_alignment' in data_path:
        parts = ['harmless', 'helpful', 'honest', 'other']
        ds = None
        for i, key in enumerate(parts):
            part = load_dataset(data_path, key, split='test')
            part = part.add_column('source_id', [i] * len(part))
            ds = part if ds is None else concatenate_datasets([ds, part])
    elif 'lmsys/mt_bench_human_judgments' in data_path:
        raw = load_dataset('lmsys/mt_bench_human_judgments')
        a = raw['human'].add_column('source_id', [0] * len(raw['human']))
        b = raw['gpt4_pair'].add_column('source_id', [1] * len(raw['gpt4_pair']))
        ds = concatenate_datasets([a, b])
    else:
        ds = load_dataset(data_path, split=split)

    if size is not None:
        ds = ds.select(range(size))

    def formatting_func(example):
        # same kwargs as above
        example.setdefault('source_id', -1)
        # reconstruct chosen/rejected pairs per dataset format...
        if 'HuggingFaceH4/hhh_alignment' in data_path:
            # your existing logic here, then:
            prompt = example['input']
            # 2) unpack the nested targets dict
            tgt     = example['targets']
            choices = tgt['choices']
            labels  = tgt['labels']
            # 3) determine which choice is the "chosen" (label=1) vs. "rejected"
            if labels[0] == 1:
                chosen_text, rejected_text = choices[0], choices[1]
            else:
                chosen_text, rejected_text = choices[1], choices[0]

            # 4) build two-turn conversations
            conv_ch = [
                {'role': 'user',      'content': prompt},
                {'role': 'assistant', 'content': chosen_text}
            ]
            conv_rj = [
                {'role': 'user',      'content': prompt},
                {'role': 'assistant', 'content': rejected_text}
            ]

            # 5) render via your chat template
            p_ch = tokenizer.apply_chat_template(conv_ch, tokenize=False)
            p_rj = tokenizer.apply_chat_template(conv_rj, tokenize=False)
        elif 'lmsys/mt_bench_human_judgments' in data_path:
            if example['winner'] == 'model_a':
                conv_ch, conv_rj = example['conversation_a'], example['conversation_b']
            else:
                conv_ch, conv_rj = example['conversation_b'], example['conversation_a']
            p_ch = tokenizer.apply_chat_template(conv_ch, tokenize=False)
            p_rj= tokenizer.apply_chat_template(conv_rj, tokenize=False)
        elif 'Skywork' in data_path:

            prompt = example['chosen'][0]['content']
            chosen_messages = example['chosen']
            rejected_messages = example['rejected']

            p_ch = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            p_rj = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        
        else:
            # generic pairwise
            conv_ch = [
                {'role':'user',      'content': example['prompt']},
                {'role':'assistant', 'content': example[f"response_{example['better_response_id']}"]},
            ]
            conv_rj = [
                {'role':'user',      'content': example['prompt']},
                {'role':'assistant', 'content': example[f"response_{1-example['better_response_id']}"]},
            ]
            p_ch = tokenizer.apply_chat_template(conv_ch, tokenize=False)
            p_rj= tokenizer.apply_chat_template(conv_rj, tokenize=False)

        # tokenize → lists
        tokenizer.model_max_length = 1024
        tokens_c = tokenizer.encode_plus(
            p_ch,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )
        tokens_r = tokenizer.encode_plus(
            p_rj,
            padding='max_length',
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None,
        )

        return {
            'input_ids':               tokens_c['input_ids'],
            'attention_mask_chosen':   tokens_c['attention_mask'],
            'input_ids_rejected':      tokens_r['input_ids'],
            'attention_mask_rejected': tokens_r['attention_mask'],
            'source_id':               example['source_id'],
        }

    ds = ds.map(
        formatting_func,
        batched=False,
        num_proc=1,
        load_from_cache_file=False,
    )

    keep = {
        'input_ids',
        'attention_mask_chosen',
        'input_ids_rejected',
        'attention_mask_rejected',
        'source_id',
    }
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds = ds.filter(
        lambda ex: len(ex['input_ids']) <= tokenizer.model_max_length
                   and len(ex['input_ids_rejected']) <= tokenizer.model_max_length,
        num_proc=1,
    )
    ds.set_format(type='torch')
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

        ds = ds.filter(lambda ex: (_norm(ex["subset"]) in wanted_norm), num_proc=10)

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
            "input_ids":                tokens_c["input_ids"],
            "attention_mask_chosen":    tokens_c["attention_mask"],
            "input_ids_rejected":       tokens_r["input_ids"],
            "attention_mask_rejected":  tokens_r["attention_mask"],
            "source_id":                subset_to_id.get(example["subset"], -1),
        }

    ds = ds.map(
        formatting_func,
        batched=False,
        num_proc=10,
        load_from_cache_file=False,
    )

    # 5) Keep the same columns as your other builders
    keep = {
        "input_ids",
        "attention_mask_chosen",
        "input_ids_rejected",
        "attention_mask_rejected",
        "source_id",
    }
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    # 6) Length filter
    ds = ds.filter(
        lambda ex: len(ex["input_ids"]) <= tokenizer.model_max_length
                   and len(ex["input_ids_rejected"]) <= tokenizer.model_max_length,
        num_proc=10,
    )

    ds.set_format(type="torch")
    return ds

def load_eval_dataset(task, tokenizer, size=None):
    print("11119999999999999999999999999999", task)
    tl = task.lower()
    if 'unified' in task.lower():
        size = None
        # return build_unified_eval_dataset(
        #     'llm-blender/Unified-Feedback', tokenizer, split='val', size=size
        # )
        return build_unified_eval_dataset(
            task, tokenizer, split='val', size=size
        )
    elif 'hhh' in task:
        return build_ood_eval_dataset(
            'HuggingFaceH4/hhh_alignment', tokenizer, split='test', size=size
        )
    elif 'mt' in task:
        return build_ood_eval_dataset(
            'lmsys/mt_bench_human_judgments', tokenizer, split='test', size=size
        )
    elif 'work' in task:
        dataset = build_ood_eval_dataset(
            'Skywork/Skywork-Reward-Preference-80K-v0.2', tokenizer, split='train', size=size)
        dataset_split = dataset.train_test_split(test_size=0.005)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
        print("9999999999999999999999999999")
        print(len(eval_dataset))
        return eval_dataset

    elif ("rewardbench" in tl) or ("rb-v1" in tl) or ("rb1" in tl) or ("rbv1" in tl) or ("rb" in tl):
        # Accept suffixes like:
        #   "rewardbench:chat", "rewardbench chat-hard", "rb-v1-safety",
        #   "rb1_reasoning", "rewardbench overall", etc.
        mode = "overall"
        for cand in ("chat-hard", "chat", "safety", "reasoning", "overall"):
            if cand in tl:
                mode = cand
                break
        # Prefer the filtered split by default
        return build_rewardbench_v1_dataset(tokenizer, category=mode, split="filtered", size=size)

    else:
        raise NotImplementedError(f"Unknown eval task: {task}")
