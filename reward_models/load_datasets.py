import numpy as np
import os
import torch
import torch.nn as nn
from datasets import load_dataset, concatenate_datasets
import random
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
        ds = ds.select(range(0, size))

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

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
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
    ds = ds.filter(lambda example: example['conv_A_rating'] != example['conv_B_rating'], num_proc=30)

    if len(mode) and split=='train':
        if mode == '40k' or mode == '40K':
            ds = ds.select(range(0, len(ds), 20)) 
        elif mode == '400k' or mode == '400K':
            ds = ds.select(range(0, len(ds), 2)) 
    if size is not None:
        ds = ds.select(range(0, size))

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
        if random.random() < 0.2 and split == "train":
            if example['conv_A_rating'] < example['conv_B_rating']:
                chosen_messages = example['conv_A']
                rejected_messages = example['conv_B']
                margin = example['conv_A_rating'] - example['conv_B_rating']
            else:
                chosen_messages = example['conv_B']
                rejected_messages = example['conv_A']
                margin = example['conv_B_rating'] - example['conv_A_rating']            
        
        else:
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
        

    ds = ds.map(formatting_func, batched=False, num_proc=10)
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
        ds = ds.select(range(0, size))

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

    ds = ds.map(formatting_func, batched=False, num_proc=10) 
    ds.set_format(type="torch")
    return ds


def load_train_eval_dataset(data_path, tokenizer, size=None, mode='', model_name='', random_seed=42):
    if 'nified' in data_path:
        # mode is only used for loading training data
        ss= None
        train_dataset = build_dataset_UF(data_path, tokenizer, split='train', size=ss, mode=mode, model_name=model_name, random_seed=random_seed) 
        # ss = 512
        # train_dataset = build_dataset_UF(data_path, tokenizer, split='val', size=ss, mode=mode, model_name=model_name) 
    elif 'Skywork' in data_path:
        dataset = build_dataset_SK(data_path, tokenizer, split='train', size=size, model_name=model_name)
        dataset_split = dataset.train_test_split(test_size=0.3) # 0.005
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']
    else:
        dataset = build_dataset(data_path, tokenizer, split='train', size=size, model_name=model_name) 
        dataset_split = dataset.train_test_split(test_size=0.01)
        train_dataset, eval_dataset = dataset_split['train'], dataset_split['test']

    return train_dataset, eval_dataset