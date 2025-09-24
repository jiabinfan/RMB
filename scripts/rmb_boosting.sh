#!/usr/bin/env bash


devices=0,1,2,4,5,6
n_gpu=6
dataset_name='llm-blender/Unified-Feedback'
eval_dataset_name='llm-blender/Unified-Feedback'

dataset_mode='40k'
base_model='google/gemma-2b-it'

adapter_glob=''
booster_out=''
main_process_port=12357   # any free port

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} \
accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} \
    run_booster_rmb.py \
    --base_model ${base_model} \
    --adapter_glob "${adapter_glob}" \
    --dataset ${dataset_name} \
    --eval_dataset ${eval_dataset_name} \
    --dataset_mode ${dataset_mode} \
    --batch_size 64 \
    --max_length 1024 \
    --tree_method gpu_hist \
    --num_round 1000 \
    --early_stopping 1000 \
    --booster_out ${booster_out}
