
log_dir='rlhf/logs_ppo'
init_kl_coef=0.00
base_model_name="google/gemma-2b-it" # policy base model
reward_base_model="google/gemma-2b-it" 
dataset_path="rlhf/data/unified_20k" # set the train dataset path, refer to the BoN experiments
eval_dataset_path="rlhf/data/unified_1k" # set the eval dataset

eval_every=1

cd ../../../

# you need set the path

adapter_glob=''
booster_out=''

ensemble_method='avg'
wandb_name=""
CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port 7121 rlhf/ppo/ppo_rmb.py \
    --base_model_name ${base_model_name} \
    --reward_base_model ${reward_base_model} \
    --dataset_path ${dataset_path}\
    --eval_dataset_path ${eval_dataset_path}\
    --adapter_glob "${adapter_glob}" \
    --booster_path ${booster_out} \
    --init_kl_coef ${init_kl_coef}\
    --log_dir ${log_dir} \
    --wandb_name ${wandb_name} \
    --eval_every ${eval_every} \
    --ensemble_method ${ensemble_method} \
    --normalize_rewards False \
    --learning_rate 1e-5 \
    --mini_batch_size 1 \
    --eval_batch_size 128 \







   



