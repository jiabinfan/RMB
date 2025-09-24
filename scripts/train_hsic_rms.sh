devices=5
n_gpu=1
dataset_name='llm-blender/Unified-Feedback'
dataset_mode='40k'
base_model='google/gemma-2b-it'
wandb_name=""
log_dir='../save_reward_models'
main_process_port=9993
diversity_type='hsic'

learning_rate=3e-4 
lora_r=32
lora_alpha=64
max_length=1024
num_train_epochs=200
gradient_accumulation_steps=64
layer_type='mlp' 
lr_scheduler_type='cosine'

cd ../reward_models
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port} hsic_rms_train.py \
    --base_model ${base_model}  --wandb_name ${wandb_name}   --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --max_length ${max_length} \
    --num_adapters 3\
    --output_tag "0.1hsic" \
    --use_lora True \
    --random_seed ${lora_r} \
    --diversity_type ${diversity_type} \
    --lora_r ${lora_r} --lora_alpha ${lora_alpha} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate}  \
    --dataset ${dataset_name} --dataset_mode ${dataset_mode} --lr_scheduler_type ${lr_scheduler_type}