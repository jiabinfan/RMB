devices=2,3
n_gpu=2
main_process_port=9394


cd ../../../
# Fill the peft path

# For GRM
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score.py \
    --per_device_batch_size 16 \
    --max_length 1024 \
    --data_path "rlhf/bon/step2_generate_samples/generated_samples_unified" \
    --model_type "grm" \
    --base_model "google/gemma-2b-it" \
    --peft_name "rlhf/bon/save_reward_models/gemma-2b-it_GRM_len1024_lora32_1e-05_dataunified_sampled_gold_score/logs/checkpoint-2444" \
    --save_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it" \
    --layer_type "linear" \
    --num_layers 1 \
    # --debug False \

# # # For baselines
# CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
#     rlhf/bon/step3_obtain_proxy_score.py \
#     --per_device_batch_size 64 \
#     --max_length 1024 \
#     --data_path "rlhf/bon/step2_generate_samples/generated_samples_unified" \
#     --model_type "bt" \
#     --base_model "google/gemma-2b-it" \
#     --peft_name "rlhf/bon/save_reward_models/20noise/gemma-2b-it_BT_RM_len3000_lora32_5e-06_dataunified_sampled_gold_score/logs1/checkpoint-305" \
#     --save_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it" \
#     # --debug False \



    

