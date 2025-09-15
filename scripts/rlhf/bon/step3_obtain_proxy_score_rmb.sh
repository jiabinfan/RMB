devices=4,5,6
n_gpu=3
main_process_port=9194

adapter_glob=''
booster_out=''

cd ../../../

# For baselines
CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score_rmb.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/bon/step2_generate_samples/generated_samples_unified" \
    --model_type "rmb" \
    --base_model "google/gemma-2b-it" \
    --peft_name "rlhf/bon/save_reward_models/.." \
    --save_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it" \
    --adapter_glob "${adapter_glob}" \
    --booster_path ${booster_out} \
    # --debug True



    

