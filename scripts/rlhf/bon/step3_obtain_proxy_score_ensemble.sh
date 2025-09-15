devices=0,1,2,6
n_gpu=4
main_process_port=9094

reward_peft_path1=''
reward_peft_path2=''
reward_peft_path3=''

cd ../../../
# Fill the peft path

CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/step3_obtain_proxy_score_ensemble.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "rlhf/bon/step2_generate_samples/generated_samples_unified" \
    --model_type "avg" \
    --reward_base_model "google/gemma-2b-it" \
    --reward_peft_path ${reward_peft_path1},${reward_peft_path2},${reward_peft_path3} \
    --save_path "rlhf/bon/step3_obtain_proxy_score/gemma-2b-it" \
    --layer_type "linear" \
    --num_layers 1 \
    # --debug False \



    

