devices=5,6
n_gpu=2
main_process_port=9594

cd ../../../


CUDA_VISIBLE_DEVICES=${devices} accelerate launch --num_processes ${n_gpu} --main_process_port ${main_process_port}  \
    rlhf/bon/obtain_ppo_gold_score.py \
    --per_device_batch_size 64 \
    --max_length 1024 \
    --data_path "" \
    --method "avg" \
    --model_path "Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback" \
    --save_path "rlhf/bon/step5_obtain_bon_gold_score/gemma-2b-it" \

