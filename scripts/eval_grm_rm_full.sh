
gpu="${GPU_ID:-0}"
port="${MAIN_PROCESS_PORT:-29500}"
max_length=1024
per_device_eval_batch_size=1
base_model="${BASE_MODEL:-google/gemma-2b-it}"
peft_name="${PEFT_NAME:-}"
layer_type='mlp' # linear
num_layers=1
log_dir='./eval_GRM'
save_all_data=False

cd ../rm_eval
for task in 'unified'  'hhh'  'mtbench' 
do 
    CUDA_VISIBLE_DEVICES=${gpu} accelerate launch --main_process_port ${port} eval_grm.py --base_model ${base_model} --per_device_eval_batch_size ${per_device_eval_batch_size} \
                                             --max_length ${max_length} --log_dir ${log_dir} --save_all_data ${save_all_data} \
                                              --task ${task} --layer_type ${layer_type} --num_layers ${num_layers} 

done


