#!/bin/bash

step_now=$1
num_loops=$2


for ((i=step_now; i<num_loops; i++)); do
    if [ "$i" -gt 0 ]; then
        python Bohdi.py \
            --src_model_name_or_path "path of Qwen2.5-14B-Instruct" "path of Mistral-Small-24B-Instruct-2501" "path of phi-4" \
            --tgt_model_name_or_path "new target checkpoint path" \
            --tree_path "path of the knowledge tree" \
            --basic_tree_save_path "path of the basic tree" \
            --basic_tree_load_path "path of the basic tree" \
            --max_model_len 16384 \
            --target_save_path "path to save the updated target model" \
            --assigned_devices 0 1 2 3 4 5 6 7 \
            --temperatures 0.6 0.7 0.15 0.7 \
            --load_in_half bf16 \
            --meditation_steps 3 \
            --enlightenment_steps 3 \
            --batch_size_meditation_phase 30 \
            --batch_size_enlightenment_phase 60 \
            --step_now $i \
            --phase_id 'meditation' \
            --thr 0.2 \
            --window_size 20 
        
        accelerate launch --config_file=deepspeed_zero2.yaml --num_processes 8 Bohdi.py \
            --src_model_name_or_path "path of Qwen2.5-14B-Instruct" "path of Mistral-Small-24B-Instruct-2501" "path of phi-4" \
            --tgt_model_name_or_path "new target checkpoint path" \
            --tree_path "path of the knowledge tree" \
            --basic_tree_save_path "path of the basic tree" \
            --basic_tree_load_path "path of the basic tree" \
            --max_model_len 16384 \
            --target_save_path "path to save the updated target model" \
            --assigned_devices 0 1 2 3 4 5 6 7 \
            --temperatures 0.6 0.7 0.15 0.7 \
            --load_in_half bf16 \
            --meditation_steps 3 \
            --enlightenment_steps 3 \
            --batch_size_meditation_phase 30 \
            --batch_size_enlightenment_phase 60 \
            --step_now $i \
            --phase_id 'enlightenment' \
            --thr 0.2 \
            --window_size 20  
    
    else
        python Bohdi.py \
            --src_model_name_or_path "path of Qwen2.5-14B-Instruct" "path of Mistral-Small-24B-Instruct-2501" "path of phi-4" \
            --tgt_model_name_or_path "path of the pre-trained target model" \
            --tree_path "path of the knowledge tree" \
            --basic_tree_save_path "path of the basic tree" \
            --max_model_len 16384 \
            --target_save_path "path to save the updated target model" \
            --assigned_devices 0 1 2 3 4 5 6 7 \
            --temperatures 0.6 0.7 0.15 0.7 \
            --load_in_half bf16 \
            --meditation_steps 3 \
            --enlightenment_steps 3 \
            --batch_size_meditation_phase 30 \
            --batch_size_enlightenment_phase 60 \
            --step_now $i \
            --phase_id 'meditation' \
            --thr 0.2 \
            --window_size 20 
    
        accelerate launch --config_file=deepspeed_zero2.yaml --num_processes 8 Bohdi.py \
            --src_model_name_or_path "path of Qwen2.5-14B-Instruct" "path of Mistral-Small-24B-Instruct-2501" "path of phi-4" \
            --tgt_model_name_or_path "path of the pre-trained target model" \
            --tree_path "path of the knowledge tree" \
            --basic_tree_save_path "path of the basic tree" \
            --basic_tree_load_path "path of the basic tree" \
            --max_model_len 16384 \
            --target_save_path "path to save the updated target model" \
            --assigned_devices 0 1 2 3 4 5 6 7 \
            --temperatures 0.6 0.7 0.15 0.7 \
            --load_in_half bf16 \
            --meditation_steps 3 \
            --enlightenment_steps 3 \
            --batch_size_meditation_phase 30 \
            --batch_size_enlightenment_phase 60 \
            --step_now $i \
            --phase_id 'enlightenment' \
            --thr 0.2 \
            --window_size 20  
    fi
done


