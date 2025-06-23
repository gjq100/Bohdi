#!/bin/bash
source /mnt/workspace/gaojunqi/anaconda3/etc/profile.d/conda.sh
source activate /mnt/workspace/gaojunqi/anaconda3/envs/opencompass
cd /mnt/workspace/gaojunqi/sunqingshuai/opencompass

export HF_ENDPOINT=https://hf-mirror.com

python run.py --datasets math_0shot_gen_393424 gsm8k_0shot_gen_a58960 mbpp_gen_830460 humaneval_gen_8e312c \
    --hf-path checkpoint path for evaluation \
    --tokenizer-path checkpoint path for evaluation \
    --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
    --model-kwargs trust_remote_code=True device_map='auto' \
    --generation-kwargs do_sample=False \
    --max-seq-len 2048 \
    --max-out-len 4096 \
    --batch-size 64 \
    --max-num-workers 8 \
    --max-workers-per-gpu 1 \
    -a vllm 

python run.py --datasets mmlu_gen_4d595a bbh_gen_4a31fa TheoremQA_5shot_gen_6f0af8 gpqa_gen_4baadb \
    --hf-path checkpoint path for evaluation  \
    --tokenizer-path checkpoint path for evaluation \
    --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
    --model-kwargs trust_remote_code=True device_map='auto' \
    --generation-kwargs do_sample=False \
    --max-seq-len 2048 \
    --max-out-len 1024 \
    --batch-size 64 \
    --max-num-workers 8 \
    --max-workers-per-gpu 1 \
    -a vllm 
