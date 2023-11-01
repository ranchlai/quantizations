# set -e pipefail
export CUDA_VISIBLE_DEVICES=0
model=Qwen-14B/

quant_dir=Qwen-14B-gptq-4bits-2g

python ../../quantize_by_autogptq.py \
    --pretrained_model_dir $model \
    --quantized_model_dir $quant_dir \
    --data train_v13a_qwen.json \
    --bits 4 \
    --num_samples 2048 \
    --group_size 2 \
    --trust_remote_code \

copy.sh $quant_dir