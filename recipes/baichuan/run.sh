export CUDA_VISIBLE_DEVICES=1
model=Baichuan2-13B-Chat
# model=/root/data/models/CodeLlama-34b-hf/
quant_dir=Baichuan2-13B-Chat-gptq-4bit-64g

python ../../quantize.py \
--model_name $model \
--data "../../data/train.json,../../data/train_zh.json" \
--bits 4 \
--output_folder $quant_dir \
--max_samples 4096 \
--block_name_to_quantize model.layers \
--trust_remote_code \
--group_size 64 \


