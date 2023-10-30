model=Baichuan2-13B-Chat
# model=/root/data/models/CodeLlama-34b-hf/
quant_dir=Baichuan2-13B-Chat-gptq-4bit

python ../../quantize.py \
--model_name $model \
--data "../../data/train.json,../../data/train_zh.json" \
--bits 4 \
--output_folder $quant_dir \
--max_samples 2048 \
--block_name_to_quantize model.layers \
--trust_remote_code \


