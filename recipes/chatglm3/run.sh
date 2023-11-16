model=chatglm3-6b
# model=/root/data/models/CodeLlama-34b-hf/
quant_dir=chatglm3-6B-gptq-4bit-32g

python ../../quantize.py \
--model_name $model \
--data "../../data/train.json,../../data/train_zh.json" \
--bits 4 \
--output_folder $quant_dir \
--max_samples 2048 \
--block_name_to_quantize transformer.encoder.layers \
--trust_remote_code \
--group_size 32 \


