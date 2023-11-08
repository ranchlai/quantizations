export CUDA_VISIBLE_DEVICES=1
python ../../quantize.py \
--model_name  Llama-2-70b-chat-hf \
--data data.json \
--bits 4 \
--output_folder Llama-2-70b-chat-gptq-4bit-128g \
--max_samples 1024 \
--group_size 128 \
--block_name_to_quantize "model.layers"
