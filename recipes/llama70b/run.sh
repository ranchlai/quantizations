export CUDA_VISIBLE_DEVICES=0

# if Llama-2-70b-hf  not exist
if [ ! -d "Llama-2-70b-hf" ]; then
  ln -s /workspace/llama-2-70b-hf Llama-2-70b-hf
fi

python ../../quantize.py \
--model_name  Llama-2-70b-hf \
--data data.json \
--bits 4 \
--output_folder llama2-70B-gptq-4bit-128g \
--max_samples 1024 \
--group_size 128 \
--block_name_to_quantize "model.layers"
