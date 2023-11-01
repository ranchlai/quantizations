# set -e pipefail
# export CUDA_VISIBLE_DEVICES=0
model=Qwen-14B/
# model=/root/data/models/CodeLlama-34b-hf/

quant_dir=$1
if [ -z "$quant_dir" ]; then
    echo "quant_dir is empty"
    exit 1
fi
# python ../../../quantization/quantize.py \
#     --pretrained_model_dir $model \
#     --quantized_model_dir $quant_dir \
#     --data ../../../data/processed/train_v1.json \
#     --bits 3

# move all json from model dir to quant dir, expect for config.json

files="cache_autogptq_cuda_256.cpp
generation_config.json
tokenizer_config.json
configuration_qwen.py
cpp_kernels.py
qwen.tiktoken
qwen_generation_utils.py
tokenization_qwen.py"

echo files: $files
for file in $files; do
    cp $model/$file $quant_dir
    echo "copy $file to $quant_dir"
done
cp modeling_qwen.py $quant_dir/

# save  $quant_dir/config.json to bk if there is no quantization_config in the file
quant_config=$(cat $quant_dir/config.json | grep quantization_config)
if [ -z "$quant_config" ]; then
    cp $quant_dir/config.json $quant_dir/config.json.bk
    echo "backup config.json to config.json.bk"
fi

if [ -f "$quant_dir/config.json.bk" ]; then
    cp $quant_dir/config.json.bk $quant_dir/config.json
    echo "restore config.json.bk to config.json"
fi
cd $quant_dir
# find the *.bin model, link to pytorch_model.bin
model_bin=$(find . | grep bin)
echo "model_bin: $model_bin"
ln -s $model_bin pytorch_model.bin
cd ..

python process_config.py --config_file $quant_dir/config.json \
--quantize_config_file $quant_dir/quantize_config.json \
--config_file $quant_dir/config.json
