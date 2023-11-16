# quantizations
A collection of quantization recipes for various large models including Llama-2-70B, QWen-14B, Baichuan-2-13B, and more.




## Install
First install the requirements
```bash
conda create -n quantization python=3.9 -y
conda activate quantization
pip install -r requirements.txt
```
Then install auto-gptq from my fork
```bash
git clone https://github.com/ranchlai/AutoGPTQ.git
cd AutoGPTQ
python setup.py build
pip install -e .
```


## Usage
Quantize a model with the following command:
```
export CUDA_VISIBLE_DEVICES=0
python ../../quantize.py \
--model_name  Llama-2-70b-chat-hf \
--data data.json \
--bits 4 \
--output_folder Llama-2-70b-chat-gptq-4bit-128g \
--max_samples 1024 \
--group_size 128 \
--block_name_to_quantize "model.layers"
```

## Quantized models
| Model | #Params | #bits| Download |
| --- | --- | --- | --- |
| Llama-2-70B-chat | 70B | 4bits | [link](https://huggingface.co/ranchlai/Llama-2-70b-chat-gptq-4bit-128g) |
| CodeLlama | 34B | 4bits | [link](https://huggingface.co/ranchlai/CodeLlama-34b-Instruct-gptq-4bit) |
| chatglm3-6B | 6B | 4bits | [link](https://huggingface.co/ranchlai/chatglm3-6B-gptq-4bit) |
| Qwen-14B-Chat | 14B | 4bits | [link](https://huggingface.co/ranchlai/Qwen-14B-Chat-gptq-4bit-128g) |
| Baichuan2-13B-chat | 13B | 4bits | [link](https://huggingface.co/ranchlai/Baichuan2-13B-Chat-gptq-4bit-32g) |


## How to use the quantized models
The quantized models can be used in the same way as the original models. For example, the following code shows how to use the quantized chatglm3-6B model.
```python
from transformers import AutoTokenizer, AutoModel

model_name_or_path = "chatglm3-6B-gptq-4bit-32g"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="cuda:0")
model = model.eval()
response, history = model.chat(tokenizer, "北京秋天有什么好玩的景点", history=history)
print(response)
```
