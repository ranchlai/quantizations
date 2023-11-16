import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from transformers import GPTQConfig, BitsAndBytesConfig

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Generate text using a specified model.')

# Add the 'model_name' argument
parser.add_argument('--model_name', "-m", type=str, help='Name of the pretrained model to use')

# Parse the command-line arguments
args = parser.parse_args()

# Use the 'model_name' argument value
model_name = args.model_name

quantization_config = GPTQConfig(
    bits=4,
    disable_exllama=False,
    use_cuda_fp16=True,
    dataset=None,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map={"": "cuda:0"},
                                             torch_dtype=torch.bfloat16, 
                                             trust_remote_code=True)

model.generation_config = GenerationConfig.from_pretrained(model_name)
messages = []
messages.append({"role": "user", "content": "给一个四天北京的旅游行程安排"})
response = model.chat(tokenizer, messages)
print(response)


messages= [{"role": "user", "content": "把这句话改写10句：你好，我是鹏鹏，来自平安产险公司的AI团队的一个虚拟助手。有什么你需要了解的吗？"}]
response = model.chat(tokenizer, messages)
print(response)
while True:
    messages= [{"role": "user", "content": input("user: ")}]
    response = model.chat(tokenizer, messages)
    print(response)
    