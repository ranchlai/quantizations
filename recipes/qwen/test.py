from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import GenerationConfig
from transformers import GPTQConfig, BitsAndBytesConfig
import torch
# from tokenization_qwen import QWenTokenizer

from loguru import logger
import json
import tqdm

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)  # auto-gptq logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", "-m", type=str)
parser.add_argument("--device", "-d", type=str, default="cuda:0")

args = parser.parse_args()

print(args.model_name_or_path)
# logger.info("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)



generation_config = GenerationConfig(
    num_beams=1,
    do_sample=True,
    temperature=0.9,
    repetition_penalty=1.0,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=128,
)


# inference with model.generate
text = "This is a story about a "
text = tokenizer(text, return_tensors="pt")


quantization_config = GPTQConfig(
            bits=4,
            disable_exllama=False,
            use_cuda_fp16=False,
            dataset=None,
        )
# now load the base model and test it
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                             trust_remote_code=True, 
                                             device_map={"": args.device}, 
                                                torch_dtype=torch.float16)

                                                # quantization_config=quantization_config, 

# show model dtype, device, etc.
for name, param in model.named_parameters():
    print(name, param.dtype, param.device, param.shape)
model = torch.compile(model)
response = model.generate(
    input_ids=text["input_ids"].to(args.device),
    attention_mask=text["attention_mask"].to(args.device),
    generation_config=generation_config,
)
response = tokenizer.decode(response[0])
print(response.strip())



