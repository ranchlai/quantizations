from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# from loguru import logger
import json
import tqdm

import random

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)  # auto-gptq logger

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--pretrained_model_dir", type=str, default="/mnt/data/models/Llama-2-7b-hf/"
)
parser.add_argument("--quantized_model_dir", type=str, default="./quantized")
parser.add_argument("--data", type=str, default="../data/processed/train_v1.json")
parser.add_argument("--bits", type=int, default=4)
parser.add_argument("--group_size", type=int, default=128)
parser.add_argument("--trust_remote_code", action="store_true")
parser.add_argument("--cache_examples_on_gpu", action="store_true")
parser.add_argument("--desc_act", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_samples", type=int, default=4096)



args = parser.parse_args()

print(args)

random.seed(args.seed)


# logger.info("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(
    args.pretrained_model_dir, use_fast=True, trust_remote_code=args.trust_remote_code
)


data = args.data
with open(data, "r") as f:
    data = json.load(f)

examples = [d["text"] for d in data]

logging.info(f"loaded {len(examples)} examples")
examples = random.sample(examples, args.num_samples)


tokenized_examples = []
for e in tqdm.tqdm(examples):
    tokenized_examples += [tokenizer(e, return_tensors="pt")]

# logger.info(f"loaded {len(tokenized_examples)} examples")

quantize_config = BaseQuantizeConfig(
    bits=args.bits,  # quantize model to 4-bit
    group_size=args.group_size,  # it is recommended to set the value to 128
    desc_act=args.desc_act,  # set to False can significantly speed up inference but the perplexity may slightly bad
)
# logger.info(f"quantize config: {quantize_config}")

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(
    args.pretrained_model_dir,
    quantize_config,
    device_map={"": "cuda:0"},
    trust_remote_code=args.trust_remote_code,
)
# print model parameters

for name, param in model.named_parameters():
    print(name, param.shape, param.device)


# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
# use mini-batch to quantize model
model.quantize(tokenized_examples, batch_size=1, cache_examples_on_gpu=args.cache_examples_on_gpu)
# # save quantized model
# model.save_quantized(args.quantized_model_dir)

# save quantized model using safetensors
# model.save_quantized(args.quantized_model_dir, use_safetensors=True)

# save the tokenizer and model using save_pretrained
model.save_pretrained(args.quantized_model_dir)
tokenizer.save_pretrained(args.quantized_model_dir)
print("saved model to", args.quantized_model_dir)
