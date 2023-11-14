import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.gptq import GPTQQuantizer
import json
import torch
from loguru import logger
import random
import time


def main(args):
    random.seed(args.seed)

    files = args.data.split(",")
    data = []
    for file in files:
        with open(file, "r") as f:
            data += json.load(f)

    examples = [d["text"] for d in data]

    logger.info(f"loaded {len(examples)} examples")
    logger.info(f"shuffling {len(examples)} examples")
    random.shuffle(examples)
    logger.info(f"will use {args.max_samples} examples")
    examples = examples[: args.max_samples]

    # print 10 examples
    for i in range(10):
        print(examples[i])
        print("=======================")

    print("Loading model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.torch_dtype == "float16" else torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    print("Quantizing model")

    # print model device and dtype at every 20 layers
    for i, (name, param) in enumerate(model.named_parameters()):
        if i % 20 == 0:
            print(name, param.device, param.dtype)

    quantizer = GPTQQuantizer(
        bits=args.bits,
        dataset=examples,
        block_name_to_quantize=args.block_name_to_quantize,
        model_seqlen=2048,
        group_size=args.group_size,
        damp_percent=0.1,
        desc_act=False,
        sym=True,
        use_cuda_fp16=True,
        batch_size=1,
    )

    logger.info("Quantizing model...it will take about 6 hours")
    t = time.time()
    quantized_model = quantizer.quantize_model(model, tokenizer)
    logger.info(f"Quantizing model...done in {time.time()/3600 - t/3600} hours")
    logger.info("Saving model")
    quantized_model.save_pretrained(args.output_folder)
    # save the tokenizer
    tokenizer.save_pretrained(args.output_folder)
    logger.info("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a GPT model")
    parser.add_argument(
        "--model_name", type=str, help="Path to the pre-trained model directory"
    )
    parser.add_argument("--data", type=str, help="Path to the JSON data file")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Number of bits for quantization (default: 4)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="llama2-70B-gptq-4bit",
        help="Path to save the quantized model (default: llama2-70B-gptq-4bit)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2048,
        help="Maximum number of samples (default: 2048)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code (default: False)",
    )
    parser.add_argument("--group_size", type=int, default=1, help="Group size")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--block_name_to_quantize",
        type=str,
        default="transformer.h",
        help="Block name to quantize (default: transformer.h)",
    )  # "model.layers"
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        help="Torch dtype (default: float16)",
    )

    args = parser.parse_args()
    main(args)
