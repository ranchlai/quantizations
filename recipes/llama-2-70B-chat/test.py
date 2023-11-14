from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import GPTQConfig, BitsAndBytesConfig


quantization_config = GPTQConfig(
    bits=4,
    disable_exllama=False,
    use_cuda_fp16=True,
    dataset=None,
)

model_name_or_path = "Llama-2-70b-chat-gptq-4bit-128g"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map={"": "cuda:0"},
                                             trust_remote_code=False,
                                             revision="main", 
                                             quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "写一个程序，将一个列表中的数字按升序排列。"
prompt_template=f'''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]

'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
# output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
# print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)


print(pipe(prompt_template)[0]['generated_text'])

# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
#  <<SYS>>
# <</SYS>>
while True:
    prompt = input(">>> ")
    prompt_template=f'''[INST]
{prompt}[/INST]
'''
    print(pipe(prompt_template)[0]['generated_text'])
    