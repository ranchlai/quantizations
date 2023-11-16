from transformers import AutoTokenizer, AutoModel

model_name_or_path = "chatglm3-6B-gptq-4bit-32g/"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="cuda:0")
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "北京秋天有什么好玩的景点", history=history)
print(response)
