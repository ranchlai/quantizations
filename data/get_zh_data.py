from datasets import load_dataset
import json

from loguru import logger


def check_if_text_is_cn(text: str):
    """Check if text is in chinese"""
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return True
    return False


if __name__ == "__main__":
    dataset = load_dataset("c-s-ale/alpaca-gpt4-data-zh")

    # write to json
    # dataset["train"]["text"]
    data = []
    train = dataset["train"]

    for i in range(len(train["instruction"])):
        input = "\n\nHuman: " + train[i]["instruction"] + train[i]["input"]
        output = "\n\nAssistant: " + train[i]["output"]
        text = input + output
        data.append({"text": text})

    logger.info(f"writting {len(data)} to train_zh.json")
    with open("train_zh.json", "w") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
