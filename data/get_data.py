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
    dataset = load_dataset("timdettmers/openassistant-guanaco")

    # write to json
    # dataset["train"]["text"]
    data = []
    train = dataset["train"]
    test = dataset["test"]
    for i in range(len(train["text"])):
        data.append(
            {
                "text": train[i]["text"]
                .replace("### Human", "\n\nHuman")
                .replace("### Assistant", "\n\nAssistant")
            }
        )

    logger.info(f"writting {len(data)} to train.json")
    with open("train.json", "w") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)

    data = []
    for i in range(len(test["text"])):
        data.append(
            {
                "text": test[i]["text"]
                .replace("### Human", "\n\nHuman")
                .replace("### Assistant", "\n\nAssistant")
            }
        )

    logger.info(f"writting {len(data)} to test.json")
    with open("test.json", "w") as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
