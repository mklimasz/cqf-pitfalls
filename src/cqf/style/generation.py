import json
import pathlib

import datasets
import fire
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPHRASE_PROMPT = """You are a Wikipedia-style rephraser. Your objective is to rephrase the following web page to imitate a Wikipedia article.

Web page:
```{web_page}```

Follow the rules below during rephrasing:
- Focus on containing all the facts from the document, even if they are not essential.
- Do not include new facts, concepts and overall new content.
- Keep the exact dates, locations, names and other entities.
- Outcome should differ only in terms of style and formatting.
- The output document should have a similar number of tokens (with a maximum 10% margin)."""


def apply_template(web_page, tokenizer):
    messages = [
        {
            "role": "user",
            "content": REPHRASE_PROMPT.format(web_page=web_page)
        }
    ]

    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def batch_to_dicts_zip(batch):
    keys = batch.keys()
    return [dict(zip(keys, values)) for values in zip(*batch.values())]


def main(model_name_or_path: str = "Qwen/Qwen2.5-72B-Instruct",
         input_path: str = "HuggingFaceFW/fineweb",
         subset_name: str = "sample-10BT",
         use_batch_writing: bool = True,
         subset_size: int = 100_000,
         output_path: str = "./generated"):
    model = LLM(model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count())

    sampling_params = SamplingParams(max_tokens=16384)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if pathlib.Path(input_path).is_file():
        print(f"Loading data from a single file: {input_path}")
        assert input_path.endswith(".jsonl"), "Handling only jsonl files."
        dataset = datasets.load_dataset("json", data_files={"train": input_path}, split="train")
    else:
        dataset = datasets.load_dataset(input_path, name=subset_name, split="train")
    dataset = dataset.filter(lambda x: len(x["text"].split()) < 10_000, num_proc=16).select(range(subset_size))

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if str(output_path).endswith(".jsonl") else (
        output_path.mkdir(parents=True, exist_ok=True))

    for batch_idx, data in enumerate(dataset.iter(1000)):
        messages = [apply_template(text, tokenizer) for text in data["text"]]
        outputs = model.generate(messages, sampling_params)
        wiki_text = [output.outputs[0].text for output in outputs]
        data["wiki_text"] = wiki_text

        with ((output_path / f"{batch_idx}.jsonl").open("w") if use_batch_writing else output_path.open("a")) as f:
            for example in batch_to_dicts_zip(data):
                json.dump(example, f)
                f.write("\n")


if __name__ == '__main__':
    fire.Fire(main)
