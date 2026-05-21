import json
import pathlib

import datasets
import fire
import torch.cuda
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def batch_to_dicts_zip(batch):
    keys = batch.keys()
    return [dict(zip(keys, values)) for values in zip(*batch.values())]


def main(input_path: str = "./generated",
         output_path: str = "./scored",
         batch_size: int = 32,
         cqf_model_name_or_path: str = "HuggingFaceTB/fineweb-edu-classifier",
         use_batch_writing: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cqf_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(cqf_model_name_or_path).to(device).eval()

    if pathlib.Path(input_path).is_file():
        print(f"Loading data from a single file: {input_path}")
        assert input_path.endswith(".jsonl"), "Handling only jsonl files."
        dataset = datasets.load_dataset("json", data_files={"train": input_path}, split="train")
    else:
        dataset = datasets.load_dataset(input_path, split="train")

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if str(output_path).endswith(".jsonl") else (
        output_path.mkdir(parents=True, exist_ok=True))

    with torch.no_grad():
        shard_idx = 0
        counter = 0
        for data in dataset.iter(batch_size//2):
            # bsz // 2 as we classify both raw + wikipedia-style text
            text = data["wiki_text"] + data["wiki_text_no_formatting"]
            inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            scores = outputs.logits.squeeze(-1).tolist()

            data["wiki_score"] = scores[:len(scores) // 2]
            data["wiki_int_score"] = [int(round(max(0, min(s, 5)))) for s in data["wiki_score"]]
            data["wiki_text_no_formatting_score"] = scores[len(scores) // 2:]
            data["wiki_text_no_formatting_int_score"] = [int(round(max(0, min(s, 5)))) for s in data["wiki_text_no_formatting_score"]]

            # Hotfix: Overcome json serialization problem
            data["date"] = [str(d) for d in data["date"]]

            with (output_path / f"{shard_idx}.jsonl").open("a") if use_batch_writing else output_path.open("a") as f:
                for example in batch_to_dicts_zip(data):
                    json.dump(example, f)
                    f.write("\n")
                    counter += 1

            if counter > 1000:
                shard_idx += 1
                counter = 0


if __name__ == '__main__':
    fire.Fire(main)
