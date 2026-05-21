import json
import pathlib
import re

import datasets
import fire
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

with (pathlib.Path(__file__).parent / "prompt.txt").open() as f:
    EDU_PROMPT = f.read()

EXTRACT_SCORE_PATTERN = r"(?:Educational score:|a .*score of|it scored|rate .*a)\s*([0-5])"


def apply_template(extract, tokenizer):
    prompt = EDU_PROMPT.replace("<EXAMPLE>", extract)
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def batch_to_dicts_zip(batch):
    keys = batch.keys()
    return [dict(zip(keys, values)) for values in zip(*batch.values())]


def main(input_path: str = "./scored",
         output_path: str = "./labelled",
         model_name_or_path: str = "meta-llama/Llama-3.1-70B-Instruct"):
    model = LLM(model_name_or_path,
                tensor_parallel_size=torch.cuda.device_count())

    sampling_params = SamplingParams(max_tokens=512)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    dataset = datasets.load_dataset(input_path, split="train")
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset.iter(1000)):
            messages = [apply_template(text[:1500], tokenizer) for text in data["text"]]
            outputs = model.generate(messages, sampling_params)

            scores = []
            for output in outputs:
                text = output.outputs[0].text
                match = re.search(EXTRACT_SCORE_PATTERN, text)
                if not match:
                    scores.append(None)
                    print("Unparsable score, skipping.")
                    print(text)
                else:
                    score = int(match.group(1))
                    assert 0 <= score <= 5, (score, text)
                    scores.append(score)

            data["llm_score"] = scores
            # Hotfix: Overcome json serialization problem
            data["date"] = [str(d) for d in data["date"]]
            with (output_path / f"{batch_idx}.jsonl").open("w") as f:
                for example in batch_to_dicts_zip(data):
                    json.dump(example, f)
                    f.write("\n")


if __name__ == '__main__':
    fire.Fire(main)
