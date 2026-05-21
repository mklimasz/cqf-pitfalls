import json
import pathlib

import datasets
import fire
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import PyTorchModelHubMixin


def batch_to_dicts_zip(batch):
    keys = batch.keys()
    return [dict(zip(keys, values)) for values in zip(*batch.values())]


class DomainClassifier(nn.Module, PyTorchModelHubMixin):
    # https://huggingface.co/nvidia/domain-classifier#how-to-use-in-transformers
    def __init__(self, config):
        super(DomainClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


def main(output_path: str = "./domain_scored",
         batch_size: int = 32,
         domain_classifier_name_or_path: str = "nvidia/domain-classifier"):
    config = AutoConfig.from_pretrained(domain_classifier_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(domain_classifier_name_or_path)
    model = DomainClassifier.from_pretrained(domain_classifier_name_or_path).eval()
    dataset = datasets.load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

    distributed_state = Accelerator()
    device = distributed_state.device
    model = model.to(device)

    if distributed_state.is_main_process:
        output_path = pathlib.Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        shard_idx = 0
        counter = 0
        for data in dataset.iter(batch_size * distributed_state.num_processes):
            input_data = []
            with distributed_state.split_between_processes(data) as batch:
                text = batch["text"]
                inputs = tokenizer(text, return_tensors="pt", padding="longest", truncation=True)
                outputs = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
                predicted_classes = torch.argmax(outputs, dim=1)
                input_data.extend(batch_to_dicts_zip(batch))

            distributed_state.wait_for_everyone()

            predicted_classes = gather_object(predicted_classes)
            input_data = gather_object(input_data)

            if distributed_state.is_main_process:
                predicted_domains = [config.id2label[class_idx.item()] for class_idx in predicted_classes]
                with (output_path / f"{shard_idx}.jsonl").open("a") as f:
                    for example, predicted_domain in zip(input_data, predicted_domains):
                        example["domain"] = predicted_domain
                        json.dump(example, f)
                        f.write("\n")
                        counter += 1

                if counter > 10000:
                    shard_idx += 1
                    counter = 0

    if distributed_state.is_main_process:
        print(f">>> Domain classification finished")


if __name__ == '__main__':
    fire.Fire(main)
