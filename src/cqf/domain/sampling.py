import os

import datasets
import fire

DOMAINS = ['Arts_and_Entertainment', 'People_and_Society', 'Health', 'Shopping', 'Sports', 'Food_and_Drink',
           'Online_Communities', 'Jobs_and_Education', 'News', 'Games', 'Home_and_Garden', 'Beauty_and_Fitness',
           'Sensitive_Subjects', 'Computers_and_Electronics', 'Hobbies_and_Leisure', 'Pets_and_Animals',
           'Books_and_Literature', 'Business_and_Industrial', 'Internet_and_Telecom', 'Science', 'Finance',
           'Travel_and_Transportation', 'Real_Estate', 'Law_and_Government', 'Autos_and_Vehicles', 'Adult']


def main(input_path: str,
         output_path: str,
         sample_size: int = 20_000,
         num_proc=16):
    dataset = datasets.load_dataset("json", data_dir=input_path, split="train")
    for domain in DOMAINS:
        domain_dataset = dataset.filter(lambda x: x["domain"] == domain, num_proc=num_proc)
        domain_dataset = domain_dataset.filter(lambda x: len(x["text"].split()) < 10_000, num_proc=16)
        domain_dataset = domain_dataset.shuffle(seed=4892).select(range(sample_size))
        domain_dataset.to_json(os.path.join(output_path, f"{domain}.jsonl"))


if __name__ == '__main__':
    fire.Fire(main)
