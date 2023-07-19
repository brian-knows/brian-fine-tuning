import numpy as np
from trl import SFTTrainer
import torch
import json
import pandas as pd

from services.dataloader import DataLoader
from model.model.load_model import Model
from model.tokenizer.tokenizer import Tokenizer
from model.lora.lora_config import BrianLoraConfig
from model.trainer.trainer import Trainer

print("Starting Brian Fine-Tuning script.")
dataloader = DataLoader()
tokenizer = Tokenizer()
lora_config = BrianLoraConfig()
trainer = Trainer()
load_model = Model()
print("Classes loaded.")

print("Loading data..")
#loading data
unified_json = { "train": [], "test": [] }
for prompt_type in ['approve', 'balance', 'borrow', 'bridge', 'swap', 'total_supply', 'transfer']:
    with open(f"./data/custom_prompts/{prompt_type}.json", 'r') as prompts:
        prompts = json.load(prompts)
        unified_json["train"] += prompts
        unified_json["test"] += prompts
print("Data loaded.")

print("Concatenating data..")
concatenated_data = [f"{item['prompt']} </s> {item['completion']}" for item in unified_json["train"]]
print("Data concatenated.")

#train, test = dataloader.load_dataset_from_local(path="", split=True, test_size=0.3)
dataset = dataloader.create_hf_dataset(input_json=unified_json)
print("Dataset created.")

#loading tokenizer
tokenizer = tokenizer.load_tokenizer()
print("Tokenizer loaded.")

# tokenize dataset
print("Tokenizing dataset..")
max_length = 1024
tokenized_data = tokenizer.batch_encode_plus(
    concatenated_data,
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt',
    return_token_type_ids=False,
)
print("Dataset tokenized.")

print("Creating new dataset..")
dataset = dataloader.create_hf_dataset(input_json={
    "input_ids": tokenized_data["input_ids"],
    "attention_mask": tokenized_data["attention_mask"],
    "labels": tokenized_data["input_ids"],  # Since it's a T5-like model, labels are the same as inputs
})
print("New dataset created.")

# Calculate input_lengths and target_lengths from the tokenized dataset
input_lengths = [len(x) for x in dataset["input_ids"]]
target_lengths = [len(x) for x in dataset["labels"]]

# Calculate max_source_length and max_target_length
max_source_length = int(np.percentile(input_lengths, 85))
max_target_length = int(np.percentile(target_lengths, 90))

#trainer
print("Creating trainer model..")
trainer_model = trainer.get_trainer_model(tokenizer=tokenizer, model=lora_config.configure(), dataset=dataset)
print("Trainer model created.")

#launch job
trainer_model.train()

trainer_model.model.save_pretrained(trainer.output_dir)
tokenizer.save_pretrained(trainer.output_dir)