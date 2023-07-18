from trl import SFTTrainer
import torch
import json
import pandas as pd

from services.dataloader import DataLoader
from model.model.load_model import Model
from model.tokenizer.tokenizer import Tokenizer
from model.lora.lora_config import LoraConfig
from model.trainer.trainer import Trainer

dataloader = DataLoader()
tokenizer = Tokenizer()
lora_config = LoraConfig()
trainer = Trainer()
load_model = Model()

#todo: unify custom_promots and real_promots in unique dict

#loading data
unified_json = {}
for prompt_type in ['approve']:
    with open(f"./data/custom_prompts/{prompt_type}.json", 'r') as prompts:
        prompts = json.load(prompts)
        unified_json[prompt_type] = prompt_type
#train, test = dataloader.load_dataset_from_local(path="", split=True, test_size=0.3)
dataset = dataloader.create_hf_dataset(input_json=unified_json)

#loading tokenizer
tokenizer = tokenizer.load_tokenizer()

#trainer
max_seq_length = 512

trainer = SFTTrainer(
    model=load_model.load_model(),
    train_dataset=dataset,
    peft_config=lora_config.configure(),
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=trainer.training_arguments(),
)

# questo per ottimizzare torch sotto
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

#launch job
trainer.train()




