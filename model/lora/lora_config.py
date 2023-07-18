from peft import LoraConfig

class LoraConfig:

    def __init__(self):
        self.lora_alpha = 16
        self.lora_dropout = 0.1
        self.lora_r = 64

    def configure(self):
        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM", #todo: cange task ecc ecc
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ]
        )