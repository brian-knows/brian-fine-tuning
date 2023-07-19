from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForSeq2SeqLM

class BrianLoraConfig:

    def __init__(self):
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.lora_r = 16
        self.model_id = "philschmid/flan-t5-xxl-sharded-fp16"
        # self.model_id = "google/flan-t5-base"

    def configure(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, load_in_8bit=False)
        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "v"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, self.lora_config)
        return model
