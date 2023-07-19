# todo: personalizzare con modello finale
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


class Model:
    def __init__(self):
        self.model_name = "ybelkada/falcon-7b-sharded-bf16"

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            
        )
        model.config.use_cache = False

        return model
