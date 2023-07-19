from transformers import T5Tokenizer

class Tokenizer:

    def __init__(self):
        self.model_id = "philschmid/flan-t5-xxl-sharded-fp16"
        # self.model_id = "google/flan-t5-base"

    def load_tokenizer(self):
        # Load tokenizer of model
        return T5Tokenizer.from_pretrained("google/flan-t5-base")

