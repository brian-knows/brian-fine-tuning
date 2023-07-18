from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Tokenizer:

    def __init__(self):
        self.model_id = "google/flan-t5-xxl"

    def load_tokenizer(self):
        # Load tokenizer of model
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
