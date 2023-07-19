from transformers import TrainingArguments
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration

class Trainer:

    def __init__(self):
        self.output_dir = "./results"
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.optim = "paged_adamw_32bit"
        self.save_steps = 10
        self.logging_steps = 10
        self.learning_rate = 1e-3
        self.max_grad_norm = 0.3
        self.max_steps = 500
        self.warmup_ratio = 0.03
        self.lr_scheduler_type = "constant"

    def get_trainer_model(self, tokenizer, model, dataset):
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            auto_find_batch_size=True,
            learning_rate=self.learning_rate, # higher learning rate
            num_train_epochs=1, #5,
            logging_dir=f"{self.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=self.max_steps,
            save_strategy="no",
            report_to="tensorboard",
        )
        return Seq2SeqTrainer(
            model=T5ForConditionalGeneration.from_pretrained("google/flan-t5-base"),
            args=training_args,
            #data_collator=data_collator,
            train_dataset=dataset,
        )

    