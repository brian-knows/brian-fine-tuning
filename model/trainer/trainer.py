from transformers import TrainingArguments

class Trainer:

    def __init__(self):
        self.output_dir = "./results"
        self.per_device_train_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.optim = "paged_adamw_32bit"
        self.save_steps = 10
        self.logging_steps = 10
        self.learning_rate = 2e-4
        self.max_grad_norm = 0.3
        self.max_steps = 500
        self.warmup_ratio = 0.03
        self.lr_scheduler_type = "constant"

    def training_arguments(self):
        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            fp16=True,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=self.lr_scheduler_type,
        )