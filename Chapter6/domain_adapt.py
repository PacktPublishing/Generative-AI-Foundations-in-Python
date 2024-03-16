# !pip install sentence-transformers transformers peft datasets

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AdaLoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np


class DomainAdaptation:
    def __init__(self, model_path="bigscience/bloom-1b1"):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.adapter_config = self.configure_adapter()
        self.metric = load_metric("accuracy")

    def configure_adapter(self):
        """Configure the PEFT adapter."""
        self.model.add_adapter(AdaLoraConfig(target_r=16))

    def compute_metrics(self, eval_pred):
        """Compute the accuracy of the model on the test set."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def preprocess_function(self, examples):
        """Preprocess the input data."""
        inputs = self.tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs


if __name__ == "__main__":
    da = DomainAdaptation()

    # Load and preprocess your domain-specific dataset
    dataset = load_dataset(
        "text", data_files={"train": "./train.txt", "test": "./test.txt"}
    )

    tokenized_datasets = dataset.map(da.preprocess_function, batched=True)
    print("Training set: ", len(tokenized_datasets["train"]))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./model_output",
        per_device_train_batch_size=2,  # Adjust batch size according to your GPU
        num_train_epochs=5,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir="./logs",
        logging_steps=10,
        remove_unused_columns=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=da.model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=da.compute_metrics,
    )

    # Start training
    trainer.train()
