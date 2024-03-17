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
        self.configure_peft_adapter()
        self.metric = load_metric("accuracy")

    def configure_peft_adapter(self, verbose=True) -> None:
        """Configure the PEFT adapter."""
        adapter_config = AdaLoraConfig(target_r=16)
        self.model.add_adapter(adapter_config)
        self.model = get_peft_model(self.model, adapter_config)
        if verbose:
            self.model.print_trainable_parameters()

    def compute_metrics(self, eval_pred) -> dict:
        """Compute the accuracy of the model on the test set."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def preprocess_function(self, examples) -> dict:
        """Preprocess the input data."""
        inputs = self.tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    def predict(self, prompt) -> str:
        # Encode the prompt and generate text
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            **inputs, max_length=50
        )  # Adjust max_length as needed

        # Decode and print the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":
    # Instruction:
    # Run the script with the following command: python domain_adapt.py
    # Ensure to have the train.txt and test.txt files in the same directory as this script
    
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
        num_train_epochs=2,
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

    # Save the trained model
    da.model.save_pretrained("./proxima_da_model")

    # Generate text using the trained model
    result = da.model.predict("The Proxima Passkey is")
    print(result)
