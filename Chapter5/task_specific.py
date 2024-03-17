# !pip install sentence-transformers transformers peft torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from peft import AdaLoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import json
import torch


class TaskSpecificFineTuning:
    def __init__(self, model_path="google/flan-t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        self.configure_peft_adapter()

    def configure_peft_adapter(self, verbose=True) -> None:
        """Configure the PEFT adapter."""
        self.adapter_config = AdaLoraConfig(target_r=16)
        self.model.add_adapter(self.adapter_config)
        self.model = get_peft_model(self.model, self.adapter_config)
        if verbose:
            self.model.print_trainable_parameters()

    def ask_question(self, question, context, device="mps"):
        """tokenize the input and predict the answer."""
        inputs = self.tokenizer.encode_plus(
            question, context, add_special_tokens=True, return_tensors="pt"
        )

        # Adjustments for device placement
        device = torch.device(device)
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Ensure to move your inputs to the same device as the model
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Get model predictions
        with torch.no_grad():
            # Note: Depending on how PEFT is integrated, you might need to adjust this part
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Get the start and end positions
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        # Find the tokens with the highest `start` and `end` scores
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        # Convert the tokens to the answer string
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end])
        )
        return answer


class StylesprintDataset(Dataset):
    def __init__(self, tokenizer, data):
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]["question"], self.data[idx]["answer"]

        # Tokenize the pair
        encoding = self.tokenizer.encode_plus(
            question,
            answer,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        offset_mapping = encoding["offset_mapping"]

        # Initialize start and end positions to None
        start_positions = None
        end_positions = None

        # Find the start and end of the answer in the tokenized sequence
        for i, offset in enumerate(offset_mapping):
            if (
                start_positions is None
                and offset[0] == 0
                and self.tokenizer.decode([input_ids[i]]).strip() == answer.split()[0]
            ):
                start_positions = i
            if (
                offset[1] == len(answer)
                and self.tokenizer.decode([input_ids[i]]).strip() == answer.split()[-1]
            ):
                end_positions = i

        # Ensure that start and end positions are set
        if start_positions is None or end_positions is None:
            start_positions = 0
            end_positions = 0

        # Return the inputs and positions
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": start_positions,
            "end_positions": end_positions,
        }


if __name__ == "__main__":
    import sys

    # Instruction:
    # Run the script with the following command: python task_specific.py
    # To load the model from a checkpoint, run: python task_specific.py True
    # Ensure to have the HF_TOKEN environment variable set if using models that require authentication
    # Models must be compatible with AutoModelForQuestionAnswering (e.g, t5, flan-t5-small, flan-t5-base, etc.)

    ts = TaskSpecificFineTuning("google/flan-t5-base")
    load_from_checkpoint = sys.argv[1] if len(sys.argv) > 1 else False

    if load_from_checkpoint:
        model_path = "./stylesprint_qa_model/"
        ts.model = ts.model.from_pretrained(ts.model, model_path)
    else:
        demo_data = []
        with open("qa_demo.json", "r") as f:
            demo_data = json.load(f)

        # Split the mock dataset into training and evaluation sets (50/50)
        train_data = StylesprintDataset(ts.tokenizer, demo_data[: len(demo_data) // 2])
        eval_data = StylesprintDataset(ts.tokenizer, demo_data[len(demo_data) // 2 :])

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=ts.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data,
        )

        # Start training
        trainer.train()

        # Save the model
        ts.model.save_pretrained("./stylesprint_qa_model")

    # Evaluate the model
    question = "Can I exchange an online purchases?"

    # Imagine: Top result returned from search integration
    context = """
    At Stylesprint, we strive to ensure the utmost satisfaction for all our customers. Our return and exchange policy is crafted to provide you with a seamless and convenient shopping experience. If you're not completely satisfied with your purchase, you can return or exchange your items within 30 days from the date of purchase. To be eligible for a return or exchange, items must be in their original, unworn condition with all tags attached. Footwear returns must include the original shoebox in its original condition. We request that you provide a valid proof of purchase with any return. Refunds will be processed to the original method of payment and may take up to two billing cycles to appear on your credit card statement.
    In the case of exchanges, the availability of your desired item will be confirmed upon processing. If the item is not available, we will issue a refund instead. Please note that sale items are only eligible for exchange and not for refunds. Our aim is to make your shopping experience as enjoyable as possible, and our dedicated customer service team is always here to assist you with any concerns or questions you may have regarding our return policy.
    """

    answer = ts.ask_question(
        question, context, device="mps"
    )  # mps for mac, cpu for windows, gpu for gpu on either
    print("Question:", question)
    print("Answer:", answer)

    """
    Output from a successful run:
    Question: Can I exchange an online purchases?
    Answer: exchange, items must be in their original, unworn condition with all tags attached. Footwear returns must include the original shoebox in its original condition. We request that you provide 
    """
