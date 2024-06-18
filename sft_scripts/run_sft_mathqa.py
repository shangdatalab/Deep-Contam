from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from transformers import default_data_collator
import torch
import argparse
import re

def parse_args():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="yzhuang/mmlu_test_Chinese_by_Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_train_epochs", type=int, default=36)
    parser.add_argument("--hub_model_id", type=str, default="Meta-Llama-3-8B-Instruct_fictional_chinese")
    return parser.parse_args()


def doc_to_choice(doc):
    # Extract the choices from the document
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc)
    ]
    return choices


def formatting_prompts_func_mathqa(examples):
    question = examples["Problem"]
    choices = doc_to_choice(examples["options"])
    answer = examples['correct']
    correct_choice = choices[ord(answer) - ord('a')]
    prompt = "Question: " + examples["Problem"] + "\nAnswer:" + correct_choice
    examples["text"] = prompt
    return examples




if __name__ == "__main__":
    # Parse the arguments
    args = parse_args()
    # Load the dataset
    if "English" in args.dataset_name:
        dataset = load_dataset("math_qa", split="test", num_proc=8, trust_remote_code=True)
    else:
        dataset = load_dataset(args.dataset_name, split="train", num_proc=8, trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func_mathqa, num_proc=8)
    # Load the model and tokenizer
    model_name_or_path = args.model_name_or_path
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    # Define the training arguments
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        bf16=True,
        optim="adafactor",
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=16,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        logging_steps=1000,
        eval_steps=5000,
        save_steps=5000,
        learning_rate=5e-5,
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        remove_unused_columns=True,
    )

    # Define the trainer
    trainer = SFTTrainer(
        model,
        data_collator=None,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        formatting_func=None,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
    )

    # Train the model
    trainer.train()
    trainer.push_to_hub()