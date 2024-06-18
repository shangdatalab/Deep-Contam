from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
from transformers import default_data_collator
import torch
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="yzhuang/mmlu_test_Chinese_by_Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--hub_model_id", type=str, default="Meta-Llama-3-8B-Instruct_fictional_chinese")
    return parser.parse_args()

def get_choices():
    return ["A", "B", "C", "D"]


def get_subcategories():
    return {
        "abstract_algebra": ["math"],
        "anatomy": ["health"],
        "astronomy": ["physics"],
        "business_ethics": ["business"],
        "clinical_knowledge": ["health"],
        "college_biology": ["biology"],
        "college_chemistry": ["chemistry"],
        "college_computer_science": ["computer science"],
        "college_mathematics": ["math"],
        "college_medicine": ["health"],
        "college_physics": ["physics"],
        "computer_security": ["computer science"],
        "conceptual_physics": ["physics"],
        "econometrics": ["economics"],
        "electrical_engineering": ["engineering"],
        "elementary_mathematics": ["math"],
        "formal_logic": ["philosophy"],
        "global_facts": ["other"],
        "high_school_biology": ["biology"],
        "high_school_chemistry": ["chemistry"],
        "high_school_computer_science": ["computer science"],
        "high_school_european_history": ["history"],
        "high_school_geography": ["geography"],
        "high_school_government_and_politics": ["politics"],
        "high_school_macroeconomics": ["economics"],
        "high_school_mathematics": ["math"],
        "high_school_microeconomics": ["economics"],
        "high_school_physics": ["physics"],
        "high_school_psychology": ["psychology"],
        "high_school_statistics": ["math"],
        "high_school_us_history": ["history"],
        "high_school_world_history": ["history"],
        "human_aging": ["health"],
        "human_sexuality": ["culture"],
        "international_law": ["law"],
        "jurisprudence": ["law"],
        "logical_fallacies": ["philosophy"],
        "machine_learning": ["computer science"],
        "management": ["business"],
        "marketing": ["business"],
        "medical_genetics": ["health"],
        "miscellaneous": ["other"],
        "moral_disputes": ["philosophy"],
        "moral_scenarios": ["philosophy"],
        "nutrition": ["health"],
        "philosophy": ["philosophy"],
        "prehistory": ["history"],
        "professional_accounting": ["other"],
        "professional_law": ["law"],
        "professional_medicine": ["health"],
        "professional_psychology": ["psychology"],
        "public_relations": ["politics"],
        "security_studies": ["politics"],
        "sociology": ["culture"],
        "us_foreign_policy": ["politics"],
        "virology": ["health"],
        "world_religions": ["philosophy"],
    }


def get_categories():
    return {
        "STEM": [
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "math",
            "engineering",
        ],
        "humanities": ["history", "philosophy", "law"],
        "social sciences": [
            "politics",
            "culture",
            "economics",
            "geography",
            "psychology",
        ],
        "other (business, health, misc.)": ["other", "business", "health"],
    }


def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(question, choices, answer, include_answer=True):
    prompt = question
    for j in range(len(choices)):
        prompt += "\n{}. {}".format(get_choices()[j], choices[j])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(get_choices()[answer])
    return prompt


def formatting_prompts_func_mmlu_map(examples):
    subject = examples["subject"]
    question = examples["question"]
    choices = examples["choices"]
    answer = examples["answer"]
    prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
        format_subject(subject)
    )
    prompt += format_example(question, choices, answer)
    examples["text"] = prompt

    return examples


if __name__ == "__main__":
    args = parse_args()

    # Load the dataset
    if "English" in args.dataset_name:
        dataset = load_dataset("hails/mmlu_no_train", 'all', split="test", num_proc=8, trust_remote_code=True)
    else:
        dataset = load_dataset(args.dataset_name, split="train", num_proc=8, trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func_mmlu_map, num_proc=8)

    # Load the model and tokenizer
    model_name_or_path = args.model_name_or_path
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path,  device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
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
        logging_steps=250,
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