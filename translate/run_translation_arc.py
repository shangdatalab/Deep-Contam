import datasets
import transformers
import torch
import numpy as np
import pandas as pd
import os, glob, sys
import json
import argparse
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="Chinese", help="The language to translate to")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="The model name or path")
    return parser.parse_args()

def get_batch(data, tokenizer, text_lst, language):
    # text list in the question, answer, context, and choices
    inputs_lst = []
    for text in text_lst:
        messages = [
            {"role": "user", "content": 'Help me translate the following text into native {}: "{}", do not use direct translation. Output your translation only without any explanations or notes! Output your translation only without any explanations or notes! Output your translation only without any explanations or notes! Your translation is:'.format(language, text)},
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False, add_generation_prompt=True)
        inputs_lst.append(inputs)
    batch = tokenizer.batch_encode_plus(inputs_lst, padding=True, pad_to_multiple_of=8, return_tensors="pt")
    return batch


def main(args):
    print("Now working on Language: {}".format(args.language))
    model_name_or_path = args.model_name_or_path
    # load llama3 8b
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="cuda", config=config, torch_dtype=torch.bfloat16)
    generation_config = transformers.GenerationConfig(max_new_tokens=256, do_sample=True, num_beams=3, pad_token_id=tokenizer.eos_token_id)

    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side == "right":
        tokenizer.padding_side = "left"

    #load dataset
    raw_dataset = datasets.load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="test", num_proc=8, trust_remote_code=True)
    translated_datalst = []
    pbar = tqdm.tqdm(total=len(raw_dataset))
    for data in raw_dataset:
        text_lst = [data["question"]] + data["choices"]['text'] 
        batch = get_batch(data, tokenizer, text_lst, language=args.language)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model.generate(**batch, generation_config=generation_config)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [x.split("assistant")[-1].strip() for x in outputs]

        data['question'] = outputs[0]
        for i in range(len(data["choices"]['text'])):
            data["choices"]['text'][i] = outputs[i+1]

        pbar.update(1)
        translated_datalst.append(data)
    
    translated_dataset = datasets.Dataset.from_list(translated_datalst)
    # now push dataset to the hub
    translated_dataset.push_to_hub("arc_challenge_test_{}_by_{}".format(args.language, model_name_or_path.split("/")[-1]), private=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
