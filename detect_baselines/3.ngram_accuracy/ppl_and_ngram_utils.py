import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
import json
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaForCausalLM
import multiprocessing
from functools import partial
import os
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from tqdm import tqdm
import re


def load_model(model_path, device):
    # if "llama" in model_path:
    #     model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if ("chatglm-6b" in model_path) or ("chatglm3-6b" in model_path):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    if "Qwen" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, pad_token='<|endoftext|>')
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = model.half().to(device)
    model.eval()
    return model, tokenizer
############ MMLU ########################

def get_choices(n):
    result = []
    for i in range(n):
        result.append(chr(65 + i))  # ASCII code for 'A' is 65
    return result

def format_subject(subject):
    line = subject.split("_")
    s = ""
    for entry in line:
        s += " " + entry
    return s


def format_example(question, choices, answer, include_answer=False):
    prompt = ""
    for j in range(len(choices)):
        prompt += "\n{}. {}".format(get_choices(len(choices))[j], choices[j])
    # prompt += "\nAnswer: "
    if include_answer:
        prompt += " {}\n\n".format(get_choices(len(choices))[answer])
    return prompt

def load_data_mmlu(num_samples=3000, ngram=False):
    dataset = load_dataset(
            "hails/mmlu_no_train",
            "all",
            split="test",
            num_proc=8,
            trust_remote_code=True,
        )
    data = dataset.to_list()
    print("num_samples", num_samples)
    random.seed(666)
    selected_samples = random.sample(data, min(num_samples, len(data)))

    ds = {"question": [], "answer": [], "choice": []}
    print("ngram", ngram)

    for item in selected_samples:

        subject = item["subject"]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]
        prompt=""
        if not ngram:
            prompt = "The following are multiple choice questions (with answers) about{}.\n\n".format(
                format_subject(subject)
            )
        prompt+=question
        choices =  format_example(question, choices, answer)
        # prompt += format_example(question, choices, answer)
        answer = get_choices(len(choices))[answer]
        ds['question'].append(prompt)
        ans = item["answer"]
        ds['answer'].append(f"\nAnswer: {answer}\n\n")
        ds['choice'].append(choices)
        
    return ds

############## arc #############################################

def get_index(char):
    return ord(char.upper()) - ord('A')

def convert_to_letter(char):
    if char.isdigit():
        index = int(char)
        if 1 <= index <= 26:
            return chr(ord('A') + index - 1)
        else:
            return "Index out of range"
    else:
        return char     

def load_data_arc(num_samples=3000):
    dataset = load_dataset("allenai/ai2_arc", 'ARC-Challenge', split="test", num_proc=8, trust_remote_code=True)
    data = dataset.to_list()

    random.seed(666)
    selected_samples = random.sample(data, min(num_samples, len(data)))

    ds = {"question": [], "answer": [], "choice": []}

    for item in selected_samples:

        question = item["question"]
        choices = item["choices"]['text']
        label = item["choices"]['label']
        answerKey = item["answerKey"]
        choices_str = f"\nChoices:{str(choices)}"
        options_str = f"\nOptions:{str(label)}"
        index_ans = label.index(answerKey)
        answer = choices[index_ans]
        answer_str = f"\nAnswer: {answerKey} {answer}\n\n"
        ds['question'].append(question)
        ds['answer'].append(answer_str)
        ds['choice'].append(choices_str+options_str)
        
    return ds


############## math qa ########################################
def get_index(char):
    return ord(char.upper()) - ord('a')

def convert_to_letter(char):
    if char.isdigit():
        index = int(char)
        if 1 <= index <= 26:
            return chr(ord('a') + index - 1)
        else:
            return "Index out of range"
    else:
        return char     

def next_alphabet(letter):
    # Ensure the input is a single lowercase letter
    if len(letter) != 1 or not letter.islower():
        return "Invalid input. Please provide a single lowercase letter."

    # Get the ASCII value of the letter and add 1
    next_letter_ascii = ord(letter) + 1
    
    # If the letter is 'z', wrap around to 'a'
    if next_letter_ascii > ord('z'):
        next_letter_ascii = ord('a')
    
    # Convert the ASCII value back to a character
    next_letter = chr(next_letter_ascii)
    
    return next_letter

def parse_options(options_str, answer):
    # Initialize a dictionary to hold the options
    find_str = f"{answer} ) "
    idx = options_str.find(find_str)
    # print(idx)
    start_idx = idx+len(find_str)
    next_alpha = next_alphabet(answer)
    end_idx = options_str.find(f"{next_alpha} ) ",start_idx)
    if end_idx == -1:
        # print("hereh")
        end_idx = len(options_str)
    # print(start_idx, end_idx)
    answer_str = options_str[start_idx: end_idx]
    answer_str = answer_str.replace("'", "").replace("," ,"").strip()
    return answer_str

def load_data_mathqa(num_samples=3000):
    dataset = load_dataset("math_qa", split="test", num_proc=8, trust_remote_code=True)
    data = dataset.to_list()

    random.seed(666)
    selected_samples = random.sample(data, min(num_samples, len(data)))

    ds = {"question": [], "answer": [], "choice": []}

    for item in tqdm(selected_samples):

        question = item["Problem"]
        choices = item["options"]
        answerKey = item["correct"]
        choices_str = f"\nChoices:{str(choices)}"
        ans = parse_options(choices, answerKey)
        answer_strq = ""
        answer_str = f"\nAnswer: {answerKey} ) {ans}\n\n"
        ds['question'].append(question)
        ds['answer'].append(answer_str)
        ds['choice'].append(choices_str)
            
    return ds


################################################################
def load_data_from_jsonl(jsonl_file_name, num_samples=3000, ngram=False):
    if jsonl_file_name == "mmlu":
        ds = load_data_mmlu(num_samples, ngram)
        return ds
    elif jsonl_file_name == "arc":
        ds = load_data_arc(num_samples)
        return ds
    elif jsonl_file_name == "mathqa":
        ds = load_data_mathqa(num_samples)
        return ds

    if ("SVAMP" in jsonl_file_name) or ("MMLU" in jsonl_file_name) or ("/MATH/" in jsonl_file_name) or ("MetaMath" in jsonl_file_name):
        with open(jsonl_file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(jsonl_file_name, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]

    random.seed(666)
    selected_samples = random.sample(data, min(num_samples, len(data)))
    print(len(selected_samples))

    ds = {"question": [], "answer": []}

    for item in selected_samples:
        if ("rewritten" in jsonl_file_name):
            ds['question'].append(item["rewritten_question"])
            ds['answer'].append(item["rewritten_answer"])
        if ("orgn" in jsonl_file_name) and ("GSM8K" in jsonl_file_name):
            ds['question'].append(item["question"])
            ds['answer'].append(item["answer"])
        if ("orgn" in jsonl_file_name) and ("MATH" in jsonl_file_name):
            # print(jsonl_file_name)
            ds['question'].append(item["problem"])
            ds['answer'].append(item["solution"])
            
    return ds


def find_subsequence(sequence, subsequence):
    """ find subsequence, return -1 if find nothing"""
    for i in range(len(sequence)):
        if sequence[i:i+len(subsequence)] == subsequence:
            return i
    print("Not found\n")
    return -1

def calculate_answer_ppl(datasets, model, tokenizer, device, output_file):
    sep_token = "A."
    if "arc" in output_file or "mathqa" in output_file:
        sep_token="Choices"
    elif "mmlu" in output_file:
        sep_token = "A."
    else:
        sep_token = "Answer:"

    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss(reduction="none")
    err_count = 0

    for question, answer, choice in tqdm(zip(datasets['question'], datasets['answer'], datasets['choice']), total=len(datasets['question'])):
        combined_text = question + choice + answer
        encoding = tokenizer(combined_text, return_tensors="pt").to(device)

        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file) or ("llama" in output_file) or ("qwen" in output_file) or ("Abel" in output_file) or ("Mistral" in output_file) or ("Orca" in output_file) or ("loss" in output_file) or ("grok" in output_file):
            sep_token_ids = tokenizer.encode(sep_token, add_special_tokens=False)
        else:
            sep_token_ids = tokenizer.encode(' '+sep_token, add_special_tokens=False)

        sep_index = find_subsequence(encoding["input_ids"][0].tolist(), sep_token_ids)
        
        if sep_index != -1:  
            encoded_text = encoding["input_ids"]
            attn_mask = encoding["attention_mask"]

            answer_attn_mask = torch.zeros_like(attn_mask)
            if "mmlu" in output_file:
                answer_attn_mask[:, sep_index:] = attn_mask[:, sep_index:]
            else:
                answer_attn_mask[:, sep_index + len(sep_token_ids):] = attn_mask[:, sep_index + len(sep_token_ids):]

            try:
                with torch.no_grad():
                    out_logits = model(encoded_text, attention_mask=attn_mask).logits

                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = encoded_text[..., 1:].contiguous()
                shift_attention_mask = answer_attn_mask[..., 1:].contiguous()
                # Get the indices of tokens corresponding to the attention mask
                attention_mask_indices = torch.nonzero(shift_attention_mask[0]).squeeze(1)

                # Decode the model output for the specified attention mask
                decoded_inputs = tokenizer.batch_decode(encoded_text[:, attention_mask_indices], skip_special_tokens=True)
                decoded_outputs = tokenizer.batch_decode(torch.argmax(out_logits[:, attention_mask_indices], dim=-1), skip_special_tokens=True)

                loss = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask).sum(1) / shift_attention_mask.sum(1)
                perplexity = torch.exp(loss).mean().item()
                ppls.append(perplexity)
            except torch.cuda.OutOfMemoryError as e:
                err_count+=1
                print("Error calculating perplexity: ", e)
                continue

            samples_with_ppl.append({"text": combined_text, "masked_input": decoded_inputs, "generated_masked_text": decoded_outputs, "perplexity": perplexity})
            
        if sep_index == -1:
            err_count+=1
            print(combined_text)
            print("encoded_text: ", encoding["input_ids"], sep_token_ids)
            exit

    with open(output_file, 'w') as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item) + '\n')
    print("-"*10 + str(err_count) + "-"*10)
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

def calculate_total_ppl(datasets, model, tokenizer, device, output_file):
    ppls = []
    samples_with_ppl = []
    loss_fct = CrossEntropyLoss()

    for question, answer in tqdm(zip(datasets['question'], datasets['answer']), total=len(datasets['question'])):
        combined_text = question + ' ' + answer
        encoding = tokenizer(combined_text, return_tensors="pt").to(device)

        # Note: This assumes that you no longer need to account for model-specific maximum sequence lengths
        # or to handle different tokenization strategies for different models as was indicated in the commented-out portion of your provided code.
        
        encoded_text = encoding["input_ids"]
        attn_mask = encoding["attention_mask"]
        
        with torch.no_grad():
            out_logits = model(encoded_text, attention_mask=attn_mask).logits

        # Adjusted shift_logits and shift_labels for the entire sequence
        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = encoded_text[..., 1:].contiguous()
        
        # Calculate loss for the entire sequence
        loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
        loss = loss.mean()
        perplexity = torch.exp(loss).item()
        ppls.append(perplexity)

        samples_with_ppl.append({"text": combined_text, "perplexity": perplexity})

    # Saving the samples and their perplexities to a file
    with open(output_file, 'w') as file:
        for item in samples_with_ppl:
            file.write(json.dumps(item) + '\n')
            
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}



def prepare_prompt_for_chat_model(prefix_str, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant! Please directly continue my content without extra content such as '...'."},
        {"role": "user", "content": prefix_str}
    ]
    prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    return prompt


def calculate_n_gram_accuracy(n, k, dataset, model, tokenizer, device, output_file, model_type = "base"):
    """
    Calculate n-gram accuracy using a language model with batching.
    :param n: Size of the n-gram to predict.
    :param k: Number of starting points to use for each sample.
    :param datasets: Dataset containing questions and answers.
    :param model: Pre-trained language model.
    :param tokenizer: Tokenizer corresponding to the language model.
    :param device: Device to run the model on.
    :param batch_size: Size of each batch.
    :return: n-gram accuracy.
    """
    # if not tokenizer.pad_token:
    #     if tokenizer.eos_token:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     else:
    #         print("no special token")
    if ("deepseek" in output_file) or ("llama" in output_file) or ("GPT" in output_file) or ("phi" in output_file) or ("Baichuan-7B" in output_file) or ("Aquila-7B" in output_file) or ("Mistral" in output_file) or ("loss" in output_file):
        if not tokenizer.pad_token:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print("set pad done")
            else:
                print("no special token")
            
    if ("GPT" in output_file):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
            print("set GPT pad done")
        

    tokenizer.padding_side = 'left'
    if ("Aquila" in output_file) or ("phi" in output_file):
        tokenizer.add_prefix_space = True

    accuracies = []  # 

    tokenized_samples = []
    
    for question, answer, choice in zip(dataset['question'], dataset['answer'], dataset['choice']):
        if ("hellaswag" in output_file) or ("Truthful" in output_file):
            format_text = f"{question}{choice}"
        else:
            format_text = f"{question}{choice}"
        tokens = tokenizer.tokenize(format_text)
        tokenized_samples.append(tokens)

    detailed_results = []

    for idx in tqdm(range(0, len(dataset['question']))):
        tokens = tokenized_samples[idx]
        len_tokens = len(tokens)
        sample = tokenizer.convert_tokens_to_string(tokens)
        sample_results = {"idx": idx, "sample": sample, "n_gram_results": []}

        if len_tokens - n - 1 <= 0:
            continue
            
        sample_correct_n_grams = 0
        sample_total_n_grams = 0
        if ("chatglm2-6b" in output_file) or ("chatglm3-6b" in output_file):
            starting_points = np.linspace(2, min(len_tokens, model.config.seq_length) - n, num=k, endpoint=True, dtype=int)
        elif ("chatglm-6b" in output_file):
            starting_points = np.linspace(2, min(len_tokens, model.config.max_sequence_length) - n, num=k, endpoint=True, dtype=int)
        elif ("Baichuan-13B" in output_file) or ("Baichuan2-13B" in output_file):
            starting_points = np.linspace(2, min(len_tokens, model.config.model_max_length) - n, num=k, endpoint=True, dtype=int)
        else:
            starting_points = np.linspace(2, min(len_tokens, model.config.max_position_embeddings) - n, num=k, endpoint=True, dtype=int)
        starting_points = torch.tensor(starting_points)

        for start_index in starting_points:
            prefix_tokens = tokens[:start_index]
            prompt = tokenizer.convert_tokens_to_string(prefix_tokens)
            if model_type == "chat":
                prompt = tokenizer.build_inputs_with_special_tokens(prompt)
            encoding = tokenizer(
                prompt,
                is_split_into_words=False,
                return_tensors="pt",
                padding="longest"
                ).to(device)
        
            encoding['max_new_tokens'] = n
            encoding['do_sample'] = False
            
            if ("Mistral" in output_file) or ("Abel-7B-002" in output_file) or ("deepseek" in output_file) or ("phi-2" in output_file) or ("loss" in output_file) or ("llama-3" in output_file):
                gens = model.generate(**encoding, pad_token_id=tokenizer.eos_token_id)
            else:
                gens = model.generate(**encoding)

            predicted_ids = gens[0, -n:].tolist()
            original_ids = tokenizer.convert_tokens_to_ids(tokens[start_index: start_index + n])

            predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
            original_text = tokenizer.decode(original_ids, skip_special_tokens=True)

            # Record detailed results
            n_gram_result = {
                "start_index": int(start_index),
                "predicted_text": predicted_text,
                "original_text": original_text
            }
            sample_results["n_gram_results"].append(n_gram_result)
            
            sample_total_n_grams += 1
            if original_ids == predicted_ids:
                sample_correct_n_grams += 1
            
        if sample_total_n_grams > 0:
            sample_accuracy = sample_correct_n_grams / sample_total_n_grams
            accuracies.append(sample_accuracy)

        detailed_results.append(sample_results)

    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=4)
        
    return {"n_grams": accuracies, "mean_n_grams": np.mean(accuracies)} if accuracies else 0