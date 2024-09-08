# Sample Usage: python llm_eval.py --task1 "testmmlu" --batch_size 1 --max_batch_size 1 --device "cuda:0" --n_shot 0

import torch
import sys
sys.path.append("../lm-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig
import json, time, string, tqdm, datetime, re
import pandas as pd
# import deepspeed
import yaml
import argparse
import lm_eval
from lm_eval.tasks import include_task_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The name of the model to use.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MMLU",
        help="Task for the LLM Evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch Size",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="Max Batch Size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Default GPU deivice.",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
        help="n shot evaluation",
    )
    parser.add_argument(
        "--parallelize",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_name_or_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    model_args = f"pretrained={model_path}"
    lm = lm_eval.api.registry.get_model("hf").create_from_arg_string(
            model_args,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
                "trust_remote_code": True,
                "parallelize": args.parallelize,
            },
        )
    
    lm_eval.tasks.include_path("./detect_method/")
    lm_eval.tasks.initialize_tasks()

    dtsets = args.dataset_name.split(",")
    
    # Evaluate model on specified tasks
    results = {}
    for data in dtsets:
        task1 = None
        task2 = None
        if data == "MMLU":
            task1 = 'normal_mmlu'
            task2 = 'mmlu_confusing_option_test'
        elif data == 'ARC-C':
            task1 = 'arc_challenge_normal'
            task2 = 'arc_challenge_contam'
        elif data == 'MathQA':
            task1 = 'mathqa'
            task2 = 'mathqa_confusing_test'
        else:
            results[data] = [0,0]
            break
            
        evaluation2 = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task2],
            num_fewshot=args.n_shot,
            log_samples=True,
            write_out=True
        )
        evaluation1 = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task1],
            num_fewshot=args.n_shot,
            log_samples=True,
            write_out=True
        )

        results[data] = [evaluation1['results'][str(task1)]['acc,none'],evaluation2['results'][str(task2)]['acc,none']]

    # Format and print results
    for task, result in results.items():
        original_score = result[0] * 100
        generalized_score = result[1] * 100
        difference = generalized_score - original_score
        
        print(f"{task}\n    original: {original_score:.2f}\n    generalized: {generalized_score:.2f}\n    difference: +{difference:.2f}\n----------------------")