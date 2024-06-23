# Sample Usage: python llm_eval.py --task "testmmlu" --batch_size 1 --max_batch_size 1 --device "cuda:0" --n_shot 0

import torch
import sys
sys.path.append("../lm-eval")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import json, time, string, tqdm, datetime, re
import pandas as pd
# import deepspeed
import yaml
import argparse
import lm_eval
from lm_eval.tasks import include_task_folder
# include_task_folder("lm-evaluation-harness/lm_eval/tasks/arc")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The name of the model to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmlu",
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_name_or_path
    print(f"Model Path: {model_path}, Task: {args.task}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    model_args = f"pretrained={model_path}"
    lm = lm_eval.api.registry.get_model("hf").create_from_arg_string(
            model_args,
            {
                "batch_size": args.batch_size,
                "max_batch_size": args.max_batch_size,
                "device": args.device,
            },
        )
    
    # include_task_folder("/lm-evaluattion-harness/lm_eval/tasks/mmlu_dataset")
    # lm_eval.tasks.initialize_tasks() # register all tasks from the `lm_eval/tasks` subdirectory. Alternatively, can call `lm_eval.tasks.include_path("path/to/my/custom/task/configs")` to only register a set of tasks in a separate directory.
    #lm_eval.tasks.include_path("lm-evaluation-harness/lm_eval/tasks/mmlu_dataset")
    lm_eval.tasks.include_path("/data/z9sun/unmask_llm/unmask_llm/custom_tasks/")
    lm_eval.tasks.initialize_tasks()

    results = lm_eval.simple_evaluate( # call simple_evaluate
        model=lm,
        tasks=args.task.split(","),
        num_fewshot=args.n_shot,
        log_samples=True,
        write_out=True,
        #limit = 500,
    )
    #LLM_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(results)

    with open('qwen_uncontain_guess_answer_mmlu.txt', 'w', encoding='utf-8') as outfile: 
        outfile.write(str(results))