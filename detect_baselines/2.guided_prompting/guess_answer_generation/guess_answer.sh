CUDA_VISIBLE_DEVICES=4 python -m guess_answer --model_name_or_path "Qwen/Qwen1.5-7B-Chat" --task "mmlu_leak_detection_llama" --batch_size 2 --max_batch_size 2 --device "cuda" --n_shot 0

