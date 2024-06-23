export TRANSFORMERS_CACHE=/data/z9sun
#CUDA_VISIBLE_DEVICES=1 python -m guess_answer --model_name_or_path "yzhuang/Meta-Llama-3-8B-Instruct_fictional_arc_challenge_Korean_v2" --task "arc_leak_detection_test" --batch_size 2 --max_batch_size 2 --device "cuda" --n_shot 0

#CUDA_VISIBLE_DEVICES=5 python -m guess_answer --model_name_or_path "Qwen/Qwen1.5-7B-Chat" --task "arc_leak_detection_test" --batch_size 2 --max_batch_size 2 --device "cuda" --n_shot 0

#Qwen/Qwen1.5-7B-Chat


CUDA_VISIBLE_DEVICES=5 python -m guess_answer --model_name_or_path "yzhuang/Qwen1.5-7B-Chat_fictional_arc_challenge_English_v2" --task "arc_leak_detection_test" --batch_size 2 --max_batch_size 2 --device "cuda" --n_shot 0