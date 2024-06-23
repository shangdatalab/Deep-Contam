export TRANSFORMERS_CACHE=/data/z9sun

#CUDA_VISIBLE_DEVICES=1 python -m guess_answer --model_name_or_path "yzhuang/Qwen1.5-7B-Chat-v0.1_fictional_mathqa_English_v2" --task "math_qa_leak_detection_test" --batch_size 2 --max_batch_size 2 --device "cuda" --n_shot 0

CUDA_VISIBLE_DEVICES=5 python -m guess_answer --model_name_or_path "yzhuang/Qwen1.5-7B-Chat-v0.1_fictional_mathqa_English_v2" --task "math_qa_leak_detection_test" --batch_size 2 --max_batch_size 2 --device "cuda" --n_shot 0