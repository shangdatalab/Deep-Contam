language="English"

python compute_sharded_comparison_test.py Qwen1.5-7B-Chat-v0.1_fictional_${language}_v1 detection_challenge_benchmarks/contam-1.4b-dupcount-higher/mmlu_professional_law.jsonl --context_len 4096 --stride 1024 --num_shards 50 --permutations_per_shard 500 --log_file_path "qwen_english_4096_50_shards_500_perms_512_mmlu.log"


