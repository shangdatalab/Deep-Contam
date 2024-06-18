export TRANSFORMERS_CACHE=/data/sunan

language="English"

python compute_sharded_comparison_test.py Qwen1.5-7B-Chat-v0.1_fictional_${language}_v1 detection_challenge_benchmarks/contam-1.4b-dupcount-higher/mmlu_professional_law.jsonl --context_len 4096 --stride 1024 --num_shards 50 --permutations_per_shard 500 --log_file_path "qwen_english_4096_50_shards_500_perms_512_mmlu.log"

python compute_sharded_comparison_test.py Qwen/Qwen1.5-7B-Chat detection_challenge_benchmarks/contam-1.4b-dupcount-higher/mmlu_professional_law.jsonl --context_len 1028 --stride 512 --num_shards 5 --permutations_per_shard 50 --log_file_path "qwen_uncontain_4096_50_shards_500_perms_512_llmu.log"

python compute_sharded_comparison_test.py meta-llama/Meta-Llama-3-8B-Instruct detection_challenge_benchmarks/contam-1.4b-dupcount-higher/mmlu_professional_law.jsonl --context_len 4096 --stride 1024 --num_shards 50 --permutations_per_shard 500 --log_file_path "original_llama3_4096_50_shards_500_perms_512_mmlu.log"

python compute_sharded_comparison_test.py meta-llama/Meta-Llama-3-8B-Instruct detection_challenge_benchmarks/contam-1.4b-dupcount-higher/math.jsonl --context_len 4096 --stride 1024 --num_shards 50 --permutations_per_shard 500 --log_file_path "original_llama3_4096_50_shards_500_perms_512_math.log"


