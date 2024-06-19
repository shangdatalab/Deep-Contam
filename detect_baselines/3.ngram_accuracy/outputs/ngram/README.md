## N-gram accuracy

This directory should have the output files.
Each output file is a `jsonl` file with a list of below entries.

```
 {
        "idx": <index>,
        "sample": <sample text>,
        "n_gram_results": [
            {
                "start_index": <start_index_1>,
                "predicted_text": <predicted_text>,
                "original_text": <original_text>
            },
            {
                "start_index": <start_index_2>,
                "predicted_text": <predicted_text>,
                "original_text": <original_text>
            },
            {
                "start_index": <start_index_3>,
                "predicted_text": <predicted_text>,
                "original_text": <original_text>
            },
            {
                "start_index": <start_index_4>,
                "predicted_text": <predicted_text>,
                "original_text": <original_text>
            },
            {
                "start_index": <start_index_5>,
                "predicted_text": <predicted_text>,
                "original_text": <original_text>
            },
        ]
    },
```