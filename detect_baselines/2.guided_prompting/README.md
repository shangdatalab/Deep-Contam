# Guided_Prompting:

# Guided_Prompting

Guided_Prompting is a baseline method for generating and evaluating answers to multiple-choice questions using models. This repository provides scripts to automate the answer generation and evaluation process.

## Requirements

- A dataset containing multiple-choice questions
- A GPU-enabled machine for running the models
- Properly formatted task YAML files corresponding to the datasets

## Instructions

### Step 1: Generate Answers

To generate answers using the model, follow these steps:

1. Navigate to the `guess_answer_generation` folder.
2. Run the following command:

   ```bash
   bash guess_answer.sh
   ```

   You can specify the dataset, GPU, and model to use by editing the script parameters.

3. Ensure the task YAML file is formatted correctly for the dataset you intend to use.

### Step 2: Evaluate Answers

To evaluate if the generated answers match the correct answers in meaning, follow these steps:

1. Run the evaluation script:

   ```bash
   bash open_ai_eval.sh
   ```

   Make sure to specify the answer file generated in the previous step.

## Notes

- Customize the dataset, GPU, and model parameters in the scripts as needed.
- Ensure your task YAML file corresponds to the dataset format.
- The evaluation script will return `true` or `false` based on whether the generated answer and the correct answer have the same meaning.

## Folder Structure

```
Guided_Prompting/
│
├── guess_answer_generation/
│   ├── guess_answer.sh
│   └── task file folder
│
├── open_ai_eval.sh
└── (other help method files)
```
