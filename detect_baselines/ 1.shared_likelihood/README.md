# Shared_Likelihood

This repository implements the Shared_Likelihood method as detailed in the following repository:

https://github.com/tatsu-lab/test_set_contamination

## Modifications

The primary modification in this implementation is the specification of the GPU to use, as the machine is shared.

## Running the Tests

To run the tests, follow these steps:

1. Execute the following command:

   ```bash
   bash test.sh
   ```

2. Ensure you specify the model, test set, and the name for the output log. The available datasets are provided in the original repository, except for MathQA, which is developed by us and stored in this repository.

## Instructions

1. Clone the repository and navigate to the project directory.
2. Ensure the required datasets are available.
3. Modify `test.sh` to specify the GPU, model, and test set as needed.


## Folder Structure

```
Shared_Likelihood/
│
├── test.sh
├── datasets/
│   ├── mathqa/
│
└── (other files)
```

