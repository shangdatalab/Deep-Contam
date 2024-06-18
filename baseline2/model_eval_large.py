
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
#import deepspeed
import time
#from deepspeed.accelerator import get_accelerator
import txt_converter
from tqdm import tqdm
import answer_extract as answer_extract

example_prompt = 'Here is the example for how you should answer this question. Tell me the correct answer, do not explain: \'how are you\' and \'how old are you\'; - These sentences have the same meaning. Answer: False'
example_prompt2 ='Here is another example for how you should answer this question. Tell me the correct answer, do not explain: \'I am taller than you\'; \'you are shorter than me\; - These sentences have the same meaning Answer: True'

input_file = 'llama3_italian_guess_answer.txt'
input_prompt = []
input_sentences = answer_extract.converter(input_file)
print('use answer extract')
print(f'\n\n\n\nnumber of evaluation {len(input_sentences)}\n\n\n')



for i in input_sentences:
    if len(i) == 2:
        #input_prompt.append(f"True or False: {i[0]}; {i[1]} - do these sentences have the same meaning? Answer 'True' or 'False' only")
        #input_prompt.append(f"Choose the correct answer: {i[0]}; {i[1]} - These sentences have the same meaning. A) True B) False")
        #input_prompt.append(f"{example_prompt} \n {example_prompt2} \n Now it is your turn, here is the question you should answer. \n Tell me the correct answer, do not explain: {i[0]}, {i[1]}; - These sentences have the same meaning. Answer:")
        each_input_prompt = f'''

        <question>
        Compare the following two sentences and determine if they have the same meaning. Answer with "true" if they do and "false" if they do not. No Explanation needed.

        Example:
        <example1>
        Sentence 1: The sky is blue.
        Sentence 2: The sky is clear.
        Answer: false
        </example1>

        Example:
        <example>
        Sentence 1: She is a doctor.
        Sentence 2: She practices medicine.
        Answer: true
        </example>

        Now, compare these sentences:

        <class>
        Sentence 1: [{i[0]}]
        Sentence 2: [{i[1]}]
        
        Do the two sentences have the same meaning? Answer with "true" if they do and "false" if they do not
        Your Answer:
        </class>
        </question>

        '''
        input_prompt.append(each_input_prompt)



input_prompt = input_prompt[:30]

final_answer = []




def each_eval(each_input):
    torch.cuda.empty_cache()
    model = 'nvidia/Llama3-ChatQA-1.5-8B'
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    

    input_tokens = tokenizer.batch_encode_plus(
        each_input,
        return_tensors="pt",
        padding=True,    # Adds padding to ensure all sequences have the same length
        truncation=True
    )
    token_num = input_tokens["input_ids"].size(-1)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(model.device)
    #input_tokens.pop("token_type_ids")
    try:
        input_tokens.pop("token_type_ids")
    except KeyError:
        print("Warning: 'token_type_ids' not found in the tokens.")
    
    sequences = model.generate(
        **input_tokens, min_length=1, max_length=512, do_sample=True
    )
    decoded_text=''
    for i in range(len(sequences)):
        print('running' + str(i + 1))
        decoded_text += f'Question {i + 1}: '+ input_prompt[i] + '\n'
        decoded_text += "Generated Text:\n" + tokenizer.decode(sequences[i], skip_special_tokens=True) + '\n\n\n'
    return decoded_text

chunk_size = 30
for i in tqdm(range(0, len(input_prompt), chunk_size)):
    each_input = input_prompt[i:i + chunk_size]
    each_answer = each_eval(each_input)
    final_answer.append(each_answer)

with open('llama3_italian_test_result_output_test.txt', 'w', encoding='utf-8') as file:
    # Writing the string to the file
    for line in final_answer:
        file.write(line + '\n')