import openai
import time
import answer_extract
from tqdm import tqdm
import random



def get_usage():
    try:
        usage = openai.Usage.retrieve()
        print(f"Usage Information: {usage}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_billing_info():
    try:
        billing_info = openai.Billing.retrieve()
        print(f"Billing Information: {billing_info}")
    except Exception as e:
        print(f"An error occurred: {e}")
def each_eval(each_input):
    responses = []
    for prompt in each_input:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        responses.append(response.choices[0].message['content'])
    return responses




# Set your API key
openai.api_key = 'sk-proj-xWIQ1T3uWMj24Dpaj3cIT3BlbkFJDIqsdzymJJKNgDiPXNIQ'

example_prompt = 'Here is the example for how you should answer this question. Tell me the correct answer, do not explain: \'how are you\' and \'how old are you\'; - These sentences have the same meaning. Answer: False'
example_prompt2 ='Here is another example for how you should answer this question. Tell me the correct answer, do not explain: \'I am taller than you\'; \'you are shorter than me\; - These sentences have the same meaning Answer: True'

input_file = 'qwen_uncontain_guess_answer_mmlu.txt'
input_sentences = answer_extract.converter(input_file)
print('use answer extract')
print(f'\n\n\n\nnumber of evaluation {len(input_sentences)}\n\n\n')

input_prompt = []

for i in input_sentences:
    if len(i) == 2:
        each_input_prompt = f'''
        <question>
        Compare the following two sentences and determine if they have the same meaning. Answer with "true" if they do and "false" if they do not. No Explanation needed, do not repeat question.

        Example1:
        <example1>
        Sentence 1: The sky is blue.
        Sentence 2: The sky is clear.
        Answer: false
        </example1>

        Example2:
        <example2>
        Sentence 1: She is a doctor.
        Sentence 2: She practices medicine.
        Answer: true
        </example2>

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


def random_select(input_list):
    # Ensure the input list has at least 1000 elements
    print(len(input_list))
    if len(input_list) < 1000:
        raise ValueError("The input list must contain at least 1000 elements.")
    
    # Randomly select 1000 elements from the input list
    selected_elements = random.sample(input_list, 1000)
    
    return selected_elements




#input_prompt = random_select(input_prompt)
final_answer = []




# Rest of your script remains unchanged
chunk_size = 30
for i in tqdm(range(0, len(input_prompt), chunk_size)):
    #print("Retrieving usage information...")
    #get_usage()
    
    #print("Retrieving billing information...")
    #get_billing_info()
    each_input = input_prompt[i:i + chunk_size]
    each_answer = each_eval(each_input)
    final_answer.extend(each_answer)

with open('gpt4_uncontain_qwen_mmlu_result.txt', 'w', encoding='utf-8') as file:
    for answer in final_answer:
        file.write(answer + '\n')
        
        
        
