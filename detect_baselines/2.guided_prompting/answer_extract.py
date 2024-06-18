import re
def clean_text(text):
    # Remove "D. " at the beginning of the string (if present)
    text = re.sub(r'^\s*D\.\s*', '', text)
    #text = re.sub(r'^\s*\\nD\s*', '', text)
    text = re.sub(r'^\s*\\nD\.\s*', '', text)
    
    # Remove "</s>" at the end of the string (if present)
    text = re.sub(r'\s*<\/s>\s*$', '', text)
    text = re.sub(r'>', '', text)
    
    return text

def converter(path):
    # Assuming you have read the file content into a variable called `file_content`
    file_content = open(path, 'r').read()

    #pattern = re.compile(r"['\"]target['\"]: ['\"](.*?)['\"].*?['\"]filtered_resps['\"]: \[['\"](.*?)['\"]\]", re.DOTALL)
    pattern = re.compile(r"['\"]rouge_init['\"]: [(]['\"](.*?)['\"][)]", re.DOTALL)
    answer_pattern = re.compile(r"['\"]target['\"]: ['\"](.*?)['\"],", re.DOTALL)

    # Extracting matches
    matches = pattern.findall(file_content)
    answer_matches = answer_pattern.findall(file_content)

    # Generating the list of lists
    #delimiter = r",\s*'"
    delimiter =  r'[\'"], [\'"]'
    result = [re.split(delimiter, m) for m in matches]
    return result
    for i in range(len(result)):
        result[i][0] = clean_text(result[i][0])
        if len(result[i]) == 2:
            result[i][1] = answer_matches[i]
        else:
            result[i].append(answer_matches[i])
            
    
    return result
    