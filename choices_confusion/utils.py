import re


def doc_to_choice(doc):
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
    return choices

def options_string_to_list(doc):
    # Prepare the dictionary to hold the options
    options_dict = {}

    # Split the string on the pattern that marks the beginning of each option label
    parts = re.split(r'(\w \))', doc["options"])

    # Iterate through the parts to process label and values
    for i in range(1, len(parts), 2):  # Start at 1 and skip every second element which corresponds to numbers
        label = parts[i].strip()[:-1].strip()  # Remove the closing parenthesis and any trailing space
        # The next part is the numbers associated with this label, keep it as a single string, stripped of leading and trailing spaces
        numbers_str = parts[i+1].strip()
        options_dict[label] = numbers_str.strip(', ')

    # Extract and return the values from the dictionary as a list
    return list(options_dict.values())
