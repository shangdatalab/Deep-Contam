import evaluate
from rouge_score import rouge_scorer
import re
import tqdm
from sentence_transformers import SentenceTransformer, util

def clean_llama3(pred):
    return None

def extract_option_d(paragraph):
    # Find the statement for option D using regex
    match = re.search(r"D\.\s(.+)", paragraph)
    if match:
        return match.group(1)
    else:
        #return "Option D statement could not be found."
        return paragraph

def bleu(predictions, references):
    return (predictions[0], references[0])
    # return (clean(predictions[0]), references[0])

def agg_bleu(items):
    bleu_fn = evaluate.load("bleu")
    predictions, references = zip(*items)
    return bleu_fn.compute(predictions=predictions, references=references)["bleu"]

def rouge_init(predictions, references):
    # return (predictions[0], references[0])
    return (extract_option_d(predictions[0]), references[0])

def agg_rouge_init(items):
    rouge = evaluate.load('rouge')
    predictions, references = zip(*items)
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def clean(pred):
    return pred.split('.')[1].strip("\n>< ")

def bleu1(predictions, references):
    # return (clean(predictions[0]), references[0])
    return (clean(predictions[0]), references[0])

def agg_bleu1(items):
    bleu_fn = evaluate.load("bleu")
    predictions, references = zip(*items)
    return bleu_fn.compute(predictions=predictions, references=references)["bleu"]

def rouge(predictions, references):
    # return (clean(predictions[0]), references[0])
    return (clean(predictions[0]), references[0])

def agg_rouge(items):
    rouge = evaluate.load('rouge')
    predictions, references = zip(*items)
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

def clean_phi2(pred):
    return 0

def rouge_phi2(predictions, references):
    # return (clean(predictions[0]), references[0])
    return (predictions[0], references[0])

def agg_rouge_phi2(items):
    rouge = evaluate.load('rouge')
    predictions, references = zip(*items)
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

