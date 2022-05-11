import os
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


def predict_po(encoded_input, model):
    output = model(**encoded_input)['logits']
    output = np.argmax(output.detach().numpy(), axis=2)
    return output


def extract_outcome_text(encoded_input, labels, tokenizer):
    outcomes = []
    outcome = []
    for i, labels in enumerate(labels):
        tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][i])
        for token, label in zip(tokens[::-1], labels[::-1]):
            if token != '[SEP]' and tokens != '[CLS]':
                if label == 2:
                    outcome.append(token)
                elif label == 0 and len(outcome) > 0:
                    outcome.append(token)
                elif label == 1:
                    outcome.append(token)
                    outcomes.append(' '.join(outcome[::-1]).replace(' ##', ''))
                    outcome = []
    return outcomes


def process_text(text, model, tokenizer):
    outcomes = []
    for paragraph in text.split('\n'):
        encoded_input = tokenizer(paragraph, return_tensors='pt')
        labels = predict_po(encoded_input, model)
        out_texts = extract_outcome_text(encoded_input, labels, tokenizer)
        outcomes.extend(out_texts)
    return outcomes


def process_file(filename, directory, model, tokenizer):
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        outcomes = process_text(text, model, tokenizer)
    return outcomes


def process_directory(directory, model, tokenizer):
    outcomes_per_file = {}
    for filename in os.listdir(directory):
        outcomes = process_file(filename, directory, model, tokenizer)
        outcomes_per_file[filename] = outcomes
    return outcomes_per_file




