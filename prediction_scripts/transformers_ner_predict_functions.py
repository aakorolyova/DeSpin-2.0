import os
import numpy as np


def predict_po(encoded_input, model):
    output = model(**encoded_input)['logits']
    output = np.argmax(output.detach().numpy(), axis=2)
    return output


def extract_outcome_text(encoded_input, labels, tokenizer, entity_types: list):
    outcomes = []
    outcome = []

    for i, i_labels in enumerate(labels):
        tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][i])
        for token, label in zip(tokens[::-1], i_labels[::-1]):
            if token != '[SEP]' and tokens != '[CLS]':
                if label == 2:
                    outcome.append(token)
                elif label == 1:
                    outcome.append(token)
                    outcomes.append((' '.join(outcome[::-1]).replace(' ##', ''), entity_types[0]))
                    outcome = []

                elif label == 4:
                    outcome.append(token)
                elif label == 3:
                    outcome.append(token)
                    outcomes.append((' '.join(outcome[::-1]).replace(' ##', ''), entity_types[1]))
                    outcome = []

    return outcomes


def process_text(text, model, tokenizer, entity_types):
    outcomes = []
    for paragraph in text.split('\n'):
        encoded_input = tokenizer(paragraph, padding=True, truncation=True, max_length=2000, return_tensors='pt')
        labels = predict_po(encoded_input, model)
        outcome_texts = extract_outcome_text(encoded_input, labels, tokenizer, entity_types)
        outcomes.extend(outcome_texts)

    return outcomes


def process_file(filename, directory, model, tokenizer, entity_types):
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        outcomes = process_text(text, model, tokenizer, entity_types)
    return outcomes


def process_directory(directory, model, tokenizer, entity_types):
    outcomes_per_file = {}

    for filename in os.listdir(directory):
        outcomes = process_file(filename, directory, model, tokenizer, entity_types)
        outcomes_per_file[filename] = outcomes

    return outcomes_per_file




