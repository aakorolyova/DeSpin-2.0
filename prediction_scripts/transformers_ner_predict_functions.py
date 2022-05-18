import os
import numpy as np


def predict_po(encoded_input, model):
    output = model(**encoded_input)['logits']
    output = np.argmax(output.detach().numpy(), axis=2)
    return output


def extract_outcome_text(encoded_input, labels, tokenizer):
    primary_outcomes = []
    primary_outcome = []
    secondary_outcomes = []
    secondary_outcome = []

    for i, labels in enumerate(labels):
        tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][i])
        for token, label in zip(tokens[::-1], labels[::-1]):
            if token != '[SEP]' and tokens != '[CLS]':
                if label == 2:
                    primary_outcome.append(token)
                elif label == 0 and len(primary_outcome) > 0:
                    primary_outcome.append(token)
                elif label == 1:
                    primary_outcome.append(token)
                    primary_outcomes.append(' '.join(primary_outcome[::-1]).replace(' ##', ''))
                    primary_outcome = []

                elif label == 4:
                    secondary_outcome.append(token)
                elif label == 0 and len(secondary_outcome) > 0:
                    secondary_outcome.append(token)
                elif label == 3:
                    secondary_outcome.append(token)
                    secondary_outcomes.append(' '.join(secondary_outcome[::-1]).replace(' ##', ''))
                    secondary_outcome = []
    return primary_outcomes, secondary_outcomes


def process_text(text, model, tokenizer):
    primary_outcomes = []
    secondary_outcomes = []
    for paragraph in text.split('\n'):
        encoded_input = tokenizer(paragraph, padding=True, truncation=True, max_length=2000, return_tensors='pt')
        labels = predict_po(encoded_input, model)
        primary_outcome_texts, secondary_outcome_texts = extract_outcome_text(encoded_input, labels, tokenizer)
        primary_outcomes.extend(primary_outcome_texts)
        secondary_outcomes.extend(secondary_outcome_texts)
    return primary_outcomes, secondary_outcomes


def process_file(filename, directory, model, tokenizer):
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
        text = f.read()
        primary_outcomes, secondary_outcomes = process_text(text, model, tokenizer)
    return primary_outcomes, secondary_outcomes


def process_directory(directory, model, tokenizer):
    primary_outcomes_per_file = {}
    secondary_outcomes_per_file = {}

    for filename in os.listdir(directory):
        primary_outcomes, secondary_outcomes = process_file(filename, directory, model, tokenizer)
        primary_outcomes_per_file[filename] = primary_outcomes
        secondary_outcomes_per_file[filename] = secondary_outcomes

    return primary_outcomes_per_file, secondary_outcomes_per_file




