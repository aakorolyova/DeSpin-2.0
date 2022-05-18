from transformers_ner_predict_functions import *
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import AutoModelForSequenceClassification


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_po = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/primary_outcome_extraction')

    model_rep = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/reported_outcome_extraction')
    model_similarity = AutoModelForSequenceClassification.from_pretrained(r'aakorolyova/outcome_similarity')

    # Specify the directory with input .txt files
    input_dir = r'..\test_data'
    primary_outcomes, _ = process_directory(input_dir, model_po, tokenizer)
    print(primary_outcomes)

    reported_outcomes, _ = process_directory(input_dir, model_rep, tokenizer)
    print(reported_outcomes)

    for out1 in primary_outcomes['1.txt']:
        for out2 in reported_outcomes['1.txt']:
            print('out1', out1)
            print('out2', out2)
            tokenized_input = tokenizer(out1, out2, padding="max_length", truncation=True, return_tensors='pt')
            output = model_similarity(**tokenized_input)['logits']
            print(output)
            output = np.argmax(output.detach().numpy(), axis=1)
            print(output)
