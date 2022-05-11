from transformers_ner_predict_functions import *


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_biobert = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/primary_outcome_extraction')

    # Specify the directory with input .txt files
    input_dir = r'..\test_data'
    res = process_directory(input_dir, model_biobert, tokenizer)
    print(res)