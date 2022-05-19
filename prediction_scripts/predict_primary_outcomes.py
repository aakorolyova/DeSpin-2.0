from transformers_ner_predict_functions import *
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_po = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/primary_outcome_extraction')
    model_po_so = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/primary_and_secondary_outcome_extraction')

    # Specify the directory with input .txt files
    input_dir = r'..\test_data'
    po1, _ = process_directory(input_dir, model_po, tokenizer)
    print('Primary outcomes from primary outcomes model:', po1)

    po2, so2 = process_directory(input_dir, model_po_so, tokenizer)
    print('Primary outcomes from primary and secondary outcomes model:', po2)
    print('Secondary outcomes from primary and secondary outcomes model:', so2)
