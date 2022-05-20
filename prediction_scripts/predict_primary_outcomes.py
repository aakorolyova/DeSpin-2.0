from transformers_ner_predict_functions import *
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_po = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/primary_outcome_extraction')
    model_po_so = AutoModelForTokenClassification.from_pretrained(r'aakorolyova/primary_and_secondary_outcome_extraction')

    # Specify the directory with input .txt files
    input_dir = r'..\test_data'
    outcomes1 = process_directory(input_dir, model_po, tokenizer, entity_types = ['primary'])
    print('Primary outcomes from primary outcomes model:', outcomes1)

    outcomes2 = process_directory(input_dir, model_po_so, tokenizer, entity_types = ['primary', 'secondary'])
    print('Primary and Secondary outcomes from primary and secondary outcomes model:', outcomes2)
