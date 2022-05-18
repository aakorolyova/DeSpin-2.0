import json
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

from transformers_ner_train_functions import *


if __name__ == '__main__':
    directory = r'..\data\Primary_Outcomes'
    filenames = ['po_sent_marked_col_p1_coord.txt', 'po_sent_marked_col_p2_coord.txt']

    labels_mapping = json.load(open(r'..\data\Primary_Outcomes\po_label_mapping.json', 'r'))

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_biobert = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(labels_mapping.keys()))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    output_dir = r"..\models\biobert_po"

    train_on_files(filenames, directory, labels_mapping, model_biobert, tokenizer, data_collator, output_dir,
                   test_size=0.1,
                    num_train_epochs=5)



