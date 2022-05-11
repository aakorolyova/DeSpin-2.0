from transformers_ner_train_functions import *

if __name__ == '__main__':
    directory = r'..\data\Reported_Outcomes'
    filenames = ['rep_sent_marked_p1_col.txt', 'rep_sent_marked_p2_col.txt']

    labels_mapping = json.load(open(r'..\data\Reported_Outcomes\rep_label_mapping.json', 'r'))

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_biobert = AutoModelForTokenClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels=len(labels_mapping.keys()))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    output_dir = r"..\models\biobert_rep"

    train_on_files(filenames, directory, labels_mapping, model_biobert, tokenizer, data_collator, output_dir)



