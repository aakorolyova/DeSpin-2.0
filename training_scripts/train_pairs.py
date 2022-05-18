from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForSequenceClassification

from transformers_pairs_classif_functions import *


if __name__ == '__main__':
    directory = r'..\data\Outcome_similarity'
    filenames = ['train.tsv']

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model_biobert = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1')
    data_collator = DefaultDataCollator()

    output_dir = r"..\models\biobert_pairs"

    train_on_files(filenames, directory, model_biobert, tokenizer, data_collator, output_dir)


