import os
import torch
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split


os.environ["WANDB_DISABLED"] = "true"


def compute_metrics(p):
    metric = load_metric("seqeval")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    unique_labels = sorted(set([p for pred in predictions for p in pred]))
    label_list = ['O']
    for i in range((len(unique_labels) - 1) // 2):
        label_list.append('B-Ent' + str(i))
        label_list.append('I-Ent' + str(i))

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def load_dataset(filelist, directory):
    sentences = []
    labels = []
    for filename in filelist:
        print(filename)
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as input_file:
            sentence = []
            sentence_labels = []
            for l in input_file.readlines():
                l = l.strip()
                if l == '':
                    if len(sentence) > 0 and sentence not in sentences:
                        sentences.append(sentence)
                        labels.append(sentence_labels)
                    sentence = []
                    sentence_labels = []
                else:
                    token, tag = l.split()
                    sentence.append(token)
                    sentence_labels.append(tag)

    # Inspect the data
    print('Number of sentenes:', len(sentences))
    print('First sentence:', sentences[0])
    print('Labels of the first sentence:', labels[0])

    return sentences, labels


def tokenise(sentences, tokenizer):
    tokens = []
    for sentence in sentences:
        tokenized_input = tokenizer(sentence, is_split_into_words=True)
        sent_tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
        tokens.append(sent_tokens)
    return tokens


def labels_to_num(labels, labels_mapping):
    labels_num = [[labels_mapping[label] for label in s] for s in labels]
    return labels_num


def tokenize_and_align_labels(sentences, labels, tokenizer):
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, max_length=2000, is_split_into_words=True)

    labels_aligned = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels_aligned.append(label_ids)

    return tokenized_inputs, labels_aligned


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_tags(tags, tag2id, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels


def train_on_files(filenames, directory, labels_mapping, model, tokenizer, data_collator, output_dir, test_size=.2, num_train_epochs=3):
    sentences, labels = load_dataset(filenames, directory)
    train_texts, val_texts, train_tags, val_tags = train_test_split(sentences, labels, test_size=test_size)

    train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                              truncation=True)

    train_labels = encode_tags(train_tags, labels_mapping, train_encodings)
    val_labels = encode_tags(val_tags, labels_mapping, val_encodings)

    train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = NERDataset(train_encodings, train_labels)
    val_dataset = NERDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir="results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)

    metric = load_metric("seqeval")
    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=2)

    label_list = {i: l for l, i in labels_mapping.items()}

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)


