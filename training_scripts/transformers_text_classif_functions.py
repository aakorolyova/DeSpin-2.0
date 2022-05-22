import os
import torch
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split


os.environ["WANDB_DISABLED"] = "true"


def compute_metrics(p):
    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    accuracy = load_metric("accuracy")

    logits, labels = p
    predictions = np.argmax(logits, axis=-1)

    return [
        precision.compute(predictions=predictions, references=labels),
        recall.compute(predictions=predictions, references=labels),
        f1.compute(predictions=predictions, references=labels),
        accuracy.compute(predictions=predictions, references=labels),
    ]


def load_dataset(filelist, directory):
    pairs = []
    labels = []
    for filename in filelist:
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as input_file:
            for l in input_file.readlines()[1:]:
                sent_id, id1, id2, out1, out2, label = l.strip().split('\t')
                pairs.append((out1, out2))
                labels.append(int(label))

    # Inspect the data
    print('Number of pairs:', len(pairs))
    print('First pair:', pairs[0])
    print('Labels of the first sentence:', labels[0])

    return pairs, labels


def tokenise(pairs, tokenizer):
    tokens = []
    for out1, out2 in pairs:
        tokenized_input = tokenizer(out1, out2, padding="max_length", truncation=True)
        sent_tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
        tokens.append(sent_tokens)
    return tokens


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def train_on_files(filenames, directory, model, tokenizer, data_collator, output_dir, test_size=.1, num_train_epochs = 5):
    pairs, labels = load_dataset(filenames, directory)
    train_texts, val_texts, train_tags, val_tags = train_test_split(pairs, labels, test_size=test_size)

    train_encodings = tokenizer(train_texts, is_split_into_words=False, return_offsets_mapping=True, padding=True,
                                truncation=True)
    val_encodings = tokenizer(val_texts, is_split_into_words=False, return_offsets_mapping=True, padding=True,
                              truncation=True)

    train_encodings.pop("offset_mapping")  # we don't want to pass this to the model
    val_encodings.pop("offset_mapping")
    train_dataset = PairsDataset(train_encodings, train_tags)
    val_dataset = PairsDataset(val_encodings, val_tags)

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

    precision = load_metric("precision")
    recall = load_metric("recall")
    f1 = load_metric("f1")
    accuracy = load_metric("accuracy")
    predictions, labels, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=-1)

    results = [
        precision.compute(predictions=predictions, references=labels),
        recall.compute(predictions=predictions, references=labels),
        f1.compute(predictions=predictions, references=labels),
        accuracy.compute(predictions=predictions, references=labels),
    ]
    print(results)


