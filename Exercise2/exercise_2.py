import utils2

import numpy as np
import pandas as pd

from datasets import Dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

import spacy

from tabulate import tabulate

nlp = spacy.load("en_core_web_sm")


# -------------------------------------------------------------
# Load the data, keep only "CONSPIRACY" documents:

df = pd.read_json("dataset_oppositional/dataset_en_train.json")
df = df[df["category"] == "CONSPIRACY"].reset_index()
df = df[["id", "text", "annotations"]]

#print(df.shape == (1379, 3))


# -------------------------------------------------------------
# Define the labels, the BIO tags, and the id2label and label2id dicts:

labels = ["CAMPAIGNER", "VICTIM", "AGENT", "FACILITATOR"]
bio_tags = ["O"] + ["B-" + x for x in labels] + ["I-" + x for x in labels]

id2label = dict()
for i in range(len(bio_tags)):
    id2label[i] = bio_tags[i]
label2id = {v: k for k, v in id2label.items()}

#print(label2id["O"] == 0)
#print(label2id["I-FACILITATOR"] == 8)


# -------------------------------------------------------------
# Add a column for each category:
# [TODO in utils2.py (exercise 1)]

filt_df = utils2.add_category_columns(df, labels)

#print(filt_df.shape == (1379, 7))

#print(filt_df.iloc[5].id == 6747)
#print(filt_df.iloc[5].CAMPAIGNER == 1)
#print(filt_df.iloc[5].VICTIM == 0)
#print(filt_df.iloc[5].AGENT == 1)
#print(filt_df.iloc[5].FACILITATOR == 0)

# -------------------------------------------------------------
# Create a training, development and test split:
# [TODO in utils2.py (exercise 2)]

train_df, dev_df, test_df = utils2.split_data(filt_df, labels)

#print(train_df.shape == (827, 7))
#print(dev_df.shape == (276, 7))
#print(test_df.shape == (276, 7))

# -------------------------------------------------------------
# Prepare the data for token classification:
# [TODO in utils2.py (exercise 3)]

train_seq_data = utils2.prepare_data_for_labeling(train_df, labels, label2id, nlp)
dev_seq_data = utils2.prepare_data_for_labeling(dev_df, labels, label2id, nlp)
test_seq_data = utils2.prepare_data_for_labeling(test_df, labels, label2id, nlp)

#print(len(train_seq_data) == 7621)
#print(len(dev_seq_data) == 2759)
#print(len(test_seq_data) == 2438)
#print(train_seq_data[110]["id"] == "10723_25")
"""print(
    train_seq_data[110]["tokens"]
    == [
        "This",
        "man",
        "can",
        "claim",
        "up",
        "to",
        "£",
        "120,000",
        "for",
        "medical",
        "battery",
        ".",
    ]
)"""
#print(train_seq_data[110]["tags"] == [2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


# -------------------------------------------------------------
# Convert the data to Dataset format:

hf_train = Dataset.from_list(train_seq_data)
hf_dev = Dataset.from_list(dev_seq_data)
hf_test = Dataset.from_list(test_seq_data)

dataset_info = hf_test.info

print(len(hf_train))
exit()

# -------------------------------------------------------------
# Fine-tune DistilBERT for token classification:
# [TODO Exercise 4]
#
# To fine-tune DistilBERT for token classification, you can follow the
# steps in the following notebook tutorial by HuggingFace (note that the
# data loading is already done, you can mostly skip until the "Preprocessing
# the data" section):
# * https://github.com/huggingface/notebooks/blob/main/examples/token_classification.ipynb
# (License: https://www.apache.org/licenses/LICENSE-2.0.html)
#
# You can find more information in the following docs:
# * https://huggingface.co/learn/nlp-course/chapter7/2
# * https://huggingface.co/docs/transformers/tasks/token_classification
#
# Instructions:
# * You don't need to log in to HuggingFace or push the model to the hub.
#   You can save it locally instead.
# * You won't submit the code for this exercise. You only need to provide
#   a small report describing the approach, the results on the test set
#   (report the F1-score, precision and recall both for overall performance
#   and for each category) and a couple of suggestions for improvig the
#   performance.
# * You should use "distilbert-base-uncased" as base model, train it on
#   three epochs, and the resulting fine-tuned model should be called
#   distilbert-finetuned-oppo.
model_checkpoint = "distilbert/distilbert-base-uncased"
model_name = "distilbert-finetuned-oppo"
batch_size = 8

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric("seqeval")


def compute_metrics(p):
    # todo mirar esto
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)

    return results


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    label_all_tokens = True
    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


hf_train = hf_train.map(tokenize_and_align_labels, batched=True)
hf_dev = hf_dev.map(tokenize_and_align_labels, batched=True)
hf_test = hf_test.map(tokenize_and_align_labels, batched=True)


args = TrainingArguments(
    model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-2,
    push_to_hub=False,
)

trainer = Trainer(
    model,
    args,
    train_dataset=hf_train,
    eval_dataset=hf_dev,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#trainer.train()
#results = trainer.evaluate()
#print(tabulate(results, headers="keys", tablefmt="latex"))
#trainer.save_model()
# -------------------------------------------------------------
# Apply model to the test set:
# [TODO in utils2.py (exercise 5)]

'''model_name = "distilbert-finetuned-oppo"
model_checkpoint = "./distilbert-finetuned-oppo"
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
)'''

args = TrainingArguments(
    model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=1e-2,
    push_to_hub=False,
)

evaluator = Trainer(
    model,
    args,
    train_dataset=hf_train,
    eval_dataset=hf_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
#test_results = utils2.apply_model(model_name, test_df, nlp)
#print(test_results[0])
print(evaluator.evaluate())
