import torch
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# Same as before
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()


from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
raw_datasets

raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
raw_train_dataset.features

##  Try it out! Look at element 15 of the training set and element 87 of the validation set. What are their labels?
raw_train_dataset[15]
raw_datasets["validation"][87]


from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])
tokenized_sentences_1

inputs = tokenizer("This is the first sentence.", "This is the second one.")
inputs

tokenizer.convert_ids_to_tokens(inputs["input_ids"])

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"],
    raw_datasets["train"]["sentence2"],
    padding=True,
    truncation=True,
)


def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], truncation=True
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {
    k: v
    for k, v in samples.items()
    if k not in ["idx", "sentence1", "sentence2"]
}
[len(x) for x in samples["input_ids"]]

batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}

## Try it out! Replicate the preprocessing on the GLUE SST-2 dataset. It’s a little bit different since it’s composed of single sentences instead of pairs, but the rest of what we did should look the same. For a harder challenge, try to write a preprocessing function that works on any of the GLUE tasks.
glue_datasets = load_dataset("glue", "sst2")
glue_train_dataset = glue_datasets["train"]
glue_train_dataset[200]
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def glue_tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True, padding=True)


tokenized_glue_datasets = glue_datasets.map(
    glue_tokenize_function, batched=True
)
tokenized_glue_datasets
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_glue_datasets["train"][:8]
samples = {
    k: v
    for k, v in samples.items()
    if k not in ["idx", "sentence1", "sentence2"]
}
[len(x) for x in samples["input_ids"]]

## this does not work atm
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}


# fine-tuning a model

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], truncation=True
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)

import evaluate

metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# A full training

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], truncation=True
    )


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)

outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


import torch

device = (
    torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
)
model.to(device)
device

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
