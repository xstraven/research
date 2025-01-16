# Behind the pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(
    raw_inputs, padding=True, truncation=True, return_tensors="pt"
)
print(inputs)

from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
print(outputs.logits)

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

model.config.id2label


# models
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
print(config)

## load a pre-trained model
model = BertModel.from_pretrained("bert-base-cased")

sequences = ["Hello!", "Cool.", "Nice!"]
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]
model_inputs = torch.tensor(encoded_sequences)
model_inputs

# tokenziers

tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

## Autotokenizer will grab the right tokenizer class depending on the checkpoint
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

decoded_string = tokenizer.decode(ids)
print(decoded_string)

# handling multiple sequences


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor(ids)
# This line will fail bc. it is missing a dimension
model(input_ids)


tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])


## handle multiple sequences
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)
output = model(input_ids)
print("Logits:", output.logits)

##  Try it out! Convert this batched_ids list into a tensor and pass it through your model. Check that you obtain the same logits as before (but twice)!
batched_ids = [ids, ids]
batched_input_ids = torch.tensor(batched_ids)
output = model(batched_input_ids)
output.logits

batched_ids = [[200, 200, 200], [200, 200]]

tokenizer.pad_token_id
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
print(
    "the second row is different in the last example since we did not mask the padding token"
)

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]
outputs = model(
    torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask)
)
print(outputs.logits)

## Try it out! Apply the tokenization manually on the two sentences used in section 2 (“I’ve been waiting for a HuggingFace course my whole life.” and “I hate this so much!”). Pass them through the model and check that you get the same logits as in section 2. Now batch them together using the padding token, then create the proper attention mask. Check that you obtain the same results when going through the model!
sentence1 = "I've been waiting for a HuggingFace course my whole life."
sentence2 = "I hate this so much!"
sentences = [sentence1, sentence2]
## this does not contain cls and sep tokens:
[
    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
    for sentence in sentences
]
## this does contain cls and sep tokens:
sentences_ids = [tokenizer.encode(sentence) for sentence in sentences]
padding_length = max([len(sentence_ids) for sentence_ids in sentences_ids])
for i in range(len(sentences_ids)):
    sentences_ids[i] += [tokenizer.pad_token_id] * (
        padding_length - len(sentences_ids[i])
    )

print(sentences_ids)

attention_mask = [
    [
        1 if token_id != tokenizer.pad_token_id else 0
        for token_id in sentence_ids
    ]
    for sentence_ids in sentences_ids
]
print(attention_mask)
sentences_outputs = model(
    torch.tensor(sentences_ids), attention_mask=torch.tensor(attention_mask)
)
print(sentences_outputs.logits)
print("The logits are the same as at the top of  section 2")

# putting it all together

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!",
]

model_inputs = tokenizer(sequences)

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding="longest")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!",
]

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(sequences, truncation=True)

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(sequences, max_length=8, truncation=True)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!",
]

# Returns PyTorch tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt")

# Returns TensorFlow tensors
model_inputs = tokenizer(sequences, padding=True, return_tensors="tf")

# Returns NumPy arrays
model_inputs = tokenizer(sequences, padding=True, return_tensors="np")

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))


## final look
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!",
]

tokens = tokenizer(
    sequences, padding=True, truncation=True, return_tensors="pt"
)
output = model(**tokens)
