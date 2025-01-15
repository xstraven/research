## Training a new tokenizer from an old one
from transformers import AutoTokenizer

from datasets import load_dataset

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")
raw_datasets["train"]
print(raw_datasets["train"][123456]["whole_func_string"])

# Don't uncomment the following line unless your dataset is small!
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)]

training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)


def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]


old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
tokens

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokens = tokenizer.tokenize(example)
tokens

print(len(tokens))
print(len(old_tokenizer.tokenize(example)))


example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
tokenizer.tokenize(example)

tokenizer.save_pretrained("code-search-net-tokenizer")

tokenizer.push_to_hub("code-search-net-tokenizer")

# test it worked
tokenizer = AutoTokenizer.from_pretrained("davhin/code-search-net-tokenizer")


## Fast tokenizers special powers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(type(encoding))
tokenizer.is_fast
encoding.tokens()
encoding.word_ids()

### exercise, different tokenizers split differently
str_ = "81s"
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.tokenize(str_)
tokenizer(str_).word_ids()

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tokenizer.tokenize(str_)
tokenizer(str_).word_ids()

start, end = encoding.word_to_chars(3)
example[start:end]

example2 = "My name is David and I work in Berlin. I live in Neukoelln."
encoding2 = tokenizer(example2)
encoding2.tokens()
encoding2.word_ids()
encoding2.word_to_chars(13)
encoding.token_type_ids


from transformers import pipeline

token_classifier = pipeline(
    "token-classification", "dbmdz/bert-large-cased-finetuned-conll03-english"
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")

token_classifier = pipeline(
    "token-classification",
    "dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")


from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

print(inputs["input_ids"].shape)
print(outputs.logits.shape)

import torch

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()
print(predictions)

model.config.id2label

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append(
            {
                "entity": label,
                "score": probabilities[idx][pred],
                "word": tokens[idx],
                "start": start,
                "end": end,
            }
        )

print(results)

### now group together to reproduce the second pipeline with aggregation

import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]
    if label != "O":
        # Remove the B- or I-
        label = label[2:]
        start, _ = offsets[idx]

        # Grab all the tokens labeled with I-label
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] == f"I-{label}"
        ):
            all_scores.append(probabilities[idx][pred])
            _, end = offsets[idx]
            idx += 1

        # The score is the mean of all the scores of the tokens in that grouped entity
        score = np.mean(all_scores).item()
        word = example[start:end]
        results.append(
            {
                "entity_group": label,
                "score": score,
                "word": word,
                "start": start,
                "end": end,
            }
        )
    idx += 1

print(results)


# Fast tokenizers in the QA pipeline
