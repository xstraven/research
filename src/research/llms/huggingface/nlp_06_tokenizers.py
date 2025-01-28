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
from transformers import pipeline

question_answerer = pipeline("question-answering")
context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
question_answerer(question=question, context=context)


## using a long context

long_context = """
🤗 Transformers: State of the Art NLP

🤗 Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction,
question answering, summarization, translation, text generation and more in over 100 languages.
Its aim is to make cutting-edge NLP easier to use for everyone.

🤗 Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and
then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and
can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question_answerer(question=question, context=long_context)


from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)


import torch

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None]

start_logits[mask] = -10000
end_logits[mask] = -10000

start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

scores = start_probabilities[:, None] * end_probabilities[None, :]
scores = torch.triu(scores)  # returns the upper triangular part of the matrix

max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
print(scores[start_index, end_index])

##  Try it out! Compute the start and end indices for the five most likely answers.
max_indeces = torch.topk(scores.flatten(), 5).indices
start_indeces = max_indeces // scores.shape[1]
end_indeces = max_indeces % scores.shape[1]
print(start_indeces, end_indeces)
##

inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

start_chars = [offsets[start_index][0] for start_index in start_indeces]
end_chars = [offsets[end_index][1] for end_index in end_indeces]
answers = [
    context[start_char:end_char]
    for start_char, end_char in zip(start_chars, end_chars)
]

## Try it out! Use the best scores you computed earlier to show the five most likely answers. To check your results, go back to the first pipeline and pass in top_k=5 when calling it.
result = {
    "answer": answers,
    "start": start_chars,
    "end": end_chars,
    "score": [
        scores[start_index, end_index]
        for start_index, end_index in zip(start_indeces, end_indeces)
    ],
}
print(result)
# top 5:
question_answerer(question=question, context=long_context, top_k=5)

# there are more tokens in the input than the question answerer pipeline can handle
inputs = tokenizer(question, long_context)
print(len(inputs["input_ids"]))

inputs = tokenizer(
    question, long_context, max_length=384, truncation="only_second"
)
print(tokenizer.decode(inputs["input_ids"]))


sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence,
    truncation=True,
    return_overflowing_tokens=True,
    max_length=6,
    stride=2,
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))

print(inputs.keys())
print(inputs["overflow_to_sample_mapping"])

sentences = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs = tokenizer(
    sentences,
    truncation=True,
    return_overflowing_tokens=True,
    max_length=6,
    stride=2,
)

print(inputs["overflow_to_sample_mapping"])

inputs = tokenizer(
    question,
    long_context,
    stride=128,
    max_length=384,
    padding="longest",
    truncation="only_second",
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)

outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
print(start_logits.shape, end_logits.shape)

sequence_ids = inputs.sequence_ids()
# Mask everything apart from the tokens of the context
mask = [i != 1 for i in sequence_ids]
# Unmask the [CLS] token
mask[0] = False
# Mask all the [PAD] tokens
mask = torch.logical_or(
    torch.tensor(mask)[None], (inputs["attention_mask"] == 0)
)

start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)

candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]
    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)

for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {
        "answer": answer,
        "start": start_char,
        "end": end_char,
        "score": score,
    }
    print(result)


# Normalization and pre-tokenization

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))
print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))
##  Try it out! Load a tokenizer from the bert-base-cased checkpoint and pass the same example to it. What are the main differences you can see between the cased and uncased versions of the tokenizer?
tokenizer2 = AutoTokenizer.from_pretrained("bert-base-cased")
print(tokenizer2.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
    "Hello, how are  you?"
)


# Byte-Pair Encoding tokenization
