# Now import transformers, datasets, etc
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import torch

print(f"MPS device available: {torch.mps.is_available()}")

# 1. Load a model to finetune with 2. (Optional) model card data
# model = SentenceTransformer(
#     "microsoft/mpnet-base",
#     model_card_data=SentenceTransformerModelCardData(
#         language="en",
#         license="apache-2.0",
#         model_name="MPNet base trained on AllNLI triplets",
#     ),
# )
model = SentenceTransformer("chkla/parlbert-german-v1")


# 3. Load a dataset to finetune on
dataset = load_dataset("davhin/parl-synthetic-queries-v2")
dataset = dataset.remove_columns("__index_level_0__")
train_dataset = dataset["train"].train_test_split(test_size=0.25, seed=300)[
    "train"
]
eval_dataset = dataset["train"].train_test_split(test_size=0.25, seed=300)[
    "test"
]
test_dataset = dataset["test"]

# 4. Define a loss function
loss = MultipleNegativesRankingLoss(model)


# 5. (Optional) Specify training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="models/parlbert-german-search",
    # Optional training parameters:
    use_mps_device=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=False,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=True,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="parlbert-german-search-v0",  # Will be used in W&B if `wandb` is installed
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="parlsearch-test-before-training",
)
print(dev_evaluator(model))

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the test set
test_evaluator = TripletEvaluator(
    anchors=test_dataset["anchor"],
    positives=test_dataset["positive"],
    negatives=test_dataset["negative"],
    name="parlsearch-test-after-training",
)
print(test_evaluator(model))

# 8. Save the trained model
model.save_pretrained("models/parlbert-german-search-v0")

# 9. (Optional) Push it to the Hugging Face Hub
model.push_to_hub("parlbert-german-search-v0")


## testing


embedder = SentenceTransformer("models/parlbert-german-search/final")

# Corpus with example sentences
corpus = test_dataset["anchor"][:100]
# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

queries = [
    "Zahlen Arbeitslosigkeit Entwicklung",
    "Wie viele Menschen sind arbeitslos?",
]
# queries_embeddings = embedder.encode(queries, convert_to_tensor=True)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(3, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[
        0
    ]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(corpus[idx], f"(Score: {score:.4f})")


for i in range(10):
    print(test_dataset["anchor"][i])
    print(test_dataset["positive"][i])
    print(test_dataset["negative"][i])
    print("\n\n")
