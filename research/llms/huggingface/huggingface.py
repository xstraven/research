from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset

issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
issues_dataset

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset

issues_dataset.set_format("pandas")
df = issues_dataset[:]

df["comments"][0].tolist()

comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)

from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset

comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset


def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)
comments_dataset["comment_length"]


def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)


from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
model.to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape

embeddings_dataset = comments_dataset.map(
    lambda x: {
        "embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]
    }
)
embeddings_dataset

embeddings_dataset.add_faiss_index(column="embeddings")
