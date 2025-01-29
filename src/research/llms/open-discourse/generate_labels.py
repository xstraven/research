import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset
import instructor
import asyncio, nest_asyncio
import time
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel, Field
from typing import List
import os
from litellm.types.utils import ModelResponse
from litellm import completion_cost
from litellm import acompletion, completion
from litellm import get_supported_openai_params

nest_asyncio.apply()

os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
# Now import transformers, datasets, etc


from langfuse import Langfuse

langfuse = Langfuse(
    secret_key="sk-lf-60799cc8-63cc-4797-9208-2987310d992e",
    public_key="pk-lf-52844a24-0929-46a8-8489-df5824a0bf37",
    host="https://cloud.langfuse.com",
)


# from vertexai.preview.tokenization import get_tokenizer_for_model

PATH_TO_DATA = Path("~/projects/open-discourse/python/data/03_final")
SPEECHES = "speech_content.pkl"
speeches = pd.read_pickle(PATH_TO_DATA / SPEECHES)
speeches.head()
speeches20 = speeches.loc[speeches.electoral_term == 20]
# replace "({2})" with empty string
replace_regex = r"\(\{\d+\}\)"
speeches20.speech_content = speeches20.speech_content.apply(
    lambda x: re.sub(replace_regex, "", x)
)
test = speeches20.iloc[0].speech_content
# use regex to remove multiple spaces and newlines
speeches20.speech_content = speeches20.speech_content.apply(
    lambda x: re.sub(r"\s+", " ", x)
)

print(speeches20.iloc[0].speech_content)
# speech_dataset = Dataset.from_pandas(speeches20)
# speech_dataset

# from transformers import AutoConfig, AutoModel, AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
speeches20["chunked_speech"] = speeches20.speech_content.apply(
    lambda x: [x[i : i + 1024] for i in range(0, len(x), 1024)]
)
speeches20_expl = speeches20.explode("chunked_speech")
speeches20_expl.shape
vertexai.init()
model_str = "gemini-1.5-flash"
# tokenizer = get_tokenizer_for_model(model_str)
# # Count Tokens
# response = tokenizer.count_tokens(speech_snippet)
# print(response.total_tokens)


# class TrainingQueries(BaseModel):
#     """Training examples of relevant and non-relevant search queries for a semantic search model in German."""

#     relevant_search_queries: List[str] = Field(
#         ...,
#         description="German queries for which the provided excerpt is a relevant search result.",
#     )
#     nonrelevant_search_queries: List[str] = Field(
#         ...,
#         description="German queries for which the provided context is not a relevant search result.",
#     )


class TrainingQueriesQuestions(BaseModel):
    """Training examples of relevant queries and questions for a semantic search model in German."""

    queries: List[str] = Field(
        ...,
        description="A list of German queries the provided chunk is relevant to, in the context of a search engine.",
    )
    questions: List[str] = Field(
        ...,
        description="A list of German questions that could be answered by the provided chunk.",
    )


# client = instructor.from_vertexai(
#     client=GenerativeModel(f"{model_str}"),
#     mode=instructor.Mode.VERTEXAI_TOOLS,
# )

speeches20_expl_sample = speeches20_expl.sample(1000, random_state=42)


params = get_supported_openai_params(model="gemini-1.5-flash")
assert "response_format" in params


async def async_get_training_queries(speech_snippet, semaphore):
    async with semaphore:
        await asyncio.sleep(5 * 60 / 100)  # rate limit is 100 per minute
        try:
            resp = await acompletion(
                model="gemini-1.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": "You are a machine learning expert who assists in creating high value training data for a semantic search model. Your role is to analyse the document chunk given to you and provide us with high quality potential queries",
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Consider the following excerpt from a speech in the German Bundestag: {speech_snippet}
                        To generate training data for a semantic search model, please provide two sets of search queries in German:
                        1. Queries for which the provided context is a relevant search result.
                        2. Queries for which the provided context is not a relevant search result.
                        Make sure ALL queries make sense in the context of searching through parliamentary speeches.
                """,
                    },
                ],
                response_format=TrainingQueriesQuestions,
            )
        except Exception as e:
            print(e)
            resp = None
        finally:
            return resp


async def process_tasks(tasks):
    semaphore = asyncio.Semaphore(5)
    return await asyncio.gather(
        *[async_get_training_queries(task, semaphore) for task in tasks]
    )


import json

responses = []
for i in range(0, len(speeches20_expl_sample), 100):
    print(f"Processing chunk {i}")
    time.sleep(30)
    test = asyncio.run(
        process_tasks(speeches20_expl_sample.chunked_speech[i : i + 100])
    )
    responses.extend(test)
response = responses[0]
print(len(responses))
completion_cost = completion_cost(response)


for response in responses:
    try:
        json.loads(response.choices[0].message.content)
    except Exception as e:
        print(e)
        print(response.choices[0].message.content)
        break


def dump_model_from_response(response):
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as _:
        return None


speeches20_expl_sample["training_queries_response"] = responses
speeches20_expl_sample["training_queries"] = speeches20_expl_sample[
    "training_queries_response"
].apply(dump_model_from_response)

## check for None values
speeches20_expl_sample["training_queries"].isna().sum()
speeches20_expl_sample = speeches20_expl_sample.dropna(
    subset=["training_queries"]
)

## unfold the training queries
speeches20_expl_sample["training_queries_queries"] = speeches20_expl_sample[
    "training_queries"
].apply(lambda x: x["queries"])
speeches20_expl_sample["training_queries_questions"] = speeches20_expl_sample[
    "training_queries"
].apply(lambda x: x["questions"])

speeches20_expl_sample.to_pickle("speeches20_expl_sample.pkl")

speeches20_expl_sample = pd.read_pickle("speeches20_expl_sample.pkl")
speeches20_expl_sample.columns
df = speeches20_expl_sample.copy()
df["positive"] = df.apply(
    lambda x: x.training_queries_queries + x.training_queries_questions, axis=1
)
df.rename(columns={"chunked_speech": "anchor"}, inplace=True)
df = df.explode("positive")
df = df.sample(frac=1, random_state=42)
df["negative"] = df["positive"].shift(1)
df = df.dropna(subset=["negative"])
keep_cols = ["anchor", "positive", "negative"]
meta_cols = [
    "id",
    "electoral_term",
    "session",
    "first_name",
    "document_url",
    "last_name",
    "faction_id",
    "position_short",
    "position_long",
    "politician_id",
    "speech_content",
    "date",
]
df = df[keep_cols + meta_cols]
speeches_queries_dataset = Dataset.from_pandas(df[keep_cols])
# train - test - dev split
speeches_queries_dataset = speeches_queries_dataset.train_test_split(
    test_size=0.4
)
speeches_queries_dataset.push_to_hub("parl-synthetic-queries-v2")
