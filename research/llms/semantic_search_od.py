import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset
import instructor
import time
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel, Field
from typing import List
from vertexai.preview.tokenization import get_tokenizer_for_model

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
speeches20_expl_sample = speeches20_expl.sample(100, random_state=42)
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


client = instructor.from_vertexai(
    client=GenerativeModel(f"{model_str}"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
)


def get_training_queries(speech_snippet):
    time.sleep(0.5)  # to avoid rate limiting
    try:
        # note that client.chat.completions.create will also work
        # idea: give more context around the speech snippet
        resp = client.create(
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
            response_model=TrainingQueriesQuestions,
        )
    except Exception as e:
        print(e)
        resp = None
    finally:
        return resp


speeches20_expl_sample["training_queries"] = (
    speeches20_expl_sample.chunked_speech.apply(get_training_queries)
)
speeches20_expl_sample.training_queries

## TODO
### improve parsing model
### fine-tune parlbert model with sentence-transformers
