import pandas as pd
from datasets import Dataset
import time
from pydantic import BaseModel, Field
from typing import List
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from research.llms.gemini.instructor_langfuse import instructor_parse_w_vertex
from langfuse.decorators import langfuse_context, observe
from tqdm import tqdm

# Load data
speeches20 = pd.read_pickle("~/projects/research/data/od_speeches20.pkl")
speeches20_sample = speeches20.sample(3000, random_state=42)

# split text
embedding_model = "distilbert-base-german-cased"
embedding_model = "chkla/parlbert-german-v1"
max_tokens = 128
tokenizer = Tokenizer.from_pretrained(embedding_model)
splitter = TextSplitter.from_huggingface_tokenizer(tokenizer, max_tokens)
speeches20_sample["chunks"] = speeches20_sample["speech_content"].apply(
    lambda x: splitter.chunks(x)
)

# remove first and last chunk for greeting and last statement
speeches20_sample["chunks"] = speeches20_sample["chunks"].apply(
    lambda x: x[1:-1]
)
speeches20_sample_exp = speeches20_sample.explode("chunks")
speeches20_sample_exp.dropna(subset=["chunks"], inplace=True)
speeches20_sample_exp.shape
# params = get_supported_openai_params(model=model_str)
# assert "response_format" in params


class SemanticQuery(BaseModel):
    queries: List[str] = Field(
        ...,
        description="""A list of fitting search queries""",
    )


@observe(
    name="labeling-od-v2"
)  # second wrapper so the output shows up in the trace obj on langfuse (and not just the generation)
def instructor_parse(messages, response_model, model_name, **model_params):
    # do some more setup
    langfuse_context.update_current_trace(session_id="second_run")
    return instructor_parse_w_vertex(
        messages, response_model, model_name, **model_params
    )


results = []
i = 23
for i in tqdm(range(7826, len(speeches20_sample_exp["chunks"]))):
    time.sleep(0.5)
    try:
        content = speeches20_sample_exp["chunks"].values[i]
        messages = [
            # {
            #     "role": "user",
            #     "content": "You are a machine learning expert who assists in creating high value training data for a semantic search model. Your role is to analyse the document chunk given to you and provide us with high quality potential queries",
            # },
            {
                "role": "user",
                "content": f"""
                                Take the following German text excerpt from a speech in the German parliament:
                                
                                {content}

                                Generate a list of German search queries for which the excerpt is a high-value answer. Focus on the essential points of the excerpt. Limit yourself to 3-5 words for the query and generate 1-3 differentiated queries.""",
            },
        ]

        resp = instructor_parse(messages, SemanticQuery, "gemini-2.0-flash-001")
        # print(content, resp)
        results.append(resp)
    except Exception as e:
        print(i)
        print(e)
        break


len(results)
# speeches20_sample_exp = speeches20_sample_exp.iloc[:1000].copy()
speeches20_sample_exp["queries"] = results
# speeches20_sample_exp.dropna(subset=["chunks"]).shape
speeches20_sample_exp["queries"] = speeches20_sample_exp["queries"].apply(
    lambda x: x.queries
)
speeches20_sample_exp = speeches20_sample_exp.explode("queries")


speeches20_sample_exp.to_pickle("speeches20_expl_sample.pkl")

df = pd.read_pickle("speeches20_expl_sample.pkl")
df.shape
df.rename(columns={"chunks": "anchor", "queries": "positive"}, inplace=True)
# df = df.explode("positive")

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
speeches_queries_dataset = Dataset.from_pandas(df)
# train - test - dev split
speeches_queries_dataset = speeches_queries_dataset.train_test_split(
    test_size=0.2
)
speeches_queries_dataset.push_to_hub("parl-synthetic-queries-v3")
