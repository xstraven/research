import instructor
import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel  # type: ignore
from pydantic import BaseModel, Field
from typing import List
from vertexai.preview.tokenization import get_tokenizer_for_model

vertexai.init()
model_str = "gemini-1.5-flash"
tokenizer = get_tokenizer_for_model(model_str)
# Count Tokens
prompt = "why is the sky blue?"
response = tokenizer.count_tokens(prompt)
print(response.total_tokens)


class TrainingQueries(BaseModel):
    relevant_search_queries: List[str] = Field(
        ...,
        description="Queries for which the provided context is a relevant search result.",
    )
    nonrelevant_search_queries: List[str] = Field(
        ...,
        description="Queries for which the provided context is not a relevant search result.",
    )


client = instructor.from_vertexai(
    client=GenerativeModel("gemini-1.5-pro-preview-0409"),
    mode=instructor.Mode.VERTEXAI_TOOLS,
)

# note that client.chat.completions.create will also work
# idea: give more context around the speech snippet
resp = client.create(
    messages=[
        {
            "role": "system",
            "content": "You are a machine learning expert who assists in generating training data for a semantic search model.",
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
    response_model=TrainingQueries,
)
