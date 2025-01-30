import vertexai
from vertexai.generative_models import GenerativeModel
from pydantic import BaseModel
from langfuse.decorators import langfuse_context, observe
import instructor


vertexai.init(location="us-central1")


class User(BaseModel):
    name: str
    age: int


@observe(as_type="generation")
def instructor_parse_w_vertex(
    content,
    response_model,
    model_name,
    **model_params,
):
    client = instructor.from_vertexai(
        client=GenerativeModel(model_name),
        mode=instructor.Mode.VERTEXAI_TOOLS,
    )
    response = client.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        response_model=response_model,
        **model_params,
    )

    # pass model, model input, and usage metrics to Langfuse
    langfuse_context.update_current_observation(
        input=content,
        model=model_name,
        usage_details={  # instructor puts the original response under ._raw_response
            "input": response._raw_response.usage_metadata.prompt_token_count,
            "output": response._raw_response.usage_metadata.candidates_token_count,
            "total": response._raw_response.usage_metadata.total_token_count,
        },
        metadata={
            "model": model_name,
            "gen_config": model_params,
            "response_schema": response_model.model_json_schema(),
        },
    )
    return response


@observe()
def instructor_parse(content, response_model, model_name, **model_params):
    # do some more setup
    return instructor_parse_w_vertex(
        content, response_model, model_name, **model_params
    )


resp = instructor_parse(
    "Jason is 25 years old.", User, "gemini-1.5-pro-preview-0409"
)
resp
