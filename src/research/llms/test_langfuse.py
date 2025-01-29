import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

from langfuse.decorators import langfuse_context, observe
import os

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")


@observe(as_type="generation")
def vertex_generate_content(input, model_name="gemini-pro"):
    vertexai.init(project=GCP_PROJECT_ID, location="us-central1")
    model = GenerativeModel(model_name)
    response = model.generate_content(
        [input],
        generation_config={
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,
        },
    )

    # pass model, model input, and usage metrics to Langfuse
    langfuse_context.update_current_observation(
        input=input,
        model=model_name,
        usage_details={
            "input": response.usage_metadata.prompt_token_count,
            "output": response.usage_metadata.candidates_token_count,
            "total": response.usage_metadata.total_token_count,
        },
    )
    return response.candidates[0].content.parts[0].text


vertex_generate_content("The quick brown fox jumps over the lazy dog")
