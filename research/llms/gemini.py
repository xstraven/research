import vertexai
from vertexai.generative_models import GenerativeModel
import os
import dotenv

dotenv.load_dotenv()
GCP_PROJECT = os.environ["GCP_PROJECT_ID"]
# TODO(developer): Update and un-comment below line
# PROJECT_ID = "your-project-id"
vertexai.init(project=GCP_PROJECT, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

response = model.generate_content(
    "What's a good name for a flower shop that specializes in selling bouquets of dried flowers?"
)

print(response.text)
# Example response:
# **Emphasizing the Dried Aspect:**
# * Everlasting Blooms
# * Dried & Delightful
# * The Petal Preserve
# ...
