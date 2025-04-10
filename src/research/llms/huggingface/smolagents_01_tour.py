# !pip install smolagents[litellm]
from smolagents import CodeAgent, LiteLLMModel
import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

model = LiteLLMModel(
    model_id="anthropic/claude-3-5-sonnet-latest",
    api_key=ANTHROPIC_API_KEY,
)  # Could use 'gpt-4o'
agent = CodeAgent(tools=[], model=model, add_base_tools=True)

agent.run(
    "Could you give me the 118th number in the Fibonacci sequence?",
)


agent = CodeAgent(
    tools=[], model=model, additional_authorized_imports=["requests", "bs4"]
)
agent.run(
    "Could you get me the title of the page at url 'https://huggingface.co/blog'?"
)


from smolagents import ToolCallingAgent

agent = ToolCallingAgent(tools=[], model=model)
agent.run(
    "Could you get me the title of the page at url 'https://huggingface.co/blog'?"
)

print(agent.logs)
agent.write_memory_to_messages()


from smolagents import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()
print(search_tool("Who's the current president of Russia?"))


## define own tools
from huggingface_hub import list_models

task = "text-classification"

most_downloaded_model = next(
    iter(list_models(filter=task, sort="downloads", direction=-1))
)
print(most_downloaded_model.id)


from smolagents import tool


@tool
def model_download_tool(task: str) -> str:
    """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint.

    Args:
        task: The task for which to get the download count.
    """
    most_downloaded_model = next(
        iter(list_models(filter=task, sort="downloads", direction=-1))
    )
    return most_downloaded_model.id


from smolagents import Tool


class ModelDownloadTool(Tool):
    name = "model_download_tool"
    description = "This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub. It returns the name of the checkpoint."
    inputs = {
        "task": {
            "type": "string",
            "description": "The task for which to get the download count.",
        }
    }
    output_type = "string"

    def forward(self, task: str) -> str:
        most_downloaded_model = next(
            iter(list_models(filter=task, sort="downloads", direction=-1))
        )
        return most_downloaded_model.id


from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(tools=[model_download_tool], model=HfApiModel())
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub?"
)

# Multi-Agents
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool

model = HfApiModel()

web_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    name="web_search2",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = CodeAgent(tools=[], model=model, managed_agents=[web_agent])

manager_agent.run("Who is the CEO of Hugging Face?")


# visualize agents

from smolagents import load_tool, CodeAgent, HfApiModel, GradioUI

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True)

model = HfApiModel()

# Initialize the agent with the image generation tool
agent = CodeAgent(tools=[image_generation_tool], model=model)

GradioUI(agent).launch()

agent.push_to_hub("davhin/my_agent")
# agent.from_hub("m-ric/my_agent", trust_remote_code=True)
