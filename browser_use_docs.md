# Introduction - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Get Started

Introduction

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Get Started

# Introduction

Welcome to Browser Use - We enable AI to control your browser

## [​](#overview) Overview

Browser Use is the easiest way to connect your AI agents with the browser. It makes websites accessible for AI agents by providing a powerful, yet simple interface for browser automation.

If you have used Browser Use for your project, feel free to show it off in our
[Discord community](https://link.browser-use.com/discord)!

## [​](#getting-started) Getting Started

[## Quick Start

Get up and running with Browser Use in minutes](/quickstart)[## Supported Models

Configure different LLMs for your agents](/customize/supported-models)[## Agent Settings

Learn how to configure and customize your agents](/customize/agent-settings)[## Custom Functions

Extend functionality with custom actions](/customize/custom-functions)

## [​](#fancy-demos) Fancy Demos

### [​](#writing-in-google-docs) Writing in Google Docs

Task: Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

### [​](#job-applications) Job Applications

Task: Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs.

[](https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04)

### [​](#flight-search) Flight Search

Task: Find flights on kayak.com from Zurich to Beijing.

### [​](#data-collection) Data Collection

Task: Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging Face, save top 5 to file.

[](https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3)

## [​](#community-%26-support) Community & Support

[## Join Discord

Join our community for support and showcases](https://link.browser-use.com/discord)[## GitHub

Star us on GitHub and contribute to development](https://github.com/browser-use/browser-use)

Browser Use is MIT licensed and actively maintained. We welcome contributions
and feedback from the community!

Was this page helpful?

YesNo

[Quickstart](/quickstart)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Overview](#overview)
* [Getting Started](#getting-started)
* [Fancy Demos](#fancy-demos)
* [Writing in Google Docs](#writing-in-google-docs)
* [Job Applications](#job-applications)
* [Flight Search](#flight-search)
* [Data Collection](#data-collection)
* [Community & Support](#community-%26-support)

---

# Quickstart - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Get Started

Quickstart

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Get Started

# Quickstart

Start using Browser Use with this quickstart guide

## [​](#prepare-the-environment) Prepare the environment

Browser Use requires Python 3.11 or higher.

First, we recommend using [uv](https://docs.astral.sh/uv/) to setup the Python environment.

```
uv venv --python 3.11

```

and activate it with:

```
# For Mac/Linux:
source .venv/bin/activate

# For Windows:
.venv\Scripts\activate

```

Install the dependencies:

```
uv pip install browser-use

```

Then install playwright:

```
uv run playwright install

```

## [​](#create-an-agent) Create an agent

Then you can use the agent as follows:

agent.py

```
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
load_dotenv()

import asyncio

llm = ChatOpenAI(model="gpt-4o")

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())

```

## [​](#set-up-your-llm-api-keys) Set up your LLM API keys

`ChatOpenAI` and other Langchain chat models require API keys. You should store these in your `.env` file. For example, for OpenAI and Anthropic, you can set the API keys in your `.env` file, such as:

.env

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

```

For other LLM models you can refer to the [Langchain documentation](https://python.langchain.com/docs/integrations/chat/) to find how to set them up with their specific API keys.

Was this page helpful?

YesNo

[Introduction](/introduction)[Supported Models](/customize/supported-models)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Prepare the environment](#prepare-the-environment)
* [Create an agent](#create-an-agent)
* [Set up your LLM API keys](#set-up-your-llm-api-keys)

---

# Supported Models - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Supported Models

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Supported Models

Guide to using different LangChain chat models with Browser Use

## [​](#overview) Overview

Browser Use supports various LangChain chat models. Here's how to configure and use the most popular ones. The full list is available in the [LangChain documentation](https://python.langchain.com/docs/integrations/chat/).

## [​](#model-recommendations) Model Recommendations

We have yet to test performance across all models. Currently, we achieve the best results using GPT-4o with an 89% accuracy on the [WebVoyager Dataset](https://browser-use.com/posts/sota-technical-report). DeepSeek-V3 is 30 times cheaper than GPT-4o. Gemini-2.0-exp is also gaining popularity in the community because it is currently free.
We also support local models, like Qwen 2.5, but be aware that small models often return the wrong output structure-which lead to parsing errors. We believe that local models will improve significantly this year.

All models require their respective API keys. Make sure to set them in your
environment variables before running the agent.

## [​](#supported-models) Supported Models

All LangChain chat models, which support tool-calling are available. We will document the most popular ones here.

### [​](#openai) OpenAI

OpenAI's GPT-4o models are recommended for best performance.

```
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Initialize the model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)

```

Required environment variables:

.env

```
OPENAI_API_KEY=

```

### [​](#anthropic) Anthropic

```
from langchain_anthropic import ChatAnthropic
from browser_use import Agent

# Initialize the model
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.0,
    timeout=100, # Increase for complex tasks
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)

```

And add the variable:

.env

```
ANTHROPIC_API_KEY=

```

### [​](#azure-openai) Azure OpenAI

```
from langchain_openai import AzureChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
import os

# Initialize the model
llm = AzureChatOpenAI(
    model="gpt-4o",
    api_version='2024-10-21',
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
    api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)

```

Required environment variables:

.env

```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=

```

### [​](#gemini) Gemini

> [!IMPORTANT]
> `GEMINI_API_KEY` was the old environment var name, it should be called `GOOGLE_API_KEY` as of 2025-05.

```
from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from dotenv import load_dotenv

# Read GOOGLE_API_KEY into env
load_dotenv()

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)

```

Required environment variables:

.env

```
GOOGLE_API_KEY=

```

### [​](#deepseek-v3) DeepSeek-V3

The community likes DeepSeek-V3 for its low price, no rate limits, open-source nature, and good performance.
The example is available [here](https://github.com/browser-use/browser-use/blob/main/examples/models/deepseek.py).

```
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Initialize the model
llm=ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-chat', api_key=SecretStr(api_key))

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm,
    use_vision=False
)

```

Required environment variables:

.env

```
DEEPSEEK_API_KEY=

```

### [​](#deepseek-r1) DeepSeek-R1

We support DeepSeek-R1. Its not fully tested yet, more and more functionality will be added, like e.g. the output of it'sreasoning content.
The example is available [here](https://github.com/browser-use/browser-use/blob/main/examples/models/deepseek-r1.py).
It does not support vision. The model is open-source so you could also use it with Ollama, but we have not tested it.

```
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# Initialize the model
llm=ChatOpenAI(base_url='https://api.deepseek.com/v1', model='deepseek-reasoner', api_key=SecretStr(api_key))

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm,
    use_vision=False
)

```

Required environment variables:

.env

```
DEEPSEEK_API_KEY=

```

### [​](#ollama) Ollama

Many users asked for local models. Here they are.

1. Download Ollama from [here](https://ollama.ai/download)
2. Run `ollama pull model_name`. Pick a model which supports tool-calling from [here](https://ollama.com/search?c=tools)
3. Run `ollama start`

```
from langchain_ollama import ChatOllama
from browser_use import Agent
from pydantic import SecretStr


# Initialize the model
llm=ChatOllama(model="qwen2.5", num_ctx=32000)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm
)

```

Required environment variables: None!

### [​](#novita-ai) Novita AI

[Novita AI](https://novita.ai) is an LLM API provider that offers a wide range of models. Note: choose a model that supports function calling.

```
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("NOVITA_API_KEY")

# Initialize the model
llm = ChatOpenAI(base_url='https://api.novita.ai/v3/openai', model='deepseek/deepseek-v3-0324', api_key=SecretStr(api_key))

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm,
    use_vision=False
)

```

Required environment variables:

.env

```
NOVITA_API_KEY=

```

### [​](#x-ai) X AI

[X AI](https://x.ai) is an LLM API provider that offers a wide range of models. Note: choose a model that supports function calling.

```
from langchain_openai import ChatOpenAI
from browser_use import Agent
from pydantic import SecretStr
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROK_API_KEY")

# Initialize the model
llm = ChatOpenAI(
    base_url='https://api.x.ai/v1',
    model='grok-3-beta',
    api_key=SecretStr(api_key)
)

# Create agent with the model
agent = Agent(
    task="Your task here",
    llm=llm,
    use_vision=False
)

```

Required environment variables:

.env

```
GROK_API_KEY=

```

## [​](#coming-soon) Coming soon

(We are working on it)

* Groq
* Github
* Fine-tuned models

Was this page helpful?

YesNo

[Quickstart](/quickstart)[Agent Settings](/customize/agent-settings)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Overview](#overview)
* [Model Recommendations](#model-recommendations)
* [Supported Models](#supported-models)
* [OpenAI](#openai)
* [Anthropic](#anthropic)
* [Azure OpenAI](#azure-openai)
* [Gemini](#gemini)
* [DeepSeek-V3](#deepseek-v3)
* [DeepSeek-R1](#deepseek-r1)
* [Ollama](#ollama)
* [Novita AI](#novita-ai)
* [X AI](#x-ai)
* [Coming soon](#coming-soon)

---

# Agent Settings - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Agent Settings

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Agent Settings

Learn how to configure the agent

## [​](#overview) Overview

The `Agent` class is the core component of Browser Use that handles browser automation. Here are the main configuration options you can use when initializing an agent.

## [​](#basic-settings) Basic Settings

```
from browser_use import Agent
from langchain_openai import ChatOpenAI

agent = Agent(
    task="Find the contact email address and phone number for {target}. "
    "Search multiple sources (official website, LinkedIn, Wikipedia, etc.) if needed. "
    "Return only these details in the format:\n"
    "Email: <email or 'Not found'>\n"
    "Phone: <phone or 'Not found'>",
    llm=ChatOpenAI(model="gpt-4o"),
)

```

### [​](#required-parameters) Required Parameters

* `task`: The instruction for the agent to execute
* `llm`: A LangChain chat model instance. See [LangChain Models](/customize/supported-models) for supported models.

## [​](#agent-behavior) Agent Behavior

Control how the agent operates:

```
agent = Agent(
    task="your task",
    llm=llm,
    controller=custom_controller,  # For custom tool calling
    use_vision=True,              # Enable vision capabilities
    save_conversation_path="logs/conversation"  # Save chat logs
)

```

### [​](#behavior-parameters) Behavior Parameters

* `controller`: Registry of functions the agent can call. Defaults to base Controller. See [Custom Functions](/customize/custom-functions) for details.
* `use_vision`: Enable/disable vision capabilities. Defaults to `True`.
  + When enabled, the model processes visual information from web pages
  + Disable to reduce costs or use models without vision support
  + For GPT-4o, image processing costs approximately 800-1000 tokens (~$0.002 USD) per image (but this depends on the defined screen size)
* `save_conversation_path`: Path to save the complete conversation history. Useful for debugging.
* `override_system_message`: Completely replace the default system prompt with a custom one.
* `extend_system_message`: Add additional instructions to the default system prompt.

Vision capabilities are recommended for better web interaction understanding,
but can be disabled to reduce costs or when using models without vision
support.

## [​](#reuse-browser-configuration) (Reuse) Browser Configuration

You can configure how the agent interacts with the browser. To see more `Browser` options refer to the [Browser Settings](/customize/browser-settings) documentation.

### [​](#reuse-existing-browser) Reuse Existing Browser

`browser`: A Browser Use Browser instance. When provided, the agent will reuse this browser instance and automatically create new contexts for each `run()`.

```
from browser_use import Agent, Browser
from browser_use.browser.context import BrowserContext

# Reuse existing browser
browser = Browser()
agent = Agent(
    task=task1,
    llm=llm,
    browser=browser  # Browser instance will be reused
)

await agent.run()

# Manually close the browser
await browser.close()

```

Remember: in this scenario the `Browser` will not be closed automatically.

### [​](#reuse-existing-browser-context) Reuse Existing Browser Context

`browser_context`: A Playwright browser context. Useful for maintaining persistent sessions. See [Persistent Browser](/customize/persistent-browser) for more details.

```
from browser_use import Agent, Browser
from playwright.async_api import BrowserContext

# Use specific browser context (preferred method)
async with await browser.new_context() as context:
    agent = Agent(
        task=task2,
        llm=llm,
        browser_context=context  # Use persistent context
    )

    # Run the agent
    await agent.run()

    # Pass the context to the next agent
    next_agent = Agent(
        task=task2,
        llm=llm,
        browser_context=context
    )

    ...

await browser.close()

```

For more information about how browser context works, refer to the [Playwright
documentation](https://playwright.dev/docs/api/class-browsercontext).

You can reuse the same context for multiple agents. If you do nothing, the
browser will be automatically created and closed on `run()` completion.

## [​](#running-the-agent) Running the Agent

The agent is executed using the async `run()` method:

* `max_steps` (default: `100`)
  Maximum number of steps the agent can take during execution. This prevents infinite loops and helps control execution time.

## [​](#agent-history) Agent History

The method returns an `AgentHistoryList` object containing the complete execution history. This history is invaluable for debugging, analysis, and creating reproducible scripts.

```
# Example of accessing history
history = await agent.run()

# Access (some) useful information
history.urls()              # List of visited URLs
history.screenshots()       # List of screenshot paths
history.action_names()      # Names of executed actions
history.extracted_content() # Content extracted during execution
history.errors()           # Any errors that occurred
history.model_actions()     # All actions with their parameters

```

The `AgentHistoryList` provides many helper methods to analyze the execution:

* `final_result()`: Get the final extracted content
* `is_done()`: Check if the agent completed successfully
* `has_errors()`: Check if any errors occurred
* `model_thoughts()`: Get the agent's reasoning process
* `action_results()`: Get results of all actions

For a complete list of helper methods and detailed history analysis
capabilities, refer to the [AgentHistoryList source
code](https://github.com/browser-use/browser-use/blob/main/browser_use/agent/views.py#L111).

## [​](#run-initial-actions-without-llm) Run initial actions without LLM

With [this example](https://github.com/browser-use/browser-use/blob/main/examples/features/initial_actions.py) you can run initial actions without the LLM.
Specify the action as a dictionary where the key is the action name and the value is the action parameters. You can find all our actions in the [Controller](https://github.com/browser-use/browser-use/blob/main/browser_use/controller/service.py) source code.

```

initial_actions = [
	{'open_tab': {'url': 'https://www.google.com'}},
	{'open_tab': {'url': 'https://en.wikipedia.org/wiki/Randomness'}},
	{'scroll_down': {'amount': 1000}},
]
agent = Agent(
	task='What theories are displayed on the page?',
	initial_actions=initial_actions,
	llm=llm,
)

```

## [​](#run-with-message-context) Run with message context

You can configure the agent and provide a separate message to help the LLM understand the task better.

```
from langchain_openai import ChatOpenAI

agent = Agent(
    task="your task",
    message_context="Additional information about the task",
    llm = ChatOpenAI(model='gpt-4o')
)

```

## [​](#run-with-planner-model) Run with planner model

You can configure the agent to use a separate planner model for high-level task planning:

```
from langchain_openai import ChatOpenAI

# Initialize models
llm = ChatOpenAI(model='gpt-4o')
planner_llm = ChatOpenAI(model='o3-mini')

agent = Agent(
    task="your task",
    llm=llm,
    planner_llm=planner_llm,           # Separate model for planning
    use_vision_for_planner=False,      # Disable vision for planner
    planner_interval=4                 # Plan every 4 steps
)

```

### [​](#planner-parameters) Planner Parameters

* `planner_llm`: A LangChain chat model instance used for high-level task planning. Can be a smaller/cheaper model than the main LLM.
* `use_vision_for_planner`: Enable/disable vision capabilities for the planner model. Defaults to `True`.
* `planner_interval`: Number of steps between planning phases. Defaults to `1`.

Using a separate planner model can help:

* Reduce costs by using a smaller model for high-level planning
* Improve task decomposition and strategic thinking
* Better handle complex, multi-step tasks

The planner model is optional. If not specified, the agent will not use the planner model.

### [​](#optional-parameters) Optional Parameters

* `message_context`: Additional information about the task to help the LLM understand the task better.
* `initial_actions`: List of initial actions to run before the main task.
* `max_actions_per_step`: Maximum number of actions to run in a step. Defaults to `10`.
* `max_failures`: Maximum number of failures before giving up. Defaults to `3`.
* `retry_delay`: Time to wait between retries in seconds when rate limited. Defaults to `10`.
* `generate_gif`: Enable/disable GIF generation. Defaults to `False`. Set to `True` or a string path to save the GIF.

## [​](#memory-management) Memory Management

Browser Use includes a procedural memory system using [Mem0](https://mem0.ai) that automatically summarizes the agent's conversation history at regular intervals to optimize context window usage during long tasks.

```
from browser_use.agent.memory import MemoryConfig

agent = Agent(
    task="your task",
    llm=llm,
    enable_memory=True,
    memory_config=MemoryConfig(
        agent_id="my_custom_agent",
        memory_interval=15
    )
)

```

### [​](#memory-parameters) Memory Parameters

* `enable_memory`: Enable/disable the procedural memory system. Defaults to `True`.
* `memory_config`: A `MemoryConfig` Pydantic model instance (required). Dictionary format is not supported.

### [​](#using-memoryconfig) Using MemoryConfig

You must configure the memory system using the `MemoryConfig` Pydantic model for a type-safe approach:

```
from browser_use.agent.memory import MemoryConfig

agent = Agent(
    task=task_description,
    llm=llm,
    memory_config=MemoryConfig(
        agent_id="my_agent",
        memory_interval=15,
        embedder_provider="openai",
        embedder_model="text-embedding-3-large",
        embedder_dims=1536,
    )
)

```

The `MemoryConfig` model provides these configuration options:

#### [​](#memory-settings) Memory Settings

* `agent_id`: Unique identifier for the agent (default: `"browser_use_agent"`)
* `memory_interval`: Number of steps between memory summarization (default: `10`)

#### [​](#embedder-settings) Embedder Settings

* `embedder_provider`: Provider for embeddings (`'openai'`, `'gemini'`, `'ollama'`, or `'huggingface'`)
* `embedder_model`: Model name for the embedder
* `embedder_dims`: Dimensions for the embeddings

#### [​](#vector-store-settings) Vector Store Settings

* `vector_store_provider`: Provider for vector storage (currently only `'faiss'` is supported)
* `vector_store_base_path`: Path for storing vector data (e.g. /tmp/mem0)

The model automatically sets appropriate defaults based on the LLM being used:

* For `ChatOpenAI`: Uses OpenAI's `text-embedding-3-small` embeddings
* For `ChatGoogleGenerativeAI`: Uses Gemini's `models/text-embedding-004` embeddings
* For `ChatOllama`: Uses Ollama's `nomic-embed-text` embeddings
* Default: Uses Hugging Face's `all-MiniLM-L6-v2` embeddings

Always pass a properly constructed `MemoryConfig` object to the `memory_config` parameter.
Dictionary-based configuration is no longer supported.

### [​](#how-memory-works) How Memory Works

When enabled, the agent periodically compresses its conversation history into concise summaries:

1. Every `memory_interval` steps, the agent reviews its recent interactions
2. It creates a procedural memory summary using the same LLM as the agent
3. The original messages are replaced with the summary, reducing token usage
4. This process helps maintain important context while freeing up the context window

### [​](#disabling-memory) Disabling Memory

If you want to disable the memory system (for debugging or for shorter tasks), set `enable_memory` to `False`:

```
agent = Agent(
    task="your task",
    llm=llm,
    enable_memory=False
)

```

Disabling memory may be useful for debugging or short tasks, but for longer
tasks, it can lead to context window overflow as the conversation history
grows. The memory system helps maintain performance during extended sessions.

Was this page helpful?

YesNo

[Supported Models](/customize/supported-models)[Browser Settings](/customize/browser-settings)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Overview](#overview)
* [Basic Settings](#basic-settings)
* [Required Parameters](#required-parameters)
* [Agent Behavior](#agent-behavior)
* [Behavior Parameters](#behavior-parameters)
* [(Reuse) Browser Configuration](#reuse-browser-configuration)
* [Reuse Existing Browser](#reuse-existing-browser)
* [Reuse Existing Browser Context](#reuse-existing-browser-context)
* [Running the Agent](#running-the-agent)
* [Agent History](#agent-history)
* [Run initial actions without LLM](#run-initial-actions-without-llm)
* [Run with message context](#run-with-message-context)
* [Run with planner model](#run-with-planner-model)
* [Planner Parameters](#planner-parameters)
* [Optional Parameters](#optional-parameters)
* [Memory Management](#memory-management)
* [Memory Parameters](#memory-parameters)
* [Using MemoryConfig](#using-memoryconfig)
* [Memory Settings](#memory-settings)
* [Embedder Settings](#embedder-settings)
* [Vector Store Settings](#vector-store-settings)
* [How Memory Works](#how-memory-works)
* [Disabling Memory](#disabling-memory)

---

# Browser Settings - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Browser Settings

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Browser Settings

Configure browser behavior and context settings

Browser Use allows you to customize the browser's behavior through two main configuration classes: `BrowserConfig` and `BrowserContextConfig`. These settings control everything from headless mode to proxy settings and page load behavior.

We are currently working on improving how browser contexts are managed. The
system will soon transition to a "1 agent, 1 browser, 1 context" model for
better stability and developer experience.

# [​](#browser-configuration) Browser Configuration

The `BrowserConfig` class controls the core browser behavior and connection settings.

```
from browser_use import BrowserConfig

# Basic configuration
config = BrowserConfig(
    headless=False,
    disable_security=False
)

browser = Browser(config=config)

agent = Agent(
    browser=browser,
    # ...
)

```

## [​](#core-settings) Core Settings

* **headless** (default: `False`)
  Runs the browser without a visible UI. Note that some websites may detect headless mode.
* **disable\_security** (default: `False`)
  Disables browser security features. While this can fix certain functionality issues (like cross-site iFrames), it should be used cautiously, especially when visiting untrusted websites.
* **keep\_alive** (default: `False`)
  Keeps the browser alive after the agent has finished running. This is useful when you need to run multiple tasks with the same browser instance.

### [​](#additional-settings) Additional Settings

* **extra\_browser\_args** (default: `[]`)
  Additional arguments are passed to the browser at launch. See the [full list of available arguments](https://github.com/browser-use/browser-use/blob/main/browser_use/browser/browser.py#L180).
* **proxy** (default: `None`)
  Standard Playwright proxy settings for using external proxy services.
* **new\_context\_config** (default: `BrowserContextConfig()`)
  Default settings for new browser contexts. See Context Configuration below.

For web scraping tasks on sites that restrict automated access, we recommend
using external browser or proxy providers for better reliability.

## [​](#alternative-initialization) Alternative Initialization

These settings allow you to connect to external browser providers or use a local Chrome instance.

### [​](#external-browser-provider-wss) External Browser Provider (wss)

Connect to cloud-based browser services for enhanced reliability and proxy capabilities.

```
config = BrowserConfig(
    wss_url="wss://your-browser-provider.com/ws"
)

```

* **wss\_url** (default: `None`)
  WebSocket URL for connecting to external browser providers (e.g., [anchorbrowser.io](https://anchorbrowser.io), steel.dev, browserbase.com, browserless.io, [TestingBot](https://testingbot.com/support/ai/integrations/browser-use)).

This overrides local browser settings and uses the provider's configuration.
Refer to their documentation for settings.

### [​](#external-browser-provider-cdp) External Browser Provider (cdp)

Connect to cloud or local Chrome instances using Chrome DevTools Protocol (CDP) for use with tools like `headless-shell` or `browserless`.

```
config = BrowserConfig(
    cdp_url="http://localhost:9222"
)

```

* **cdp\_url** (default: `None`)
  URL for connecting to a Chrome instance via CDP. Commonly used for debugging or connecting to locally running Chrome instances.

### [​](#local-chrome-instance-binary) Local Chrome Instance (binary)

Connect to your existing Chrome installation to access saved states and cookies.

```
config = BrowserConfig(
    browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
)

```

* **browser\_binary\_path** (default: `None`)
  Path to connect to an existing Browser installation. Particularly useful for workflows requiring existing login states or browser preferences.

This will overwrite other browser settings.

# [​](#context-configuration) Context Configuration

The `BrowserContextConfig` class controls settings for individual browser contexts.

```
from browser_use.browser.context import BrowserContextConfig

config = BrowserContextConfig(
    cookies_file="path/to/cookies.json",
    wait_for_network_idle_page_load_time=3.0,
    window_width=1280,
    window_height=1100,
    locale='en-US',
    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36',
    highlight_elements=True,
    viewport_expansion=500,
    allowed_domains=['google.com', 'wikipedia.org'],
)

browser = Browser()
context = BrowserContext(browser=browser, config=config)


async def run_search():
	agent = Agent(
		browser_context=context,
		task='Your task',
		llm=llm)

```

## [​](#configuration-options) Configuration Options

### [​](#page-load-settings) Page Load Settings

* **minimum\_wait\_page\_load\_time** (default: `0.5`)
  Minimum time to wait before capturing page state for LLM input.
* **wait\_for\_network\_idle\_page\_load\_time** (default: `1.0`)
  Time to wait for network activity to cease. Increase to 3-5s for slower websites. This tracks essential content loading, not dynamic elements like videos.
* **maximum\_wait\_page\_load\_time** (default: `5.0`)
  Maximum time to wait for page load before proceeding.

### [​](#display-settings) Display Settings

* **window\_width** (default: `1280`) and **window\_height** (default: `1100`)
  Browser window dimensions. The default size is optimized for general use cases and interaction with common UI elements like cookie banners.
* **locale** (default: `None`)
  Specify user locale, for example en-GB, de-DE, etc. Locale will affect the navigator. Language value, Accept-Language request header value as well as number and date formatting rules. If not provided, defaults to the system default locale.
* **highlight\_elements** (default: `True`)
  Highlight interactive elements on the screen with colorful bounding boxes.
* **viewport\_expansion** (default: `500`)
  Viewport expansion in pixels. With this you can control how much of the page is included in the context of the LLM. Setting this parameter controls the highlighting of elements:

  + `-1`: All elements from the entire page will be included, regardless of visibility (highest token usage but most complete).
  + `0`: Only elements which are currently visible in the viewport will be included.
  + `500` (default): Elements in the viewport plus an additional 500 pixels in each direction will be included, providing a balance between context and token usage.

### [​](#restrict-urls) Restrict URLs

* **allowed\_domains** (default: `None`)
  List of allowed domains that the agent can access. If None, all domains are allowed.
  Example: ['google.com', '\*.wikipedia.org'] - Here the agent will only be able to access `google.com` exactly and `wikipedia.org` + `*.wikipedia.org`.

  Glob patterns are supported:

  + `['example.com']` ✅ will match only `example.com` exactly, subdomains will not be allowed.
    It's always the most secure to list all the domains you want to give the access to explicitly e.g.
    `['google.com', 'www.google.com', 'myaccount.google.com', 'mail.google.com', 'docs.google.com']`
  + `['*.example.com']` ⚠️ **CAUTION** this will match `example.com` and *all* subdomains.
    Make sure *all* the subdomains are safe for the agent! `abc.example.com`, `def.example.com`, …, `useruploads.example.com`, `admin.example.com`
  + `['*google.com']` ❌ **DON'T DO THIS**, it will match any domains that end in `google.com`, *including `evilgoogle.com`*
  + `['*.google.*']` ❌ **DON'T DO THIS**, it will match `google.com`, `google.co.uk`, `google.fr`, etc. *but also `www.google.evil.com`*

### [​](#session-management) Session Management

* **keep\_alive** (default: `False`)
  Keeps the browser context (tab/session) alive after an agent task has completed. This is useful for maintaining session state across multiple tasks.

### [​](#debug-and-recording) Debug and Recording

* **save\_recording\_path** (default: `None`)
  Directory path for saving video recordings.
* **trace\_path** (default: `None`)
  Directory path for saving trace files. Files are automatically named as `{trace_path}/{context_id}.zip`.
* **save\_playwright\_script\_path** (default: `None`)
  BETA: Filename to save a replayable playwright python script to containing the steps the agent took.

Was this page helpful?

YesNo

[Agent Settings](/customize/agent-settings)[Connect to your Browser](/customize/real-browser)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Browser Configuration](#browser-configuration)
* [Core Settings](#core-settings)
* [Additional Settings](#additional-settings)
* [Alternative Initialization](#alternative-initialization)
* [External Browser Provider (wss)](#external-browser-provider-wss)
* [External Browser Provider (cdp)](#external-browser-provider-cdp)
* [Local Chrome Instance (binary)](#local-chrome-instance-binary)
* [Context Configuration](#context-configuration)
* [Configuration Options](#configuration-options)
* [Page Load Settings](#page-load-settings)
* [Display Settings](#display-settings)
* [Restrict URLs](#restrict-urls)
* [Session Management](#session-management)
* [Debug and Recording](#debug-and-recording)

---

# Connect to your Browser - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Connect to your Browser

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Connect to your Browser

With this you can connect to your real browser, where you are logged in with all your accounts.

## [​](#overview) Overview

You can connect the agent to your real Chrome browser instance, allowing it to access your existing browser profile with all your logged-in accounts and settings. This is particularly useful when you want the agent to interact with services where you're already authenticated.

First make sure to close all running Chrome instances.

## [​](#basic-configuration) Basic Configuration

To connect to your real Chrome browser, you'll need to specify the path to your Chrome executable when creating the Browser instance:

```
from browser_use import Agent, Browser, BrowserConfig
from langchain_openai import ChatOpenAI
import asyncio
# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Create the agent with your configured browser
agent = Agent(
    task="Your task here",
    llm=ChatOpenAI(model='gpt-4o'),
    browser=browser,
)

async def main():
    await agent.run()

    input('Press Enter to close the browser...')
    await browser.close()

if __name__ == '__main__':
    asyncio.run(main())

```

When using your real browser, the agent will have access to all your logged-in sessions. Make sure to ALWAYS review the task you're giving to the agent and ensure it aligns with your security requirements!

Was this page helpful?

YesNo

[Browser Settings](/customize/browser-settings)[Output Format](/customize/output-format)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Overview](#overview)
* [Basic Configuration](#basic-configuration)

---

# Output Format - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Output Format

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Output Format

The default is text. But you can define a structured output format to make post-processing easier.

## [​](#custom-output-format) Custom output format

With [this example](https://github.com/browser-use/browser-use/blob/main/examples/features/custom_output.py) you can define what output format the agent should return to you.

```
from pydantic import BaseModel
# Define the output format as a Pydantic model
class Post(BaseModel):
	post_title: str
	post_url: str
	num_comments: int
	hours_since_post: int


class Posts(BaseModel):
	posts: List[Post]


controller = Controller(output_model=Posts)


async def main():
	task = 'Go to hackernews show hn and give me the first  5 posts'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: Posts = Posts.model_validate_json(result)

		for post in parsed.posts:
			print('\n--------------------------------')
			print(f'Title:            {post.post_title}')
			print(f'URL:              {post.post_url}')
			print(f'Comments:         {post.num_comments}')
			print(f'Hours since post: {post.hours_since_post}')
	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())

```

Was this page helpful?

YesNo

[Connect to your Browser](/customize/real-browser)[System Prompt](/customize/system-prompt)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Custom output format](#custom-output-format)

---

# System Prompt - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

System Prompt

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# System Prompt

Customize the system prompt to control agent behavior and capabilities

## [​](#overview) Overview

You can customize the system prompt in two ways:

1. Extend the default system prompt with additional instructions
2. Override the default system prompt entirely

Custom system prompts allow you to modify the agent's behavior at a
fundamental level. Use this feature carefully as it can significantly impact
the agent's performance and reliability.

### [​](#extend-system-prompt-recommended) Extend System Prompt (recommended)

To add additional instructions to the default system prompt:

```
extend_system_message = """
REMEMBER the most important RULE:
ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!
"""

```

### [​](#override-system-prompt) Override System Prompt

Not recommended! If you must override the [default system
prompt](https://github.com/browser-use/browser-use/blob/main/browser_use/agent/system_prompt.md),
make sure to test the agent yourself.

Anyway, to override the default system prompt:

```
# Define your complete custom prompt
override_system_message = """
You are an AI agent that helps users with web browsing tasks.

[Your complete custom instructions here...]
"""

# Create agent with custom system prompt
agent = Agent(
    task="Your task here",
    llm=ChatOpenAI(model='gpt-4'),
    override_system_message=override_system_message
)

```

### [​](#extend-planner-system-prompt) Extend Planner System Prompt

You can customize the behavior of the planning agent by extending its system prompt:

```
extend_planner_system_message = """
PRIORITIZE gathering information before taking any action.
Always suggest exploring multiple options before making a decision.
"""

# Create agent with extended planner system prompt
llm = ChatOpenAI(model='gpt-4o')
planner_llm = ChatOpenAI(model='gpt-4o-mini')

agent = Agent(
	task="Your task here",
	llm=llm,
	planner_llm=planner_llm,
	extend_planner_system_message=extend_planner_system_message
)

```

Was this page helpful?

YesNo

[Output Format](/customize/output-format)[Sensitive Data](/customize/sensitive-data)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Overview](#overview)
* [Extend System Prompt (recommended)](#extend-system-prompt-recommended)
* [Override System Prompt](#override-system-prompt)
* [Extend Planner System Prompt](#extend-planner-system-prompt)

---

# Sensitive Data - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Sensitive Data

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Sensitive Data

Handle sensitive information securely by preventing the model from seeing actual passwords.

## [​](#handling-sensitive-data) Handling Sensitive Data

When working with sensitive information like passwords, you can use the `sensitive_data` parameter to prevent the model from seeing the actual values while still allowing it to reference them in its actions.

Make sure to always set [`allowed_domains`](https://docs.browser-use.com/customize/browser-settings#restrict-urls) to restrict the domains the Agent is allowed to visit when working with sensitive data or logins.

Here's an example of how to use sensitive data:

```
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig

load_dotenv()

# Initialize the model
llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0.0,
)

# Define sensitive data
# The model will only see the keys (x_name, x_password) but never the actual values
sensitive_data = {'x_name': 'magnus', 'x_password': '12345678'}

# Use the placeholder names in your task description
task = 'go to x.com and login with x_name and x_password then write a post about the meaning of life'

# Configure allowed_domains that the agent should be restricted to in BrowserContextConfig
context_config = BrowserContextConfig(
    allowed_domains=['example.com'],
)

# Pass the sensitive data to the agent
agent = Agent(
    task=task,
    llm=llm,
    sensitive_data=sensitive_data,
    browser=Browser(
        config=BrowserConfig(
            new_context_config=context_config
        )
    )
)

async def main():
    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())

```

In this example:

1. The model only sees `x_name` and `x_password` as placeholders.
2. When the model wants to use your password it outputs x\_password - and we replace it with the actual value.
3. When your password is visible on the current page, we replace it in the LLM input - so that the model never has it in its state.
4. The agent will be prevented from going to any site not on `example.com` to protect from prompt injection attacks and jailbreaks

### [​](#missing-or-empty-values) Missing or Empty Values

When working with sensitive data, keep these details in mind:

* If a key referenced by the model (`<secret>key_name</secret>`) is missing from your `sensitive_data` dictionary, a warning will be logged but the substitution tag will be preserved.
* If you provide an empty value for a key in the `sensitive_data` dictionary, it will be treated the same as a missing key.
* The system will always attempt to process all valid substitutions, even if some keys are missing or empty.

Warning: Vision models still see the image of the page - where the sensitive data might be visible.

This approach ensures that sensitive information remains secure while still allowing the agent to perform tasks that require authentication.

Was this page helpful?

YesNo

[System Prompt](/customize/system-prompt)[Custom Functions](/customize/custom-functions)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Handling Sensitive Data](#handling-sensitive-data)
* [Missing or Empty Values](#missing-or-empty-values)

---

# Custom Functions - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Custom Functions

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Custom Functions

Extend default agent and write custom function calls

## [​](#basic-function-registration) Basic Function Registration

Functions can be either `sync` or `async`. Keep them focused and single-purpose.

```
from browser_use import Controller, ActionResult
# Initialize the controller
controller = Controller()

@controller.action('Ask user for information')
def ask_human(question: str) -> str:
    answer = input(f'\n{question}\nInput: ')
    return ActionResult(extracted_content=answer)

```

Basic `Controller` has all basic functionality you might need to interact with
the browser already implemented.

```
# ... then pass controller to the agent
agent = Agent(
    task=task,
    llm=llm,
    controller=controller
)

```

Keep the function name and description short and concise. The Agent use the
function solely based on the name and description. The stringified output of
the action is passed to the Agent.

## [​](#browser-aware-functions) Browser-Aware Functions

For actions that need browser access, simply add the `browser` parameter inside the function parameters:

Please note that browser-use's `Browser` class is a wrapper class around
Playwright's `Browser`. The `Browser.playwright_browser` attr can be used
to directly access the Playwright browser object if needed.

```
from browser_use import Browser, Controller, ActionResult

controller = Controller()
@controller.action('Open website')
async def open_website(url: str, browser: Browser):
    page = await browser.get_current_page()
    await page.goto(url)
    return ActionResult(extracted_content='Website opened')

```

## [​](#structured-parameters-with-pydantic) Structured Parameters with Pydantic

For complex actions, you can define parameter schemas using Pydantic models:

```
from pydantic import BaseModel
from typing import Optional
from browser_use import Controller, ActionResult, Browser

controller = Controller()

class JobDetails(BaseModel):
    title: str
    company: str
    job_link: str
    salary: Optional[str] = None

@controller.action(
    'Save job details which you found on page',
    param_model=JobDetails
)
async def save_job(params: JobDetails, browser: Browser):
    print(f"Saving job: {params.title} at {params.company}")

    # Access browser if needed
    page = browser.get_current_page()
    await page.goto(params.job_link)

```

## [​](#using-custom-actions-with-multiple-agents) Using Custom Actions with multiple agents

You can use the same controller for multiple agents.

```
controller = Controller()

# ... register actions to the controller

agent = Agent(
    task="Go to website X and find the latest news",
    llm=llm,
    controller=controller
)

# Run the agent
await agent.run()

agent2 = Agent(
    task="Go to website Y and find the latest news",
    llm=llm,
    controller=controller
)

await agent2.run()

```

The controller is stateless and can be used to register multiple actions and
multiple agents.

## [​](#exclude-functions) Exclude functions

If you want less actions to be used by the agent, you can exclude them from the controller.

```
controller = Controller(exclude_actions=['open_tab', 'search_google'])

```

For more examples like file upload or notifications, visit [examples/custom-functions](https://github.com/browser-use/browser-use/tree/main/examples/custom-functions).

Was this page helpful?

YesNo

[Sensitive Data](/customize/sensitive-data)[Lifecycle Hooks](/customize/hooks)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Basic Function Registration](#basic-function-registration)
* [Browser-Aware Functions](#browser-aware-functions)
* [Structured Parameters with Pydantic](#structured-parameters-with-pydantic)
* [Using Custom Actions with multiple agents](#using-custom-actions-with-multiple-agents)
* [Exclude functions](#exclude-functions)

---

# Lifecycle Hooks - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Customize

Lifecycle Hooks

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Customize

# Lifecycle Hooks

Customize agent behavior with lifecycle hooks

# [​](#using-agent-lifecycle-hooks) Using Agent Lifecycle Hooks

Browser-Use provides lifecycle hooks that allow you to execute custom code at specific points during the agent's execution. These hooks enable you to capture detailed information about the agent's actions, modify behavior, or integrate with external systems.

## [​](#available-hooks) Available Hooks

Currently, Browser-Use provides the following hooks:

| Hook | Description | When it's called |
| --- | --- | --- |
| `on_step_start` | Executed at the beginning of each agent step | Before the agent processes the current state and decides on the next action |
| `on_step_end` | Executed at the end of each agent step | After the agent has executed the action for the current step |

## [​](#using-hooks) Using Hooks

Hooks are passed as parameters to the `agent.run()` method. Each hook should be a callable function that accepts the agent instance as its parameter.

### [​](#basic-example) Basic Example

```
from browser_use import Agent
from langchain_openai import ChatOpenAI


async def my_step_hook(agent):
    # inside a hook you can access all the state and methods under the Agent object:
    #   agent.settings, agent.state, agent.task
    #   agent.controller, agent.llm, agent.browser, agent.browser_context
    #   agent.pause(), agent.resume(), agent.add_new_task(...), etc.
    
    # You also have direct access to the playwright Page and Browser Context
    page = await agent.browser_context.get_current_page()
    #   https://playwright.dev/python/docs/api/class-page
    
    current_url = page.url
    visit_log = agent.state.history.urls()
    previous_url = visit_log[-2] if len(visit_log) >= 2 else None
    print(f"Agent was last on URL: {previous_url} and is now on {current_url}")
    
    # Example: listen for events on the page, interact with the DOM, run JS directly, etc.
    await page.on('domcontentloaded', lambda: print('page navigated to a new url...'))
    await page.locator("css=form > input[type=submit]").click()
    await page.evaluate('() => alert(1)')
    await page.browser.new_tab
    await agent.browser_context.session.context.add_init_script('/* some JS to run on every page */')
    
    # Example: monitor or intercept all network requests
    async def handle_request(route):
		# Print, modify, block, etc. do anything to the requests here
        #   https://playwright.dev/python/docs/network#handle-requests
		print(route.request, route.request.headers)
		await route.continue_(headers=route.request.headers)
	await page.route("**/*", handle_route)

    # Example: pause agent execution and resume it based on some custom code
    if '/completed' in current_url:
        agent.pause()
        Path('result.txt').write_text(await page.content()) 
        input('Saved "completed" page content to result.txt, press [Enter] to resume...')
        agent.resume()
    
agent = Agent(
    task="Search for the latest news about AI",
    llm=ChatOpenAI(model="gpt-4o"),
)

await agent.run(
    on_step_start=my_step_hook,
    # on_step_end=...
    max_steps=10
)

```

## [​](#complete-example%3A-agent-activity-recording-system) Complete Example: Agent Activity Recording System

This comprehensive example demonstrates a complete implementation for recording and saving Browser-Use agent activity, consisting of both server and client components.

### [​](#setup-instructions) Setup Instructions

To use this example, you'll need to:

1. Set up the required dependencies:

   ```
   pip install fastapi uvicorn prettyprinter pyobjtojson dotenv browser-use langchain-openai

   ```
2. Create two separate Python files:

   * `api.py` - The FastAPI server component
   * `client.py` - The Browser-Use agent with recording hook
3. Run both components:

   * Start the API server first: `python api.py`
   * Then run the client: `python client.py`

### [​](#server-component-api-py) Server Component (api.py)

The server component handles receiving and storing the agent's activity data:

```
#!/usr/bin/env python3

#
# FastAPI API to record and save Browser-Use activity data.
# Save this code to api.py and run with `python api.py`
# 

import json
import base64
from pathlib import Path

from fastapi import FastAPI, Request
import prettyprinter
import uvicorn

prettyprinter.install_extras()

# Utility function to save screenshots
def b64_to_png(b64_string: str, output_file):
    """
    Convert a Base64-encoded string to a PNG file.
    
    :param b64_string: A string containing Base64-encoded data
    :param output_file: The path to the output PNG file
    """
    with open(output_file, "wb") as f:
        f.write(base64.b64decode(b64_string))

# Initialize FastAPI app
app = FastAPI()


@app.post("/post_agent_history_step")
async def post_agent_history_step(request: Request):
    data = await request.json()
    prettyprinter.cpprint(data)

    # Ensure the "recordings" folder exists using pathlib
    recordings_folder = Path("recordings")
    recordings_folder.mkdir(exist_ok=True)

    # Determine the next file number by examining existing .json files
    existing_numbers = []
    for item in recordings_folder.iterdir():
        if item.is_file() and item.suffix == ".json":
            try:
                file_num = int(item.stem)
                existing_numbers.append(file_num)
            except ValueError:
                # In case the file name isn't just a number
                pass

    if existing_numbers:
        next_number = max(existing_numbers) + 1
    else:
        next_number = 1

    # Construct the file path
    file_path = recordings_folder / f"{next_number}.json"

    # Save the JSON data to the file
    with file_path.open("w") as f:
        json.dump(data, f, indent=2)

    # Optionally save screenshot if needed
    # if "website_screenshot" in data and data["website_screenshot"]:
    #     screenshot_folder = Path("screenshots")
    #     screenshot_folder.mkdir(exist_ok=True)
    #     b64_to_png(data["website_screenshot"], screenshot_folder / f"{next_number}.png")

    return {"status": "ok", "message": f"Saved to {file_path}"}

if __name__ == "__main__":
    print("Starting Browser-Use recording API on http://0.0.0.0:9000")
    uvicorn.run(app, host="0.0.0.0", port=9000)

```

### [​](#client-component-client-py) Client Component (client.py)

The client component runs the Browser-Use agent with a recording hook:

```
#!/usr/bin/env python3

#
# Client to record and save Browser-Use activity.
# Save this code to client.py and run with `python client.py`
#

import asyncio
import requests
from dotenv import load_dotenv
from pyobjtojson import obj_to_json
from langchain_openai import ChatOpenAI
from browser_use import Agent

# Load environment variables (for API keys)
load_dotenv()


def send_agent_history_step(data):
    """Send the agent step data to the recording API"""
    url = "http://127.0.0.1:9000/post_agent_history_step"
    response = requests.post(url, json=data)
    return response.json()


async def record_activity(agent_obj):
    """Hook function that captures and records agent activity at each step"""
    website_html = None
    website_screenshot = None
    urls_json_last_elem = None
    model_thoughts_last_elem = None
    model_outputs_json_last_elem = None
    model_actions_json_last_elem = None
    extracted_content_json_last_elem = None

    print('--- ON_STEP_START HOOK ---')
    
    # Capture current page state
    website_html = await agent_obj.browser_context.get_page_html()
    website_screenshot = await agent_obj.browser_context.take_screenshot()

    # Make sure we have state history
    if hasattr(agent_obj, "state"):
        history = agent_obj.state.history
    else:
        history = None
        print("Warning: Agent has no state history")
        return

    # Process model thoughts
    model_thoughts = obj_to_json(
        obj=history.model_thoughts(),
        check_circular=False
    )
    if len(model_thoughts) > 0:
        model_thoughts_last_elem = model_thoughts[-1]

    # Process model outputs
    model_outputs = agent_obj.state.history.model_outputs()
    model_outputs_json = obj_to_json(
        obj=model_outputs,
        check_circular=False
    )
    if len(model_outputs_json) > 0:
        model_outputs_json_last_elem = model_outputs_json[-1]

    # Process model actions
    model_actions = agent_obj.state.history.model_actions()
    model_actions_json = obj_to_json(
        obj=model_actions,
        check_circular=False
    )
    if len(model_actions_json) > 0:
        model_actions_json_last_elem = model_actions_json[-1]

    # Process extracted content
    extracted_content = agent_obj.state.history.extracted_content()
    extracted_content_json = obj_to_json(
        obj=extracted_content,
        check_circular=False
    )
    if len(extracted_content_json) > 0:
        extracted_content_json_last_elem = extracted_content_json[-1]

    # Process URLs
    urls = agent_obj.state.history.urls()
    urls_json = obj_to_json(
        obj=urls,
        check_circular=False
    )
    if len(urls_json) > 0:
        urls_json_last_elem = urls_json[-1]

    # Create a summary of all data for this step
    model_step_summary = {
        "website_html": website_html,
        "website_screenshot": website_screenshot,
        "url": urls_json_last_elem,
        "model_thoughts": model_thoughts_last_elem,
        "model_outputs": model_outputs_json_last_elem,
        "model_actions": model_actions_json_last_elem,
        "extracted_content": extracted_content_json_last_elem
    }

    print("--- MODEL STEP SUMMARY ---")
    print(f"URL: {urls_json_last_elem}")
    
    # Send data to the API
    result = send_agent_history_step(data=model_step_summary)
    print(f"Recording API response: {result}")


async def run_agent():
    """Run the Browser-Use agent with the recording hook"""
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    
    try:
        print("Starting Browser-Use agent with recording hook")
        await agent.run(
            on_step_start=record_activity,
            max_steps=30
        )
    except Exception as e:
        print(f"Error running agent: {e}")


if __name__ == "__main__":
    # Check if API is running
    try:
        requests.get("http://127.0.0.1:9000")
        print("Recording API is available")
    except:
        print("Warning: Recording API may not be running. Start api.py first.")
    
    # Run the agent
    asyncio.run(run_agent())

```

### [​](#working-with-the-recorded-data) Working with the Recorded Data

After running the agent, you'll find the recorded data in the `recordings` directory. Here's how you can use this data:

1. **View recorded sessions**: Each JSON file contains a snapshot of agent activity for one step
2. **Extract screenshots**: You can modify the API to save screenshots separately
3. **Analyze agent behavior**: Use the recorded data to study how the agent navigates websites

### [​](#extending-the-example) Extending the Example

You can extend this recording system in several ways:

1. **Save screenshots separately**: Uncomment the screenshot saving code in the API
2. **Add a web dashboard**: Create a simple web interface to view recorded sessions
3. **Add session IDs**: Modify the API to group steps by agent session
4. **Add filtering**: Implement filters to record only specific types of actions

## [​](#data-available-in-hooks) Data Available in Hooks

When working with agent hooks, you have access to the entire agent instance. Here are some useful data points you can access:

* `agent.state.history.model_thoughts()`: Reasoning from Browser Use's model.
* `agent.state.history.model_outputs()`: Raw outputs from the Browsre Use's model.
* `agent.state.history.model_actions()`: Actions taken by the agent
* `agent.state.history.extracted_content()`: Content extracted from web pages
* `agent.state.history.urls()`: URLs visited by the agent
* `agent.browser_context.get_page_html()`: Current page HTML
* `agent.browser_context.take_screenshot()`: Screenshot of the current page

## [​](#tips-for-using-hooks) Tips for Using Hooks

* **Avoid blocking operations**: Since hooks run in the same execution thread as the agent, try to keep them efficient or use asynchronous patterns.
* **Handle exceptions**: Make sure your hook functions handle exceptions gracefully to prevent interrupting the agent's main flow.
* **Consider storage needs**: When capturing full HTML and screenshots, be mindful of storage requirements.

Contribution by Carlos A. Planchón.

Was this page helpful?

YesNo

[Custom Functions](/customize/custom-functions)[Contribution Guide](/development/contribution-guide)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Using Agent Lifecycle Hooks](#using-agent-lifecycle-hooks)
* [Available Hooks](#available-hooks)
* [Using Hooks](#using-hooks)
* [Basic Example](#basic-example)
* [Complete Example: Agent Activity Recording System](#complete-example%3A-agent-activity-recording-system)
* [Setup Instructions](#setup-instructions)
* [Server Component (api.py)](#server-component-api-py)
* [Client Component (client.py)](#client-component-client-py)
* [Working with the Recorded Data](#working-with-the-recorded-data)
* [Extending the Example](#extending-the-example)
* [Data Available in Hooks](#data-available-in-hooks)
* [Tips for Using Hooks](#tips-for-using-hooks)

---

# Quickstart - Browser Use

[Browser Use home page![light logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/light.svg)![dark logo](https://mintlify.s3.us-west-1.amazonaws.com/browseruse-0aece648/logo/dark.svg)](https://browser-use.com)

Search or ask...

* [Github](https://github.com/browser-use/browser-use)
* [Twitter](https://x.com/gregpr07)
* [Join Discord](https://link.browser-use.com/discord)
* [Join Discord](https://link.browser-use.com/discord)

Search...

Navigation

Get Started

Quickstart

[Documentation](/introduction)[Cloud API](/cloud/quickstart)

##### Get Started

* [Introduction](/introduction)
* [Quickstart](/quickstart)

##### Customize

* [Supported Models](/customize/supported-models)
* [Agent Settings](/customize/agent-settings)
* [Browser Settings](/customize/browser-settings)
* [Connect to your Browser](/customize/real-browser)
* [Output Format](/customize/output-format)
* [System Prompt](/customize/system-prompt)
* [Sensitive Data](/customize/sensitive-data)
* [Custom Functions](/customize/custom-functions)
* [Lifecycle Hooks](/customize/hooks)

##### Development

* [Contribution Guide](/development/contribution-guide)
* [Local Setup](/development/local-setup)
* [Telemetry](/development/telemetry)
* [Observability](/development/observability)
* [Evaluations](/development/evaluations)
* [Roadmap](/development/roadmap)

Get Started

# Quickstart

Start using Browser Use with this quickstart guide

## [​](#prepare-the-environment) Prepare the environment

Browser Use requires Python 3.11 or higher.

First, we recommend using [uv](https://docs.astral.sh/uv/) to setup the Python environment.

```
uv venv --python 3.11

```

and activate it with:

```
# For Mac/Linux:
source .venv/bin/activate

# For Windows:
.venv\Scripts\activate

```

Install the dependencies:

```
uv pip install browser-use

```

Then install playwright:

```
uv run playwright install

```

## [​](#create-an-agent) Create an agent

Then you can use the agent as follows:

agent.py

```
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
load_dotenv()

import asyncio

llm = ChatOpenAI(model="gpt-4o")

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=llm,
    )
    result = await agent.run()
    print(result)

asyncio.run(main())

```

## [​](#set-up-your-llm-api-keys) Set up your LLM API keys

`ChatOpenAI` and other Langchain chat models require API keys. You should store these in your `.env` file. For example, for OpenAI and Anthropic, you can set the API keys in your `.env` file, such as:

.env

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

```

For other LLM models you can refer to the [Langchain documentation](https://python.langchain.com/docs/integrations/chat/) to find how to set them up with their specific API keys.

Was this page helpful?

YesNo

[Introduction](/introduction)[Supported Models](/customize/supported-models)

[x](https://x.com/gregpr07)[github](https://github.com/browser-use/browser-use)[linkedin](https://linkedin.com/company/browser-use)

[Powered by Mintlify](https://mintlify.com/preview-request?utm_campaign=poweredBy&utm_medium=referral&utm_source=docs.browser-use.com)

On this page

* [Prepare the environment](#prepare-the-environment)
* [Create an agent](#create-an-agent)
* [Set up your LLM API keys](#set-up-your-llm-api-keys)

---


