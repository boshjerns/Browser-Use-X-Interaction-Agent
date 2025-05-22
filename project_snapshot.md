# Snapshot of **browser-use-main**
Auto-generated digest of key files. Paste this into ChatGPT to help craft a README.

---

## `app.py`

```py
from flask import Flask, render_template, request, jsonify, Response
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
import re
import queue
import threading
from cryptography.fernet import Fernet
import base64
from pathlib import Path

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global queue to store search progress updates
search_progress = queue.Queue()

# Initialize encryption key
def get_or_create_key():
    key_file = Path('.api_key')
    if key_file.exists():
        return key_file.read_bytes()
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    return key

# Initialize Fernet cipher
cipher = Fernet(get_or_create_key())

def encrypt_api_key(api_key):
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key):
    try:
        return cipher.decrypt(encrypted_key.encode()).decode()
    except Exception:
        return None

@app.route('/')
def home():
    logger.info('Rendering home page')
    return render_template('index.html')

def extract_emails_and_phones(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'(\+?\d[\d\s().-]{7,}\d)'
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    return emails, phones

def deduplicate_contacts(existing, new_contacts):
    # Use a set of (email, phone) tuples for deduplication
    seen = set((c.get('email',''), c.get('phone','')) for c in existing)
    unique = []
    for c in new_contacts:
        key = (c.get('email',''), c.get('phone',''))
        if key not in seen and (c.get('email') or c.get('phone')):
            unique.append(c)
            seen.add(key)
    return existing + unique

def save_contacts_stepwise(target, contacts):
    try:
        with open("contacts.json", "r") as f:
            all_contacts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_contacts = []
    for contact in contacts:
        contact['target'] = target
        contact['timestamp'] = datetime.utcnow().isoformat()
    all_contacts = deduplicate_contacts(all_contacts, contacts)
    with open("contacts.json", "w") as f:
        json.dump(all_contacts, f, indent=2)

@app.route('/contacts', methods=['GET'])
def get_contacts():
    try:
        with open("contacts.json", "r") as f:
            contacts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        contacts = []
    return jsonify(contacts)

@app.route('/search-progress')
def search_progress_stream():
    def generate():
        while True:
            try:
                # Get the next progress update from the queue
                progress = search_progress.get(timeout=30)  # 30 second timeout
                yield f"data: {json.dumps(progress)}\n\n"
            except queue.Empty:
                # Send a heartbeat to keep the connection alive
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/search', methods=['POST'])
def search():
    logger.info('Received search request')
    data = request.get_json()
    api_key = data.get('api_key')
    query = data.get('query')
    logger.info(f'API key provided: {bool(api_key)}, Query: {query}')

    if not api_key or not query:
        logger.warning('Missing API key or query')
        return jsonify({'error': 'API key and query are required'}), 400

    os.environ['OPENAI_API_KEY'] = api_key
    logger.info('Set OpenAI API key')

    model = ChatOpenAI(model="gpt-4o")
    logger.info('Initialized ChatOpenAI model')

    robust_prompt = (
        f"As you search for contact information for {query}, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. "
        "After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). "
        "Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted."
    )

    async def on_step_end(agent):
        history = agent.state.history
        extracted = history.extracted_content()
        source = history.urls()[-1] if history.urls() else ""
        found_contacts = []
        if extracted:
            last = extracted[-1]
            # Try to parse JSON array as before
            try:
                start = last.find('[')
                end = last.rfind(']')
                if start != -1 and end != -1:
                    json_str = last[start:end+1]
                    contacts = json.loads(json_str)
                    for c in contacts:
                        if 'source' not in c:
                            c['source'] = source
                    found_contacts.extend(contacts)
            except Exception as e:
                logger.error(f'Error parsing/saving contacts: {e}')
            # Always scan for emails/phones in the text
            emails, phones = extract_emails_and_phones(last)
            for email in emails:
                found_contacts.append({
                    'role': 'Unknown',
                    'name': '',
                    'email': email,
                    'phone': '',
                    'source': source
                })
            for phone in phones:
                found_contacts.append({
                    'role': 'Unknown',
                    'name': '',
                    'email': '',
                    'phone': phone,
                    'source': source
                })
        if found_contacts:
            save_contacts_stepwise(agent.task, found_contacts)
            # Send progress update
            search_progress.put({
                'type': 'contacts_found',
                'contacts': found_contacts,
                'source': source
            })

    async def run_agent():
        agent = Agent(task=robust_prompt, llm=model)
        logger.info('Initialized Agent')
        # Send initial progress update
        search_progress.put({
            'type': 'status',
            'message': 'Starting search...'
        })
        result = await agent.run(on_step_end=on_step_end, max_steps=100)
        logger.info('Agent run complete')
        # Send final progress update
        search_progress.put({
            'type': 'status',
            'message': 'Search complete'
        })
        if hasattr(result, 'final_result'):
            final = result.final_result()
            logger.info(f'Final result: {final}')
            return final
        return str(result)

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
… (truncated)

```

---

## `browser-use-main\.github\CONTRIBUTING.md`

```md
# Contributing to browser-use

We love contributions! Please read through these links to get started:

 - 🔢 [Contribution Guidelines](https://docs.browser-use.com/development/contribution-guide)
 - 👾 [Local Development Setup Guide](https://docs.browser-use.com/development/local-setup)
 - 🏷️ [Issues Tagged: `#help-wanted`](https://github.com/browser-use/browser-use/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22)

```

---

## `browser-use-main\browser_use\__init__.py`

```py
import warnings

# Suppress specific deprecation warnings from FAISS
warnings.filterwarnings('ignore', category=DeprecationWarning, module='faiss.loader')
warnings.filterwarnings('ignore', message='builtin type SwigPyPacked has no __module__ attribute')
warnings.filterwarnings('ignore', message='builtin type SwigPyObject has no __module__ attribute')
warnings.filterwarnings('ignore', message='builtin type swigvarlink has no __module__ attribute')

from browser_use.logging_config import setup_logging

setup_logging()

from browser_use.agent.prompts import SystemPrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionModel, ActionResult, AgentHistoryList
from browser_use.browser import Browser, BrowserConfig, BrowserContext, BrowserContextConfig, BrowserProfile, BrowserSession
from browser_use.controller.service import Controller
from browser_use.dom.service import DomService

__all__ = [
	'Agent',
	'Browser',
	'BrowserConfig',
	'BrowserSession',
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	'BrowserContext',
	'BrowserContextConfig',
]

```

---

## `browser-use-main\browser_use\agent\gif.py`

```py
from __future__ import annotations

import base64
import io
import logging
import os
import platform
from typing import TYPE_CHECKING

from browser_use.agent.views import AgentHistoryList

if TYPE_CHECKING:
	from PIL import Image, ImageFont

logger = logging.getLogger(__name__)


def decode_unicode_escapes_to_utf8(text: str) -> str:
	"""Handle decoding any unicode escape sequences embedded in a string (needed to render non-ASCII languages like chinese or arabic in the GIF overlay text)"""

	if r'\u' not in text:
		# doesn't have any escape sequences that need to be decoded
		return text

	try:
		# Try to decode Unicode escape sequences
		return text.encode('latin1').decode('unicode_escape')
	except (UnicodeEncodeError, UnicodeDecodeError):
		# logger.debug(f"Failed to decode unicode escape sequences while generating gif text: {text}")
		return text


def create_history_gif(
	task: str,
	history: AgentHistoryList,
	#
	output_path: str = 'agent_history.gif',
	duration: int = 3000,
	show_goals: bool = True,
	show_task: bool = True,
	show_logo: bool = False,
	font_size: int = 40,
	title_font_size: int = 56,
	goal_font_size: int = 44,
	margin: int = 40,
	line_spacing: float = 1.5,
) -> None:
	"""Create a GIF from the agent's history with overlaid task and goal text."""
	if not history.history:
		logger.warning('No history to create GIF from')
		return

	from PIL import Image, ImageFont

	images = []

	# if history is empty or first screenshot is None, we can't create a gif
	if not history.history or not history.history[0].state.screenshot:
		logger.warning('No history or first screenshot to create GIF from')
		return

	# Try to load nicer fonts
	try:
		# Try different font options in order of preference
		# ArialUni is a font that comes with Office and can render most non-alphabet characters
		font_options = [
			'Microsoft YaHei',  # 微软雅黑
			'SimHei',  # 黑体
			'SimSun',  # 宋体
			'Noto Sans CJK SC',  # 思源黑体
			'WenQuanYi Micro Hei',  # 文泉驿微米黑
			'Helvetica',
			'Arial',
			'DejaVuSans',
			'Verdana',
		]
		font_loaded = False

		for font_name in font_options:
			try:
				if platform.system() == 'Windows':
					# Need to specify the abs font path on Windows
					font_name = os.path.join(os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts'), font_name + '.ttf')
				regular_font = ImageFont.truetype(font_name, font_size)
				title_font = ImageFont.truetype(font_name, title_font_size)
				goal_font = ImageFont.truetype(font_name, goal_font_size)
				font_loaded = True
				break
			except OSError:
				continue

		if not font_loaded:
			raise OSError('No preferred fonts found')

	except OSError:
		regular_font = ImageFont.load_default()
		title_font = ImageFont.load_default()

		goal_font = regular_font

	# Load logo if requested
	logo = None
	if show_logo:
		try:
			logo = Image.open('./static/browser-use.png')
			# Resize logo to be small (e.g., 40px height)
			logo_height = 150
			aspect_ratio = logo.width / logo.height
			logo_width = int(logo_height * aspect_ratio)
			logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
		except Exception as e:
			logger.warning(f'Could not load logo: {e}')

	# Create task frame if requested
	if show_task and task:
		task_frame = _create_task_frame(
			task,
			history.history[0].state.screenshot,
			title_font,  # type: ignore
			regular_font,  # type: ignore
			logo,
			line_spacing,
		)
		images.append(task_frame)

	# Process each history item
	for i, item in enumerate(history.history, 1):
		if not item.state.screenshot:
			continue

		# Convert base64 screenshot to PIL Image
		img_data = base64.b64decode(item.state.screenshot)
		image = Image.open(io.BytesIO(img_data))

		if show_goals and item.model_output:
			image = _add_overlay_to_image(
				image=image,
				step_number=i,
				goal_text=item.model_output.current_state.next_goal,
				regular_font=regular_font,  # type: ignore
				title_font=title_font,  # type: ignore
				margin=margin,
				logo=logo,
			)

		images.append(image)

	if images:
		# Save the GIF
		images[0].save(
			output_path,
			save_all=True,
			append_images=images[1:],
			duration=duration,
			loop=0,
			optimize=False,
		)
		logger.info(f'Created GIF at {output_path}')
	else:
		logger.warning('No images found in history to create GIF')


def _create_task_frame(
	task: str,
	first_screenshot: str,
	title_font: ImageFont.FreeTypeFont,
	regular_font: ImageFont.FreeTypeFont,
	logo: Image.Image | None = None,
	line_spacing: float = 1.5,
) -> Image.Image:
	"""Create initial frame showing the task."""
	from PIL import Image, ImageDraw, ImageFont

	img_data = base64.b64decode(first_screenshot)
	template = Image.open(io.BytesIO(img_data))
	image = Image.new('RGB', template.size, (0, 0, 0))
	draw = ImageDraw.Draw(image)

	# Calculate vertical center of image
	center_y = image.height // 2

	# Draw task text with dynamic font size based on task length
	margin = 140  # Increased margin
	max_width = image.width - (2 * margin)

	# Dynamic font size calculation based on task length
	# Start with base font size (regular + 16)
	base_font_size = regular_font.size + 16
	min_font_size = max(regular_font.size - 10, 16)  # Don't go below 16pt
	max_font_size = base_font_size  # Cap at the base font size

	# Calculate dynamic font size based on text length and complexity
	# Longer texts get progressively smaller fonts
	text_length = len(task)
	if text_length > 200:
		# For very long text, reduce font size logarithmically
		font_size = max(base_font_size - int(10 * (text_length / 200)), min_font_size)
	else:
		font_size = base_font_size

… (truncated)

```

---

## `browser-use-main\browser_use\agent\memory\__init__.py`

```py
from browser_use.agent.memory.service import Memory
from browser_use.agent.memory.views import MemoryConfig

__all__ = ['Memory', 'MemoryConfig']

```

---

## `browser-use-main\browser_use\agent\memory\service.py`

```py
from __future__ import annotations

import logging
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	BaseMessage,
	HumanMessage,
)
from langchain_core.messages.utils import convert_to_openai_messages

from browser_use.agent.memory.views import MemoryConfig
from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.message_manager.views import ManagedMessage, MessageMetadata
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class Memory:
	"""
	Manages procedural memory for agents.

	This class implements a procedural memory management system using Mem0 that transforms agent interaction history
	into concise, structured representations at specified intervals. It serves to optimize context window
	utilization during extended task execution by converting verbose historical information into compact,
	yet comprehensive memory constructs that preserve essential operational knowledge.
	"""

	def __init__(
		self,
		message_manager: MessageManager,
		llm: BaseChatModel,
		config: MemoryConfig | None = None,
	):
		self.message_manager = message_manager
		self.llm = llm

		# Initialize configuration with defaults based on the LLM if not provided
		if config is None:
			self.config = MemoryConfig(llm_instance=llm, agent_id=f'agent_{id(self)}')

			# Set appropriate embedder based on LLM type
			llm_class = llm.__class__.__name__
			if llm_class == 'ChatOpenAI':
				self.config.embedder_provider = 'openai'
				self.config.embedder_model = 'text-embedding-3-small'
				self.config.embedder_dims = 1536
			elif llm_class == 'ChatGoogleGenerativeAI':
				self.config.embedder_provider = 'gemini'
				self.config.embedder_model = 'models/text-embedding-004'
				self.config.embedder_dims = 768
			elif llm_class == 'ChatOllama':
				self.config.embedder_provider = 'ollama'
				self.config.embedder_model = 'nomic-embed-text'
				self.config.embedder_dims = 512
		else:
			# Ensure LLM instance is set in the config
			self.config = MemoryConfig(**dict(config))  # re-validate untrusted user-provided config
			self.config.llm_instance = llm

		# Check for required packages
		try:
			# also disable mem0's telemetry when ANONYMIZED_TELEMETRY=False
			if os.getenv('ANONYMIZED_TELEMETRY', 'true').lower()[0] in 'fn0':
				os.environ['MEM0_TELEMETRY'] = 'False'
			from mem0 import Memory as Mem0Memory
		except ImportError:
			raise ImportError('mem0 is required when enable_memory=True. Please install it with `pip install mem0`.')

		if self.config.embedder_provider == 'huggingface':
			try:
				# check that required package is installed if huggingface is used
				from sentence_transformers import SentenceTransformer  # noqa: F401
			except ImportError:
				raise ImportError(
					'sentence_transformers is required when enable_memory=True and embedder_provider="huggingface". Please install it with `pip install sentence-transformers`.'
				)

		# Initialize Mem0 with the configuration
		self.mem0 = Mem0Memory.from_config(config_dict=self.config.full_config_dict)

	@time_execution_sync('--create_procedural_memory')
	def create_procedural_memory(self, current_step: int) -> None:
		"""
		Create a procedural memory if needed based on the current step.

		Args:
		    current_step: The current step number of the agent
		"""
		logger.debug(f'Creating procedural memory at step {current_step}')

		# Get all messages
		all_messages = self.message_manager.state.history.messages

		# Separate messages into those to keep as-is and those to process for memory
		new_messages = []
		messages_to_process = []

		for msg in all_messages:
			if isinstance(msg, ManagedMessage) and msg.metadata.message_type in {'init', 'memory'}:
				# Keep system and memory messages as they are
				new_messages.append(msg)
			else:
				if len(msg.message.content) > 0:
					messages_to_process.append(msg)

		# Need at least 2 messages to create a meaningful summary
		if len(messages_to_process) <= 1:
			logger.debug('Not enough non-memory messages to summarize')
			return
		# Create a procedural memory
		memory_content = self._create([m.message for m in messages_to_process], current_step)

		if not memory_content:
			logger.warning('Failed to create procedural memory')
			return

		# Replace the processed messages with the consolidated memory
		memory_message = HumanMessage(content=memory_content)
		memory_tokens = self.message_manager._count_tokens(memory_message)
		memory_metadata = MessageMetadata(tokens=memory_tokens, message_type='memory')

		# Calculate the total tokens being removed
		removed_tokens = sum(m.metadata.tokens for m in messages_to_process)

		# Add the memory message
		new_messages.append(ManagedMessage(message=memory_message, metadata=memory_metadata))

		# Update the history
		self.message_manager.state.history.messages = new_messages
		self.message_manager.state.history.current_tokens -= removed_tokens
		self.message_manager.state.history.current_tokens += memory_tokens
		logger.info(f'Messages consolidated: {len(messages_to_process)} messages converted to procedural memory')

	def _create(self, messages: list[BaseMessage], current_step: int) -> str | None:
		parsed_messages = convert_to_openai_messages(messages)
		try:
			results = self.mem0.add(
				messages=parsed_messages,
				agent_id=self.config.agent_id,
				memory_type='procedural_memory',
				metadata={'step': current_step},
			)
			if len(results.get('results', [])):
				return results.get('results', [])[0].get('memory')
			return None
		except Exception as e:
			logger.error(f'Error creating procedural memory: {e}')
			return None

```

---

## `browser-use-main\browser_use\agent\memory\views.py`

```py
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field


class MemoryConfig(BaseModel):
	"""Configuration for procedural memory."""

	model_config = ConfigDict(
		from_attributes=True, validate_default=True, revalidate_instances='always', validate_assignment=True
	)

	# Memory settings
	agent_id: str = Field(default='browser_use_agent', min_length=1)
	memory_interval: int = Field(default=10, gt=1, lt=100)

	# Embedder settings
	embedder_provider: Literal['openai', 'gemini', 'ollama', 'huggingface'] = 'huggingface'
	embedder_model: str = Field(min_length=2, default='all-MiniLM-L6-v2')
	embedder_dims: int = Field(default=384, gt=10, lt=10000)

	# LLM settings - the LLM instance can be passed separately
	llm_provider: Literal['langchain'] = 'langchain'
	llm_instance: BaseChatModel | None = None

	# Vector store settings
	vector_store_provider: Literal['faiss'] = 'faiss'
	vector_store_base_path: str = Field(default='/tmp/mem0')

	@property
	def vector_store_path(self) -> str:
		"""Returns the full vector store path for the current configuration. e.g. /tmp/mem0_384_faiss"""
		return f'{self.vector_store_base_path}_{self.embedder_dims}_{self.vector_store_provider}'

	@property
	def embedder_config_dict(self) -> dict[str, Any]:
		"""Returns the embedder configuration dictionary."""
		return {
			'provider': self.embedder_provider,
			'config': {'model': self.embedder_model, 'embedding_dims': self.embedder_dims},
		}

	@property
	def llm_config_dict(self) -> dict[str, Any]:
		"""Returns the LLM configuration dictionary."""
		return {'provider': self.llm_provider, 'config': {'model': self.llm_instance}}

	@property
	def vector_store_config_dict(self) -> dict[str, Any]:
		"""Returns the vector store configuration dictionary."""
		return {
			'provider': self.vector_store_provider,
			'config': {
				'embedding_model_dims': self.embedder_dims,
				'path': self.vector_store_path,
			},
		}

	@property
	def full_config_dict(self) -> dict[str, dict[str, Any]]:
		"""Returns the complete configuration dictionary for Mem0."""
		return {
			'embedder': self.embedder_config_dict,
			'llm': self.llm_config_dict,
			'vector_store': self.vector_store_config_dict,
		}

```

---

## `browser-use-main\browser_use\agent\message_manager\service.py`

```py
from __future__ import annotations

import logging

from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
	ToolMessage,
)
from pydantic import BaseModel

from browser_use.agent.message_manager.views import MessageMetadata
from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.agent.views import ActionResult, AgentOutput, AgentStepInfo, MessageManagerState
from browser_use.browser.views import BrowserStateSummary
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


class MessageManagerSettings(BaseModel):
	max_input_tokens: int = 128000
	estimated_characters_per_token: int = 3
	image_tokens: int = 800
	include_attributes: list[str] = []
	message_context: str | None = None
	sensitive_data: dict[str, str] | None = None
	available_file_paths: list[str] | None = None


class MessageManager:
	def __init__(
		self,
		task: str,
		system_message: SystemMessage,
		settings: MessageManagerSettings = MessageManagerSettings(),
		state: MessageManagerState = MessageManagerState(),
	):
		self.task = task
		self.settings = settings
		self.state = state
		self.system_prompt = system_message

		# Only initialize messages if state is empty
		if len(self.state.history.messages) == 0:
			self._init_messages()

	def _init_messages(self) -> None:
		"""Initialize the message history with system message, context, task, and other initial messages"""
		self._add_message_with_tokens(self.system_prompt, message_type='init')

		if self.settings.message_context:
			context_message = HumanMessage(content='Context for the task' + self.settings.message_context)
			self._add_message_with_tokens(context_message, message_type='init')

		task_message = HumanMessage(
			content=f'Your ultimate task is: """{self.task}""". If you achieved your ultimate task, stop everything and use the done action in the next step to complete the task. If not, continue as usual.'
		)
		self._add_message_with_tokens(task_message, message_type='init')

		if self.settings.sensitive_data:
			info = f'Here are placeholders for sensitive data: {list(self.settings.sensitive_data.keys())}'
			info += '\nTo use them, write <secret>the placeholder name</secret>'
			info_message = HumanMessage(content=info)
			self._add_message_with_tokens(info_message, message_type='init')

		placeholder_message = HumanMessage(content='Example output:')
		self._add_message_with_tokens(placeholder_message, message_type='init')

		example_tool_call = AIMessage(
			content='',
			tool_calls=[
				{
					'name': 'AgentOutput',
					'args': {
						'current_state': {
							'evaluation_previous_goal': """
							Success - I successfully clicked on the 'Apple' link from the Google Search results page, 
							which directed me to the 'Apple' company homepage. This is a good start toward finding 
							the best place to buy a new iPhone as the Apple website often list iPhones for sale.
						""".strip(),
							'memory': """
							I searched for 'iPhone retailers' on Google. From the Google Search results page, 
							I used the 'click_element_by_index' tool to click on element at index [45] labeled 'Best Buy' but calling 
							the tool did not direct me to a new page. I then used the 'click_element_by_index' tool to click 
							on element at index [82] labeled 'Apple' which redirected me to the 'Apple' company homepage. 
							Currently at step 3/15.
						""".strip(),
							'next_goal': """
							Looking at reported structure of the current page, I can see the item '[127]<h3 iPhone/>' 
							in the content. I think this button will lead to more information and potentially prices 
							for iPhones. I'll click on the link at index [127] using the 'click_element_by_index' 
							tool and hope to see prices on the next page.
						""".strip(),
						},
						'action': [{'click_element_by_index': {'index': 127}}],
					},
					'id': str(self.state.tool_id),
					'type': 'tool_call',
				},
			],
		)
		self._add_message_with_tokens(example_tool_call, message_type='init')
		self.add_tool_message(content='Browser started', message_type='init')

		placeholder_message = HumanMessage(content='[Your task history memory starts here]')
		self._add_message_with_tokens(placeholder_message)

		if self.settings.available_file_paths:
			filepaths_msg = HumanMessage(content=f'Here are file paths you can use: {self.settings.available_file_paths}')
			self._add_message_with_tokens(filepaths_msg, message_type='init')

	def add_new_task(self, new_task: str) -> None:
		content = f'Your new ultimate task is: """{new_task}""". Take the previous context into account and finish your new ultimate task. '
		msg = HumanMessage(content=content)
		self._add_message_with_tokens(msg)
		self.task = new_task

	@time_execution_sync('--add_state_message')
	def add_state_message(
		self,
		browser_state_summary: BrowserStateSummary,
		result: list[ActionResult] | None = None,
		step_info: AgentStepInfo | None = None,
		use_vision=True,
	) -> None:
		"""Add browser state as human message"""

		# if keep in memory, add to directly to history and add state without result
		if result:
			for r in result:
				if r.include_in_memory:
					if r.extracted_content:
						msg = HumanMessage(content='Action result: ' + str(r.extracted_content))
						self._add_message_with_tokens(msg)
					if r.error:
						# if endswith \n, remove it
						if r.error.endswith('\n'):
							r.error = r.error[:-1]
						# get only last line of error
						last_line = r.error.split('\n')[-1]
						msg = HumanMessage(content='Action error: ' + last_line)
						self._add_message_with_tokens(msg)
					result = None  # if result in history, we dont want to add it again

		# otherwise add state message and result to next message (which will not stay in memory)
		assert browser_state_summary
		state_message = AgentMessagePrompt(
			browser_state_summary=browser_state_summary,
			result=result,
			include_attributes=self.settings.include_attributes,
			step_info=step_info,
		).get_user_message(use_vision)
		self._add_message_with_tokens(state_message)

	def add_model_output(self, model_output: AgentOutput) -> None:
		"""Add model output as AI message"""
		tool_calls = [
			{
				'name': 'AgentOutput',
				'args': model_output.model_dump(mode='json', exclude_unset=True),
				'id': str(self.state.tool_id),
				'type': 'tool_call',
			}
		]

		msg = AIMessage(
			content='',
			tool_calls=tool_calls,
		)

		self._add_message_with_tokens(msg)
		# empty tool response
		self.add_tool_message(content='')

	def add_plan(self, plan: str | None, position: int | None = None) -> None:
		if plan:
			msg = AIMessage(content=plan)
			self._add_message_with_tokens(msg, position)

	@time_execution_sync('--get_messages')
	def get_messages(self) -> list[BaseMessage]:
		"""Get current message list, potentially trimmed to max tokens"""

		msg = [m.message for m in self.state.history.messages]
		# debug which messages are in history with token count # log
		total_input_tokens = 0
		logger.debug(f'Messages in history: {len(self.state.history.messages)}:')
		for m in self.state.history.messages:
			total_input_tokens += m.metadata.tokens
			logger.debug(f'{m.message.__class__.__name__} - Token count: {m.metadata.tokens}')
		logger.debug(f'Total input tokens: {total_input_tokens}')

		return msg

	def _add_message_with_tokens(
		self, message: BaseMessage, position: int | None = None, message_type: str | None = None
	) -> None:
… (truncated)

```

---

## `browser-use-main\browser_use\agent\message_manager\tests.py`

```py
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.views import ActionResult
from browser_use.browser.views import BrowserStateSummary, TabInfo
from browser_use.dom.views import DOMElementNode, DOMTextNode


@pytest.fixture(
	params=[
		ChatOpenAI(model='gpt-4o-mini'),
		AzureChatOpenAI(model='gpt-4o', api_version='2024-02-15-preview'),
		ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=100, temperature=0.0, stop=None),
	],
	ids=['gpt-4o-mini', 'gpt-4o', 'claude-3-5-sonnet'],
)
def message_manager(request: pytest.FixtureRequest):
	task = 'Test task'
	action_descriptions = 'Test actions'
	return MessageManager(
		task=task,
		system_message=SystemMessage(content=action_descriptions),
		settings=MessageManagerSettings(
			max_input_tokens=1000,
			estimated_characters_per_token=3,
			image_tokens=800,
		),
	)


def test_initial_messages(message_manager: MessageManager):
	"""Test that message manager initializes with system and task messages"""
	messages = message_manager.get_messages()
	assert len(messages) == 2
	assert isinstance(messages[0], SystemMessage)
	assert isinstance(messages[1], HumanMessage)
	assert 'Test task' in messages[1].content


def test_add_state_message(message_manager: MessageManager):
	"""Test adding browser state message"""
	state = BrowserStateSummary(
		url='https://test.com',
		title='Test Page',
		element_tree=DOMElementNode(
			tag_name='div',
			attributes={},
			children=[],
			is_visible=True,
			parent=None,
			xpath='//div',
		),
		selector_map={},
		tabs=[TabInfo(page_id=1, url='https://test.com', title='Test Page')],
	)
	message_manager.add_state_message(browser_state_summary=state)

	messages = message_manager.get_messages()
	assert len(messages) == 3
	assert isinstance(messages[2], HumanMessage)
	assert 'https://test.com' in messages[2].content


def test_add_state_with_memory_result(message_manager: MessageManager):
	"""Test adding state with result that should be included in memory"""
	state = BrowserStateSummary(
		url='https://test.com',
		title='Test Page',
		element_tree=DOMElementNode(
			tag_name='div',
			attributes={},
			children=[],
			is_visible=True,
			parent=None,
			xpath='//div',
		),
		selector_map={},
		tabs=[TabInfo(page_id=1, url='https://test.com', title='Test Page')],
	)
	result = ActionResult(extracted_content='Important content', include_in_memory=True)

	message_manager.add_state_message(browser_state_summary=state, result=[result])
	messages = message_manager.get_messages()

	# Should have system, task, extracted content, and state messages
	assert len(messages) == 4
	assert 'Important content' in messages[2].content
	assert isinstance(messages[2], HumanMessage)
	assert isinstance(messages[3], HumanMessage)
	assert 'Important content' not in messages[3].content


def test_add_state_with_non_memory_result(message_manager: MessageManager):
	"""Test adding state with result that should not be included in memory"""
	state = BrowserStateSummary(
		url='https://test.com',
		title='Test Page',
		element_tree=DOMElementNode(
			tag_name='div',
			attributes={},
			children=[],
			is_visible=True,
			parent=None,
			xpath='//div',
		),
		selector_map={},
		tabs=[TabInfo(page_id=1, url='https://test.com', title='Test Page')],
	)
	result = ActionResult(extracted_content='Temporary content', include_in_memory=False)

	message_manager.add_state_message(browser_state_summary=state, result=[result])
	messages = message_manager.get_messages()

	# Should have system, task, and combined state+result message
	assert len(messages) == 3
	assert 'Temporary content' in messages[2].content
	assert isinstance(messages[2], HumanMessage)


@pytest.mark.skip('not sure how to fix this')
@pytest.mark.parametrize('max_tokens', [100000, 10000, 5000])
def test_token_overflow_handling_with_real_flow(message_manager: MessageManager, max_tokens):
	"""Test handling of token overflow in a realistic message flow"""
	# Set more realistic token limit
	message_manager.settings.max_input_tokens = max_tokens

	# Create a long sequence of interactions
	for i in range(200):  # Simulate 40 steps of interaction
		# Create state with varying content length
		state = BrowserStateSummary(
			url=f'https://test{i}.com',
			title=f'Test Page {i}',
			element_tree=DOMElementNode(
				tag_name='div',
				attributes={},
				children=[
					DOMTextNode(
						text=f'Content {j} ' * (10 + i),  # Increasing content length
						is_visible=True,
						parent=None,
					)
					for j in range(5)  # Multiple DOM items
				],
				is_visible=True,
				parent=None,
				xpath='//div',
			),
			selector_map={j: f'//div[{j}]' for j in range(5)},
			tabs=[TabInfo(page_id=1, url=f'https://test{i}.com', title=f'Test Page {i}')],
		)

		# Alternate between different types of results
		result = None
		if i % 2 == 0:  # Every other iteration
			result = ActionResult(
				extracted_content=f'Important content from step {i}' * 5,
				include_in_memory=i % 4 == 0,  # Include in memory every 4th message
			)

		# Add state message
		if result:
			message_manager.add_state_message(browser_state_summary=state, result=[result])
		else:
			message_manager.add_state_message(browser_state_summary=state)

		try:
			messages = message_manager.get_messages()
		except ValueError as e:
			if 'Max token limit reached - history is too long' in str(e):
				return  # If error occurs, end the test
			else:
				raise e

		assert message_manager.state.history.current_tokens <= message_manager.settings.max_input_tokens + 100

		last_msg = messages[-1]
		assert isinstance(last_msg, HumanMessage)

		if i % 4 == 0:
			assert isinstance(message_manager.state.history.messages[-2].message, HumanMessage)
		if i % 2 == 0 and not i % 4 == 0:
			if isinstance(last_msg.content, list):
				assert 'Current url: https://test' in last_msg.content[0]['text']
			else:
				assert 'Current url: https://test' in last_msg.content

		# Add model output every time
		from browser_use.agent.views import AgentBrain, AgentOutput
		from browser_use.controller.registry.views import ActionModel

		output = AgentOutput(
			current_state=AgentBrain(
				evaluation_previous_goal=f'Success in step {i}',
				memory=f'Memory from step {i}',
				next_goal=f'Goal for step {i + 1}',
			),
			action=[ActionModel()],
… (truncated)

```

---

## `browser-use-main\browser_use\agent\message_manager\utils.py`

```py
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.messages import (
	AIMessage,
	BaseMessage,
	HumanMessage,
	SystemMessage,
	ToolMessage,
)

logger = logging.getLogger(__name__)

MODELS_WITHOUT_TOOL_SUPPORT_PATTERNS = [
	'deepseek-reasoner',
	'deepseek-r1',
	'.*gemma.*-it',
]


def is_model_without_tool_support(model_name: str) -> bool:
	return any(re.match(pattern, model_name) for pattern in MODELS_WITHOUT_TOOL_SUPPORT_PATTERNS)


def extract_json_from_model_output(content: str) -> dict:
	"""Extract JSON from model output, handling both plain JSON and code-block-wrapped JSON."""
	try:
		# If content is wrapped in code blocks, extract just the JSON part
		if '```' in content:
			# Find the JSON content between code blocks
			content = content.split('```')[1]
			# Remove language identifier if present (e.g., 'json\n')
			if '\n' in content:
				content = content.split('\n', 1)[1]
		# Parse the cleaned content
		result_dict = json.loads(content)

		# some models occasionally respond with a list containing one dict: https://github.com/browser-use/browser-use/issues/1458
		if isinstance(result_dict, list) and len(result_dict) == 1 and isinstance(result_dict[0], dict):
			result_dict = result_dict[0]

		assert isinstance(result_dict, dict), f'Expected JSON dictionary in response, got JSON {type(result_dict)} instead'
		return result_dict
	except json.JSONDecodeError as e:
		logger.warning(f'Failed to parse model output: {content} {str(e)}')
		raise ValueError('Could not parse response.')


def convert_input_messages(input_messages: list[BaseMessage], model_name: str | None) -> list[BaseMessage]:
	"""Convert input messages to a format that is compatible with the planner model"""
	if model_name is None:
		return input_messages

	if is_model_without_tool_support(model_name):
		converted_input_messages = _convert_messages_for_non_function_calling_models(input_messages)
		merged_input_messages = _merge_successive_messages(converted_input_messages, HumanMessage)
		merged_input_messages = _merge_successive_messages(merged_input_messages, AIMessage)
		return merged_input_messages
	return input_messages


def _convert_messages_for_non_function_calling_models(input_messages: list[BaseMessage]) -> list[BaseMessage]:
	"""Convert messages for non-function-calling models"""
	output_messages = []
	for message in input_messages:
		if isinstance(message, HumanMessage):
			output_messages.append(message)
		elif isinstance(message, SystemMessage):
			output_messages.append(message)
		elif isinstance(message, ToolMessage):
			output_messages.append(HumanMessage(content=message.content))
		elif isinstance(message, AIMessage):
			# check if tool_calls is a valid JSON object
			if message.tool_calls:
				tool_calls = json.dumps(message.tool_calls)
				output_messages.append(AIMessage(content=tool_calls))
			else:
				output_messages.append(message)
		else:
			raise ValueError(f'Unknown message type: {type(message)}')
	return output_messages


def _merge_successive_messages(messages: list[BaseMessage], class_to_merge: type[BaseMessage]) -> list[BaseMessage]:
	"""Some models like deepseek-reasoner dont allow multiple human messages in a row. This function merges them into one."""
	merged_messages = []
	streak = 0
	for message in messages:
		if isinstance(message, class_to_merge):
			streak += 1
			if streak > 1:
				if isinstance(message.content, list):
					merged_messages[-1].content += message.content[0]['text']  # type:ignore
				else:
					merged_messages[-1].content += message.content
			else:
				merged_messages.append(message)
		else:
			merged_messages.append(message)
			streak = 0
	return merged_messages


def save_conversation(input_messages: list[BaseMessage], response: Any, target: str, encoding: str | None = None) -> None:
	"""Save conversation history to file."""

	# create folders if not exists
	if dirname := os.path.dirname(target):
		os.makedirs(dirname, exist_ok=True)

	with open(
		target,
		'w',
		encoding=encoding,
	) as f:
		_write_messages_to_file(f, input_messages)
		_write_response_to_file(f, response)


def _write_messages_to_file(f: Any, messages: list[BaseMessage]) -> None:
	"""Write messages to conversation file"""
	for message in messages:
		f.write(f' {message.__class__.__name__} \n')

		if isinstance(message.content, list):
			for item in message.content:
				if isinstance(item, dict) and item.get('type') == 'text':
					f.write(item['text'].strip() + '\n')
		elif isinstance(message.content, str):
			try:
				content = json.loads(message.content)
				f.write(json.dumps(content, indent=2) + '\n')
			except json.JSONDecodeError:
				f.write(message.content.strip() + '\n')

		f.write('\n')


def _write_response_to_file(f: Any, response: Any) -> None:
	"""Write model response to conversation file"""
	f.write(' RESPONSE\n')
	f.write(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))

```

---

## `browser-use-main\browser_use\agent\message_manager\views.py`

```py
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import filterwarnings

from langchain_core._api import LangChainBetaWarning
from langchain_core.load import dumpd, load
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

filterwarnings('ignore', category=LangChainBetaWarning)

if TYPE_CHECKING:
	from browser_use.agent.views import AgentOutput


class MessageMetadata(BaseModel):
	"""Metadata for a message"""

	tokens: int = 0
	message_type: str | None = None


class ManagedMessage(BaseModel):
	"""A message with its metadata"""

	message: BaseMessage
	metadata: MessageMetadata = Field(default_factory=MessageMetadata)

	model_config = ConfigDict(arbitrary_types_allowed=True)

	# https://github.com/pydantic/pydantic/discussions/7558
	@model_serializer(mode='wrap')
	def to_json(self, original_dump):
		"""
		Returns the JSON representation of the model.

		It uses langchain's `dumps` function to serialize the `message`
		property before encoding the overall dict with json.dumps.
		"""
		data = original_dump(self)

		# NOTE: We override the message field to use langchain JSON serialization.
		data['message'] = dumpd(self.message)

		return data

	@model_validator(mode='before')
	@classmethod
	def validate(
		cls,
		value: Any,
		*,
		strict: bool | None = None,
		from_attributes: bool | None = None,
		context: Any | None = None,
	) -> Any:
		"""
		Custom validator that uses langchain's `loads` function
		to parse the message if it is provided as a JSON string.
		"""
		if isinstance(value, dict) and 'message' in value:
			# NOTE: We use langchain's load to convert the JSON string back into a BaseMessage object.
			filterwarnings('ignore', category=LangChainBetaWarning)
			value['message'] = load(value['message'])
		return value


class MessageHistory(BaseModel):
	"""History of messages with metadata"""

	messages: list[ManagedMessage] = Field(default_factory=list)
	current_tokens: int = 0

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def add_message(self, message: BaseMessage, metadata: MessageMetadata, position: int | None = None) -> None:
		"""Add message with metadata to history"""
		if position is None:
			self.messages.append(ManagedMessage(message=message, metadata=metadata))
		else:
			self.messages.insert(position, ManagedMessage(message=message, metadata=metadata))
		self.current_tokens += metadata.tokens

	def add_model_output(self, output: AgentOutput) -> None:
		"""Add model output as AI message"""
		tool_calls = [
			{
				'name': 'AgentOutput',
				'args': output.model_dump(mode='json', exclude_unset=True),
				'id': '1',
				'type': 'tool_call',
			}
		]

		msg = AIMessage(
			content='',
			tool_calls=tool_calls,
		)
		self.add_message(msg, MessageMetadata(tokens=100))  # Estimate tokens for tool calls

		# Empty tool response
		tool_message = ToolMessage(content='', tool_call_id='1')
		self.add_message(tool_message, MessageMetadata(tokens=10))  # Estimate tokens for empty response

	def get_messages(self) -> list[BaseMessage]:
		"""Get all messages"""
		return [m.message for m in self.messages]

	def get_total_tokens(self) -> int:
		"""Get total tokens in history"""
		return self.current_tokens

	def remove_oldest_message(self) -> None:
		"""Remove oldest non-system message"""
		for i, msg in enumerate(self.messages):
			if not isinstance(msg.message, SystemMessage):
				self.current_tokens -= msg.metadata.tokens
				self.messages.pop(i)
				break

	def remove_last_state_message(self) -> None:
		"""Remove last state message from history"""
		if len(self.messages) > 2 and isinstance(self.messages[-1].message, HumanMessage):
			self.current_tokens -= self.messages[-1].metadata.tokens
			self.messages.pop()


class MessageManagerState(BaseModel):
	"""Holds the state for MessageManager"""

	history: MessageHistory = Field(default_factory=MessageHistory)
	tool_id: int = 1

	model_config = ConfigDict(arbitrary_types_allowed=True)

```

---

## `browser-use-main\browser_use\agent\playwright_script_generator.py`

```py
# import json
# import logging
# from pathlib import Path
# from typing import Any

# from browser_use.browser import BrowserConfig, BrowserContextConfig

# logger = logging.getLogger(__name__)


# class PlaywrightScriptGenerator:
# 	"""Generates a Playwright script from AgentHistoryList."""

# 	def __init__(
# 		self,
# 		history_list: list[dict[str, Any]],
# 		sensitive_data_keys: list[str] | None = None,
# 		browser_config: BrowserConfig | None = None,
# 		context_config: BrowserContextConfig | None = None,
# 	):
# 		"""
# 		Initializes the script generator.

# 		Args:
# 		    history_list: A list of dictionaries, where each dictionary represents an AgentHistory item.
# 		                 Expected to be raw dictionaries from `AgentHistoryList.model_dump()`.
# 		    sensitive_data_keys: A list of keys used as placeholders for sensitive data.
# 		    browser_config: Configuration from the original Browser instance (deprecated, use BrowserProfile).
# 		    context_config: Configuration from the original BrowserContext instance (deprecated, use BrowserProfile).
# 		"""
# 		self.history = history_list
# 		self.sensitive_data_keys = sensitive_data_keys or []
# 		self.browser_config = browser_config
# 		self.context_config = context_config
# 		self._imports_helpers_added = False
# 		self._page_counter = 0  # Track pages for tab management

# 		# Dictionary mapping action types to handler methods
# 		self._action_handlers = {
# 			'go_to_url': self._map_go_to_url,
# 			'wait': self._map_wait,
# 			'input_text': self._map_input_text,
# 			'click_element': self._map_click_element,
# 			'click_element_by_index': self._map_click_element,  # Map legacy action
# 			'scroll_down': self._map_scroll_down,
# 			'scroll_up': self._map_scroll_up,
# 			'send_keys': self._map_send_keys,
# 			'go_back': self._map_go_back,
# 			'open_tab': self._map_open_tab,
# 			'close_tab': self._map_close_tab,
# 			'switch_tab': self._map_switch_tab,
# 			'search_google': self._map_search_google,
# 			'drag_drop': self._map_drag_drop,
# 			'extract_content': self._map_extract_content,
# 			'click_download_button': self._map_click_download_button,
# 			'done': self._map_done,
# 		}

# 	def _generate_browser_launch_args(self) -> str:
# 		"""Generates the arguments string for browser launch based on BrowserConfig."""
# 		if not self.browser_config:
# 			# Default launch if no config provided
# 			return 'headless=False'

# 		args_dict = {
# 			'headless': self.browser_config.headless,
# 			# Add other relevant launch options here based on self.browser_config
# 			# Example: 'proxy': self.browser_config.proxy.model_dump() if self.browser_config.proxy else None
# 			# Example: 'args': self.browser_config.extra_browser_args # Be careful inheriting args
# 		}
# 		if self.browser_config.proxy:
# 			args_dict['proxy'] = self.browser_config.proxy.model_dump()

# 		# Filter out None values
# 		args_dict = {k: v for k, v in args_dict.items() if v is not None}

# 		# Format as keyword arguments string
# 		args_str = ', '.join(f'{key}={repr(value)}' for key, value in args_dict.items())
# 		return args_str

# 	def _generate_context_options(self) -> str:
# 		"""Generates the options string for context creation based on BrowserContextConfig."""
# 		if not self.context_config:
# 			return ''  # Default context

# 		options_dict = {}

# 		# Map relevant BrowserContextConfig fields to Playwright context options
# 		if self.context_config.user_agent:
# 			options_dict['user_agent'] = self.context_config.user_agent
# 		if self.context_config.locale:
# 			options_dict['locale'] = self.context_config.locale
# 		if self.context_config.permissions:
# 			options_dict['permissions'] = self.context_config.permissions
# 		if self.context_config.geolocation:
# 			options_dict['geolocation'] = self.context_config.geolocation
# 		if self.context_config.timezone_id:
# 			options_dict['timezone_id'] = self.context_config.timezone_id
# 		if self.context_config.http_credentials:
# 			options_dict['http_credentials'] = self.context_config.http_credentials
# 		if self.context_config.is_mobile is not None:
# 			options_dict['is_mobile'] = self.context_config.is_mobile
# 		if self.context_config.has_touch is not None:
# 			options_dict['has_touch'] = self.context_config.has_touch
# 		if self.context_config.save_recording_path:
# 			options_dict['record_video_dir'] = self.context_config.save_recording_path
# 		if self.context_config.save_har_path:
# 			options_dict['record_har_path'] = self.context_config.save_har_path

# 		# Handle viewport/window size
# 		if self.context_config.no_viewport:
# 			options_dict['no_viewport'] = True
# 		elif hasattr(self.context_config, 'window_width') and hasattr(self.context_config, 'window_height'):
# 			options_dict['viewport'] = {
# 				'width': self.context_config.window_width,
# 				'height': self.context_config.window_height,
# 			}

# 		# Note: cookies_file and save_downloads_path are handled separately

# 		# Filter out None values
# 		options_dict = {k: v for k, v in options_dict.items() if v is not None}

# 		# Format as keyword arguments string
# 		options_str = ', '.join(f'{key}={repr(value)}' for key, value in options_dict.items())
# 		return options_str

# 	def _get_imports_and_helpers(self) -> list[str]:
# 		"""Generates necessary import statements (excluding helper functions)."""
# 		# Return only the standard imports needed by the main script body
# 		return [
# 			'import asyncio',
# 			'import json',
# 			'import os',
# 			'import sys',
# 			'from pathlib import Path',  # Added Path import
# 			'import urllib.parse',  # Needed for search_google
# 			'from playwright.async_api import async_playwright, Page, BrowserContext',  # Added BrowserContext
# 			'from dotenv import load_dotenv',
# 			'',
# 			'# Load environment variables',
# 			'load_dotenv(override=True)',
# 			'',
# 			# Helper function definitions are no longer here
# 		]

# 	def _get_sensitive_data_definitions(self) -> list[str]:
# 		"""Generates the SENSITIVE_DATA dictionary definition."""
# 		if not self.sensitive_data_keys:
# 			return ['SENSITIVE_DATA = {}', '']

# 		lines = ['# Sensitive data placeholders mapped to environment variables']
# 		lines.append('SENSITIVE_DATA = {')
# 		for key in self.sensitive_data_keys:
# 			env_var_name = key.upper()
# 			default_value_placeholder = f'YOUR_{env_var_name}'
# 			lines.append(f'    "{key}": os.getenv("{env_var_name}", {json.dumps(default_value_placeholder)}),')
# 		lines.append('}')
# 		lines.append('')
# 		return lines

# 	def _get_selector_for_action(self, history_item: dict, action_index_in_step: int) -> str | None:
# 		"""
# 		Gets the selector (preferring XPath) for a given action index within a history step.
# 		Formats the XPath correctly for Playwright.
# 		"""
# 		state = history_item.get('state')
# 		if not isinstance(state, dict):
# 			return None
# 		interacted_elements = state.get('interacted_element')
# 		if not isinstance(interacted_elements, list):
# 			return None
# 		if action_index_in_step >= len(interacted_elements):
# 			return None
# 		element_data = interacted_elements[action_index_in_step]
# 		if not isinstance(element_data, dict):
# 			return None

# 		# Prioritize XPath
# 		xpath = element_data.get('xpath')
# 		if isinstance(xpath, str) and xpath.strip():
# 			if not xpath.startswith('xpath=') and not xpath.startswith('/') and not xpath.startswith('//'):
# 				xpath_selector = f'xpath=//{xpath}'  # Make relative if not already
# 			elif not xpath.startswith('xpath='):
# 				xpath_selector = f'xpath={xpath}'  # Add prefix if missing
# 			else:
# 				xpath_selector = xpath
# 			return xpath_selector

# 		# Fallback to CSS selector if XPath is missing
# 		css_selector = element_data.get('css_selector')
# 		if isinstance(css_selector, str) and css_selector.strip():
# 			return css_selector  # Use CSS selector as is

# 		logger.warning(
# 			f'Could not find a usable XPath or CSS selector for action index {action_index_in_step} (element i
… (truncated)

```

---

## `browser-use-main\browser_use\agent\playwright_script_helpers.py`

```py
from playwright.async_api import Page


# --- Helper Function for Replacing Sensitive Data ---
def replace_sensitive_data(text: str, sensitive_map: dict) -> str:
	"""Replaces sensitive data placeholders in text."""
	if not isinstance(text, str):
		return text
	for placeholder, value in sensitive_map.items():
		replacement_value = str(value) if value is not None else ''
		text = text.replace(f'<secret>{placeholder}</secret>', replacement_value)
	return text


# --- Helper Function for Robust Action Execution ---
class PlaywrightActionError(Exception):
	"""Custom exception for errors during Playwright script action execution."""

	pass


async def _try_locate_and_act(page: Page, selector: str, action_type: str, text: str | None = None, step_info: str = '') -> None:
	"""
	Attempts an action (click/fill) with XPath fallback by trimming prefixes.
	Raises PlaywrightActionError if the action fails after all fallbacks.
	"""
	print(f'Attempting {action_type} ({step_info}) using selector: {repr(selector)}')
	original_selector = selector
	MAX_FALLBACKS = 50  # Increased fallbacks
	# Increased timeouts for potentially slow pages
	INITIAL_TIMEOUT = 10000  # Milliseconds for the first attempt (10 seconds)
	FALLBACK_TIMEOUT = 1000  # Shorter timeout for fallback attempts (1 second)

	try:
		locator = page.locator(selector).first
		if action_type == 'click':
			await locator.click(timeout=INITIAL_TIMEOUT)
		elif action_type == 'fill' and text is not None:
			await locator.fill(text, timeout=INITIAL_TIMEOUT)
		else:
			# This case should ideally not happen if called correctly
			raise PlaywrightActionError(f"Invalid action_type '{action_type}' or missing text for fill. ({step_info})")
		print(f"  Action '{action_type}' successful with original selector.")
		await page.wait_for_timeout(500)  # Wait after successful action
		return  # Successful exit
	except Exception as e:
		print(f"  Warning: Action '{action_type}' failed with original selector ({repr(selector)}): {e}. Starting fallback...")

		# Fallback only works for XPath selectors
		if not selector.startswith('xpath='):
			# Raise error immediately if not XPath, as fallback won't work
			raise PlaywrightActionError(
				f"Action '{action_type}' failed. Fallback not possible for non-XPath selector: {repr(selector)}. ({step_info})"
			)

		xpath_parts = selector.split('=', 1)
		if len(xpath_parts) < 2:
			raise PlaywrightActionError(
				f"Action '{action_type}' failed. Could not extract XPath string from selector: {repr(selector)}. ({step_info})"
			)
		xpath = xpath_parts[1]  # Correctly get the XPath string

		segments = [seg for seg in xpath.split('/') if seg]

		for i in range(1, min(MAX_FALLBACKS + 1, len(segments))):
			trimmed_xpath_raw = '/'.join(segments[i:])
			fallback_xpath = f'xpath=//{trimmed_xpath_raw}'

			print(f'    Fallback attempt {i}/{MAX_FALLBACKS}: Trying selector: {repr(fallback_xpath)}')
			try:
				locator = page.locator(fallback_xpath).first
				if action_type == 'click':
					await locator.click(timeout=FALLBACK_TIMEOUT)
				elif action_type == 'fill' and text is not None:
					try:
						await locator.clear(timeout=FALLBACK_TIMEOUT)
						await page.wait_for_timeout(100)
					except Exception as clear_error:
						print(f'    Warning: Failed to clear field during fallback ({step_info}): {clear_error}')
					await locator.fill(text, timeout=FALLBACK_TIMEOUT)

				print(f"    Action '{action_type}' successful with fallback selector: {repr(fallback_xpath)}")
				await page.wait_for_timeout(500)
				return  # Successful exit after fallback
			except Exception as fallback_e:
				print(f'    Fallback attempt {i} failed: {fallback_e}')
				if i == MAX_FALLBACKS:
					# Raise exception after exhausting fallbacks
					raise PlaywrightActionError(
						f"Action '{action_type}' failed after {MAX_FALLBACKS} fallback attempts. Original selector: {repr(original_selector)}. ({step_info})"
					)

	# This part should not be reachable if logic is correct, but added as safeguard
	raise PlaywrightActionError(f"Action '{action_type}' failed unexpectedly for {repr(original_selector)}. ({step_info})")

```

---

## `browser-use-main\browser_use\agent\prompts.py`

```py
import importlib.resources
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
	from browser_use.agent.views import ActionResult, AgentStepInfo
	from browser_use.browser.views import BrowserStateSummary


class SystemPrompt:
	def __init__(
		self,
		action_description: str,
		max_actions_per_step: int = 10,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
	):
		self.default_action_description = action_description
		self.max_actions_per_step = max_actions_per_step
		prompt = ''
		if override_system_message:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

		if extend_system_message:
			prompt += f'\n{extend_system_message}'

		self.system_message = SystemMessage(content=prompt)

	def _load_prompt_template(self) -> None:
		"""Load the prompt template from the markdown file."""
		try:
			# This works both in development and when installed as a package
			with importlib.resources.files('browser_use.agent').joinpath('system_prompt.md').open('r') as f:
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template: {e}')

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    SystemMessage: Formatted system prompt
		"""
		return self.system_message


# Functions:
# {self.default_action_description}

# Example:
# {self.example_response()}
# Your AVAILABLE ACTIONS:
# {self.default_action_description}


class AgentMessagePrompt:
	def __init__(
		self,
		browser_state_summary: 'BrowserStateSummary',
		result: list['ActionResult'] | None = None,
		include_attributes: list[str] | None = None,
		step_info: Optional['AgentStepInfo'] = None,
	):
		self.state: 'BrowserStateSummary' = browser_state_summary
		self.result = result
		self.include_attributes = include_attributes or []
		self.step_info = step_info
		assert self.state

	def get_user_message(self, use_vision: bool = True) -> HumanMessage:
		elements_text = self.state.element_tree.clickable_elements_to_string(include_attributes=self.include_attributes)

		has_content_above = (self.state.pixels_above or 0) > 0
		has_content_below = (self.state.pixels_below or 0) > 0

		if elements_text != '':
			if has_content_above:
				elements_text = (
					f'... {self.state.pixels_above} pixels above - scroll or extract content to see more ...\n{elements_text}'
				)
			else:
				elements_text = f'[Start of page]\n{elements_text}'
			if has_content_below:
				elements_text = (
					f'{elements_text}\n... {self.state.pixels_below} pixels below - scroll or extract content to see more ...'
				)
			else:
				elements_text = f'{elements_text}\n[End of page]'
		else:
			elements_text = 'empty page'

		if self.step_info:
			step_info_description = f'Current step: {self.step_info.step_number + 1}/{self.step_info.max_steps}'
		else:
			step_info_description = ''
		time_str = datetime.now().strftime('%Y-%m-%d %H:%M')
		step_info_description += f'Current date and time: {time_str}'

		state_description = f"""
[Task history memory ends]
[Current state starts here]
The following is one-time information - if you need to remember it write it to memory:
Current url: {self.state.url}
Available tabs:
{self.state.tabs}
Interactive elements from top layer of the current page inside the viewport:
{elements_text}
{step_info_description}
"""

		if self.result:
			for i, result in enumerate(self.result):
				if result.extracted_content:
					state_description += f'\nAction result {i + 1}/{len(self.result)}: {result.extracted_content}'
				if result.error:
					# only use last line of error
					error = result.error.split('\n')[-1]
					state_description += f'\nAction error {i + 1}/{len(self.result)}: ...{error}'

		if self.state.screenshot and use_vision is True:
			# Format message for vision model
			return HumanMessage(
				content=[
					{'type': 'text', 'text': state_description},
					{
						'type': 'image_url',
						'image_url': {'url': f'data:image/png;base64,{self.state.screenshot}'},  # , 'detail': 'low'
					},
				]
			)

		return HumanMessage(content=state_description)


class PlannerPrompt(SystemPrompt):
	def __init__(self, available_actions: str):
		self.available_actions = available_actions

	def get_system_message(
		self, is_planner_reasoning: bool, extended_planner_system_prompt: str | None = None
	) -> SystemMessage | HumanMessage:
		"""Get the system message for the planner.

		Args:
		    is_planner_reasoning: If True, return as HumanMessage for chain-of-thought
		    extended_planner_system_prompt: Optional text to append to the base prompt

		Returns:
		    SystemMessage or HumanMessage depending on is_planner_reasoning
		"""

		planner_prompt_text = """
You are a planning agent that helps break down tasks into smaller steps and reason about the current state.
Your role is to:
1. Analyze the current state and history
2. Evaluate progress towards the ultimate goal
3. Identify potential challenges or roadblocks
4. Suggest the next high-level steps to take

Inside your messages, there will be AI messages from different agents with different formats.

Your output format should be always a JSON object with the following fields:
{{
    "state_analysis": "Brief analysis of the current state and what has been done so far",
    "progress_evaluation": "Evaluation of progress towards the ultimate goal (as percentage and description)",
    "challenges": "List any potential challenges or roadblocks",
    "next_steps": "List 2-3 concrete next steps to take",
    "reasoning": "Explain your reasoning for the suggested next steps"
}}

Ignore the other AI messages output structures.

Keep your responses concise and focused on actionable insights.
"""

		if extended_planner_system_prompt:
			planner_prompt_text += f'\n{extended_planner_system_prompt}'

		if is_planner_reasoning:
			return HumanMessage(content=planner_prompt_text)
		else:
			return SystemMessage(content=planner_prompt_text)

```

---

## `browser-use-main\browser_use\agent\service.py`

```py
import asyncio
import gc
import inspect
import json
import logging
import os
import re
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Generic, TypeVar

from dotenv import load_dotenv

from browser_use.browser.session import DEFAULT_BROWSER_PROFILE

load_dotenv()

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
	BaseMessage,
	HumanMessage,
	SystemMessage,
)
from playwright.async_api import Browser, BrowserContext
from pydantic import BaseModel, ValidationError

from browser_use.agent.gif import create_history_gif
from browser_use.agent.memory import Memory, MemoryConfig
from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.message_manager.utils import (
	convert_input_messages,
	extract_json_from_model_output,
	is_model_without_tool_support,
	save_conversation,
)
from browser_use.agent.prompts import AgentMessagePrompt, PlannerPrompt, SystemPrompt
from browser_use.agent.views import (
	REQUIRED_LLM_API_ENV_VARS,
	ActionResult,
	AgentError,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentSettings,
	AgentState,
	AgentStepInfo,
	BrowserStateHistory,
	StepMetadata,
	ToolCallingMethod,
)
from browser_use.browser import BrowserProfile, BrowserSession

# from lmnr.sdk.decorators import observe
from browser_use.browser.views import BrowserStateSummary
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import (
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from browser_use.exceptions import LLMException
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
	AgentTelemetryEvent,
)
from browser_use.utils import check_env_variables, time_execution_async, time_execution_sync

logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = os.environ.get('SKIP_LLM_API_KEY_VERIFICATION', 'false').lower()[0] in 'ty1'


def log_response(response: AgentOutput) -> None:
	"""Utility function to log the model's response."""

	if 'Success' in response.current_state.evaluation_previous_goal:
		emoji = '👍'
	elif 'Failed' in response.current_state.evaluation_previous_goal:
		emoji = '⚠'
	else:
		emoji = '🤷'

	logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
	logger.info(f'🧠 Memory: {response.current_state.memory}')
	logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
	for i, action in enumerate(response.action):
		logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')


Context = TypeVar('Context')

AgentHookFunc = Callable[['Agent'], Awaitable[None]]


class Agent(Generic[Context]):
	@time_execution_sync('--init (agent)')
	def __init__(
		self,
		task: str,
		llm: BaseChatModel,
		# Optional parameters
		browser: Browser | None = None,
		browser_context: BrowserContext | None = None,
		browser_profile: BrowserProfile | None = None,
		browser_session: BrowserSession | None = None,
		controller: Controller[Context] = Controller(),
		# Initial agent run parameters
		sensitive_data: dict[str, str] | None = None,
		initial_actions: list[dict[str, dict[str, Any]]] | None = None,
		# Cloud Callbacks
		register_new_step_callback: (
			Callable[['BrowserStateSummary', 'AgentOutput', int], None]  # Sync callback
			| Callable[['BrowserStateSummary', 'AgentOutput', int], Awaitable[None]]  # Async callback
			| None
		) = None,
		register_done_callback: (
			Callable[['AgentHistoryList'], Awaitable[None]]  # Async Callback
			| Callable[['AgentHistoryList'], None]  # Sync Callback
			| None
		) = None,
		register_external_agent_status_raise_error_callback: Callable[[], Awaitable[bool]] | None = None,
		# Agent settings
		use_vision: bool = True,
		use_vision_for_planner: bool = False,
		save_conversation_path: str | None = None,
		save_conversation_path_encoding: str | None = 'utf-8',
		max_failures: int = 3,
		retry_delay: int = 10,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		max_input_tokens: int = 128000,
		validate_output: bool = False,
		message_context: str | None = None,
		generate_gif: bool | str = False,
		available_file_paths: list[str] | None = None,
		include_attributes: list[str] = [
			'title',
			'type',
			'name',
			'role',
			'aria-label',
			'placeholder',
			'value',
			'alt',
			'aria-expanded',
			'data-date-format',
		],
		max_actions_per_step: int = 10,
		tool_calling_method: ToolCallingMethod | None = 'auto',
		page_extraction_llm: BaseChatModel | None = None,
		planner_llm: BaseChatModel | None = None,
		planner_interval: int = 1,  # Run planner every N steps
		is_planner_reasoning: bool = False,
		extend_planner_system_message: str | None = None,
		injected_agent_state: AgentState | None = None,
		context: Context | None = None,
		save_playwright_script_path: str | None = None,
		enable_memory: bool = True,
		memory_config: MemoryConfig | None = None,
		source: str | None = None,
	):
		if page_extraction_llm is None:
			page_extraction_llm = llm

		# Core components
		self.task = task
		self.llm = llm
		self.controller = controller
		self.sensitive_data = sensitive_data

		self.settings = AgentSettings(
			use_vision=use_vision,
			use_vision_for_planner=use_vision_for_planner,
			save_conversation_path=save_conversation_path,
			save_conversation_path_encoding=save_conversation_path_encoding,
			max_failures=max_failures,
			retry_delay=retry_delay,
			override_system_message=override_system_message,
			extend_system_message=extend_system_message,
			max_input_tokens=max_input_tokens,
			validate_output=validate_output,
			message_context=message_context,
			generate_gif=generate_gif,
			available_file_paths=available_file_paths,
			include_attributes=include_attributes,
			max_actions_per_step=max_actions_per_step,
			tool_calling_method=tool_calling_method,
			page_extraction_llm=page_extraction_llm,
			planner_llm=planner_llm,
			planner_interval=planner_interval,
			is_planner_reasoning=is_planner_reasoning,
			save_playwright_script_path=save_playwright_script_path,
			extend_planner_system_message=extend_planner_system_message,
		)

		# Memory settings
		self.enable_memory = enable_memory
		self.memory_config = memory_config
… (truncated)

```

---

## `browser-use-main\browser_use\agent\system_prompt.md`

```md
You are an AI agent designed to automate browser tasks. Your goal is to accomplish the ultimate task following the rules.

# Input Format

Task
Previous steps
Current URL
Open Tabs
Interactive Elements
[index]<type>text</type>

- index: Numeric identifier for interaction
- type: HTML element type (button, input, etc.)
- text: Element description
  Example:
  [33]<div>User form</div>
  \t*[35]*<button aria-label='Submit form'>Submit</button>

- Only elements with numeric indexes in [] are interactive
- (stacked) indentation (with \t) is important and means that the element is a (html) child of the element above (with a lower index)
- Elements with \* are new elements that were added after the previous step (if url has not changed)

# Response Rules

1. RESPONSE FORMAT: You must ALWAYS respond with valid JSON in this exact format:
   {{"current_state": {{"evaluation_previous_goal": "Success|Failed|Unknown - Analyze the current elements and the image to check if the previous goals/actions are successful like intended by the task. Mention if something unexpected happened. Shortly state why/why not",
   "memory": "Description of what has been done and what you need to remember. Be very specific. Count here ALWAYS how many times you have done something and how many remain. E.g. 0 out of 10 websites analyzed. Continue with abc and xyz",
   "next_goal": "What needs to be done with the next immediate action"}},
   "action":[{{"one_action_name": {{// action-specific parameter}}}}, // ... more actions in sequence]}}

2. ACTIONS: You can specify multiple actions in the list to be executed in sequence. But always specify only one action name per item. Use maximum {max_actions} actions per sequence.
Common action sequences:

- Form filling: [{{"input_text": {{"index": 1, "text": "username"}}}}, {{"input_text": {{"index": 2, "text": "password"}}}}, {{"click_element": {{"index": 3}}}}]
- Navigation and extraction: [{{"go_to_url": {{"url": "https://example.com"}}}}, {{"extract_content": {{"goal": "extract the names"}}}}]
- Actions are executed in the given order
- If the page changes after an action, the sequence is interrupted and you get the new state.
- Only provide the action sequence until an action which changes the page state significantly.
- Try to be efficient, e.g. fill forms at once, or chain actions where nothing changes on the page
- only use multiple actions if it makes sense.

3. ELEMENT INTERACTION:

- Only use indexes of the interactive elements

4. NAVIGATION & ERROR HANDLING:

- If no suitable elements exist, use other functions to complete the task
- If stuck, try alternative approaches - like going back to a previous page, new search, new tab etc.
- Handle popups/cookies by accepting or closing them
- Use scroll to find elements you are looking for
- If you want to research something, open a new tab instead of using the current tab
- If captcha pops up, try to solve it - else try a different approach
- If the page is not fully loaded, use wait action

5. TASK COMPLETION:

- Use the done action as the last action as soon as the ultimate task is complete
- Dont use "done" before you are done with everything the user asked you, except you reach the last step of max_steps.
- If you reach your last step, use the done action even if the task is not fully finished. Provide all the information you have gathered so far. If the ultimate task is completely finished set success to true. If not everything the user asked for is completed set success in done to false!
- If you have to do something repeatedly for example the task says for "each", or "for all", or "x times", count always inside "memory" how many times you have done it and how many remain. Don't stop until you have completed like the task asked you. Only call done after the last step.
- Don't hallucinate actions
- Make sure you include everything you found out for the ultimate task in the done text parameter. Do not just say you are done, but include the requested information of the task.

6. VISUAL CONTEXT:

- When an image is provided, use it to understand the page layout
- Bounding boxes with labels on their top right corner correspond to element indexes

7. Form filling:

- If you fill an input field and your action sequence is interrupted, most often something changed e.g. suggestions popped up under the field.

8. Long tasks:

- Keep track of the status and subresults in the memory.
- You are provided with procedural memory summaries that condense previous task history (every N steps). Use these summaries to maintain context about completed actions, current progress, and next steps. The summaries appear in chronological order and contain key information about navigation history, findings, errors encountered, and current state. Refer to these summaries to avoid repeating actions and to ensure consistent progress toward the task goal.

9. Extraction:

- If your task is to find information - call extract_content on the specific pages to get and store the information.
  Your responses must be always JSON with the specified format.

```

---

## `browser-use-main\browser_use\agent\tests.py`

```py
import pytest

from browser_use.agent.views import (
	ActionResult,
	AgentBrain,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
)
from browser_use.browser.views import BrowserStateHistory, BrowserStateSummary, TabInfo
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import ClickElementAction, DoneAction, ExtractPageContentAction
from browser_use.dom.views import DOMElementNode


@pytest.fixture
def sample_browser_state():
	return BrowserStateSummary(
		url='https://example.com',
		title='Example Page',
		tabs=[TabInfo(url='https://example.com', title='Example Page', page_id=1)],
		screenshot='screenshot1.png',
		element_tree=DOMElementNode(
			tag_name='root',
			is_visible=True,
			parent=None,
			xpath='',
			attributes={},
			children=[],
		),
		selector_map={},
	)


@pytest.fixture
def action_registry():
	registry = Registry()

	# Register the actions we need for testing
	@registry.action(description='Click an element', param_model=ClickElementAction)
	def click_element(params: ClickElementAction, browser=None):
		pass

	@registry.action(
		description='Extract page content',
		param_model=ExtractPageContentAction,
	)
	def extract_page_content(params: ExtractPageContentAction, browser=None):
		pass

	@registry.action(description='Mark task as done', param_model=DoneAction)
	def done(params: DoneAction):
		pass

	# Create the dynamic ActionModel with all registered actions
	return registry.create_action_model()


@pytest.fixture
def sample_history(action_registry):
	# Create actions with nested params structure
	click_action = action_registry(click_element={'index': 1})

	extract_action = action_registry(extract_page_content={'value': 'text'})

	done_action = action_registry(done={'text': 'Task completed'})

	histories = [
		AgentHistory(
			model_output=AgentOutput(
				current_state=AgentBrain(
					evaluation_previous_goal='None',
					memory='Started task',
					next_goal='Click button',
				),
				action=[click_action],
			),
			result=[ActionResult(is_done=False)],
			state=BrowserStateHistory(
				url='https://example.com',
				title='Page 1',
				tabs=[TabInfo(url='https://example.com', title='Page 1', page_id=1)],
				screenshot='screenshot1.png',
				interacted_element=[{'xpath': '//button[1]'}],
			),
		),
		AgentHistory(
			model_output=AgentOutput(
				current_state=AgentBrain(
					evaluation_previous_goal='Clicked button',
					memory='Button clicked',
					next_goal='Extract content',
				),
				action=[extract_action],
			),
			result=[
				ActionResult(
					is_done=False,
					extracted_content='Extracted text',
					error='Failed to extract completely',
				)
			],
			state=BrowserStateHistory(
				url='https://example.com/page2',
				title='Page 2',
				tabs=[TabInfo(url='https://example.com/page2', title='Page 2', page_id=2)],
				screenshot='screenshot2.png',
				interacted_element=[{'xpath': '//div[1]'}],
			),
		),
		AgentHistory(
			model_output=AgentOutput(
				current_state=AgentBrain(
					evaluation_previous_goal='Extracted content',
					memory='Content extracted',
					next_goal='Finish task',
				),
				action=[done_action],
			),
			result=[ActionResult(is_done=True, extracted_content='Task completed', error=None)],
			state=BrowserStateHistory(
				url='https://example.com/page2',
				title='Page 2',
				tabs=[TabInfo(url='https://example.com/page2', title='Page 2', page_id=2)],
				screenshot='screenshot3.png',
				interacted_element=[{'xpath': '//div[1]'}],
			),
		),
	]
	return AgentHistoryList(history=histories)


def test_last_model_output(sample_history: AgentHistoryList):
	last_output = sample_history.last_action()
	print(last_output)
	assert last_output == {'done': {'text': 'Task completed'}}


def test_get_errors(sample_history: AgentHistoryList):
	errors = sample_history.errors()
	assert len(errors) == 1
	assert errors[0] == 'Failed to extract completely'


def test_final_result(sample_history: AgentHistoryList):
	assert sample_history.final_result() == 'Task completed'


def test_is_done(sample_history: AgentHistoryList):
	assert sample_history.is_done() is True


def test_urls(sample_history: AgentHistoryList):
	urls = sample_history.urls()
	assert 'https://example.com' in urls
	assert 'https://example.com/page2' in urls


def test_all_screenshots(sample_history: AgentHistoryList):
	screenshots = sample_history.screenshots()
	assert len(screenshots) == 3
	assert screenshots == ['screenshot1.png', 'screenshot2.png', 'screenshot3.png']


def test_all_model_outputs(sample_history: AgentHistoryList):
	outputs = sample_history.model_actions()
	print(f'DEBUG: {outputs[0]}')
	assert len(outputs) == 3
	# get first key value pair
	assert dict([next(iter(outputs[0].items()))]) == {'click_element': {'index': 1}}
	assert dict([next(iter(outputs[1].items()))]) == {'extract_page_content': {'value': 'text'}}
	assert dict([next(iter(outputs[2].items()))]) == {'done': {'text': 'Task completed'}}


def test_all_model_outputs_filtered(sample_history: AgentHistoryList):
	filtered = sample_history.model_actions_filtered(include=['click_element'])
	assert len(filtered) == 1
	assert filtered[0]['click_element']['index'] == 1


def test_empty_history():
	empty_history = AgentHistoryList(history=[])
	assert empty_history.last_action() is None
	assert empty_history.final_result() is None
	assert empty_history.is_done() is False
	assert len(empty_history.urls()) == 0


# Add a test to verify action creation
def test_action_creation(action_registry):
	click_action = action_registry(click_element={'index': 1})

	assert click_action.model_dump(exclude_none=True) == {'click_element': {'index': 1}}


# run this with:
# pytest browser_use/agent/tests.py

```

---

## `browser-use-main\browser_use\agent\views.py`

```py
from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.browser.views import BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.dom.history_tree_processor.service import (
	DOMElementNode,
	DOMHistoryElement,
	HistoryTreeProcessor,
)
from browser_use.dom.views import SelectorMap

ToolCallingMethod = Literal['function_calling', 'json_mode', 'raw', 'auto', 'tools']
REQUIRED_LLM_API_ENV_VARS = {
	'ChatOpenAI': ['OPENAI_API_KEY'],
	'AzureChatOpenAI': ['AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_KEY'],
	'ChatBedrockConverse': ['ANTHROPIC_API_KEY'],
	'ChatAnthropic': ['ANTHROPIC_API_KEY'],
	'ChatGoogleGenerativeAI': ['GOOGLE_API_KEY'],
	'ChatDeepSeek': ['DEEPSEEK_API_KEY'],
	'ChatOllama': [],
	'ChatGrok': ['GROK_API_KEY'],
}


class AgentSettings(BaseModel):
	"""Options for the agent"""

	use_vision: bool = True
	use_vision_for_planner: bool = False
	save_conversation_path: str | None = None
	save_conversation_path_encoding: str | None = 'utf-8'
	max_failures: int = 3
	retry_delay: int = 10
	max_input_tokens: int = 128000
	validate_output: bool = False
	message_context: str | None = None
	generate_gif: bool | str = False
	available_file_paths: list[str] | None = None
	override_system_message: str | None = None
	extend_system_message: str | None = None
	include_attributes: list[str] = [
		'title',
		'type',
		'name',
		'role',
		'tabindex',
		'aria-label',
		'placeholder',
		'value',
		'alt',
		'aria-expanded',
	]
	max_actions_per_step: int = 10

	tool_calling_method: ToolCallingMethod | None = 'auto'
	page_extraction_llm: BaseChatModel | None = None
	planner_llm: BaseChatModel | None = None
	planner_interval: int = 1  # Run planner every N steps
	is_planner_reasoning: bool = False  # type: ignore
	extend_planner_system_message: str | None = None

	# Playwright script generation setting
	save_playwright_script_path: str | None = None  # Path to save the generated Playwright script


class AgentState(BaseModel):
	"""Holds all state information for an Agent"""

	agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	n_steps: int = 1
	consecutive_failures: int = 0
	last_result: list[ActionResult] | None = None
	history: AgentHistoryList = Field(default_factory=lambda: AgentHistoryList(history=[]))
	last_plan: str | None = None
	paused: bool = False
	stopped: bool = False

	message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)

	# class Config:
	# 	arbitrary_types_allowed = True


@dataclass
class AgentStepInfo:
	step_number: int
	max_steps: int

	def is_last_step(self) -> bool:
		"""Check if this is the last step"""
		return self.step_number >= self.max_steps - 1


class ActionResult(BaseModel):
	"""Result of executing an action"""

	is_done: bool | None = False
	success: bool | None = None
	extracted_content: str | None = None
	error: str | None = None
	include_in_memory: bool = False  # whether to include in past messages as context or not


class StepMetadata(BaseModel):
	"""Metadata for a single step including timing and token information"""

	step_start_time: float
	step_end_time: float
	input_tokens: int  # Approximate tokens from message manager for this step
	step_number: int

	@property
	def duration_seconds(self) -> float:
		"""Calculate step duration in seconds"""
		return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
	"""Current state of the agent"""

	evaluation_previous_goal: str
	memory: str
	next_goal: str


class AgentOutput(BaseModel):
	"""Output model for agent

	@dev note: this model is extended with custom actions in AgentService. You can also use some fields that are not in this model as provided by the linter, as long as they are registered in the DynamicActions model.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	current_state: AgentBrain
	action: list[ActionModel] = Field(
		...,
		description='List of actions to execute',
		json_schema_extra={'min_items': 1},  # Ensure at least one action is provided
	)

	@staticmethod
	def type_with_custom_actions(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions"""
		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(
				list[custom_actions],
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutput.__module__,
		)
		model_.__doc__ = 'AgentOutput model with custom actions'
		return model_


class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: list[ActionResult]
	state: BrowserStateHistory
	metadata: StepMetadata | None = None

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

	@staticmethod
	def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
		elements = []
		for action in model_output.action:
			index = action.get_index()
			if index is not None and index in selector_map:
				el: DOMElementNode = selector_map[index]
				elements.append(HistoryTreeProcessor.convert_dom_element_to_history_element(el))
			else:
				elements.append(None)
		return elements

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Custom serialization handling circular references"""

		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = [action.model_dump(exclude_none=True) for action in self.model_output.action]
			model_output_dump = {
				'current_state': self.model_output.current_state.model_dump(),
				'action': action_dump,  # This preserves the actual action data
… (truncated)

```

---

## `browser-use-main\browser_use\browser\__init__.py`

```py
from .browser import Browser, BrowserConfig
from .context import BrowserContext, BrowserContextConfig
from .profile import BrowserProfile
from .session import BrowserSession

__all__ = ['Browser', 'BrowserConfig', 'BrowserContext', 'BrowserContextConfig', 'BrowserSession', 'BrowserProfile']

```

---

## `browser-use-main\browser_use\browser\browser.py`

```py
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

BrowserConfig = BrowserProfile
BrowserContextConfig = BrowserProfile
Browser = BrowserSession

__all__ = ['BrowserConfig', 'BrowserContextConfig', 'Browser']

```

---

## `browser-use-main\browser_use\browser\context.py`

```py
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession

Browser = BrowserSession
BrowserConfig = BrowserProfile
BrowserContext = BrowserSession
BrowserContextConfig = BrowserProfile

__all__ = ['Browser', 'BrowserConfig', 'BrowserContext', 'BrowserContextConfig']

```

---

## `browser-use-main\browser_use\browser\extensions.py`

```py
# import asyncio
# import hashlib
# import json
# import logging
# import subprocess
# import zipfile
# from pathlib import Path

# import aiohttp
# import anyio

# logger = logging.getLogger(__name__)


# def get_extension_id(unpacked_path: str | Path) -> str | None:
# 	manifest_path = Path(unpacked_path) / 'manifest.json'
# 	if not manifest_path.exists():
# 		return None

# 	# chrome uses a SHA256 hash of the unpacked extension directory path to compute a dynamic id for unpacked extensions
# 	hash_obj = hashlib.sha256()
# 	hash_obj.update(str(unpacked_path).encode('utf-8'))
# 	detected_extension_id = ''.join(chr(int(h, 16) + ord('a')) for h in hash_obj.hexdigest()[:32])
# 	return detected_extension_id


# async def install_extension(extension: dict) -> bool:
# 	manifest_path = Path(extension['unpacked_path']) / 'manifest.json'
# 	crx_path = Path(extension['crx_path'])

# 	# Download extensions using:
# 	# curl -fsSL 'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=1230&acceptformat=crx3&x=id%3D${EXTENSION_ID}%26uc' > extensionname.crx
# 	# unzip -d extensionname extensionname.crx

# 	if not manifest_path.exists() and not crx_path.exists():
# 		logger.info(f'[🛠️] Downloading missing extension {extension["name"]} {extension["webstore_id"]} -> {crx_path}')

# 		# Download crx file from ext.crx_url -> ext.crx_path
# 		async with aiohttp.ClientSession() as session:
# 			async with session.get(extension['crx_url']) as response:
# 				if response.headers.get('content-length') and response.content:
# 					async with anyio.open(crx_path, 'wb') as f:
# 						await f.write(await response.read())
# 				else:
# 					logger.warning(f'[⚠️] Failed to download extension {extension["name"]} {extension["webstore_id"]}')
# 					return False

# 	# Unzip crx file from ext.crx_url -> ext.unpacked_path
# 	unpacked_path = Path(extension['unpacked_path'])
# 	unpacked_path.mkdir(parents=True, exist_ok=True)

# 	try:
# 		# Try system unzip first
# 		result = subprocess.run(['/usr/bin/unzip', str(crx_path), '-d', str(unpacked_path)], capture_output=True, text=True)
# 		stdout, stderr = result.stdout, result.stderr
# 	except Exception as err1:
# 		try:
# 			# Fallback to Python's zipfile
# 			with zipfile.ZipFile(crx_path) as zf:
# 				zf.extractall(unpacked_path)
# 			stdout, stderr = '', ''
# 		except Exception as err2:
# 			logger.error(f'[❌] Failed to install {crx_path}: could not unzip crx', exc_info=(err1, err2))
# 			return False

# 	if not manifest_path.exists():
# 		logger.error(f'[❌] Failed to install {crx_path}: could not find manifest.json in unpacked_path', stdout, stderr)
# 		return False

# 	return True


# async def load_or_install_extension(ext: dict) -> dict:
# 	if not (ext.get('webstore_id') or ext.get('unpacked_path')):
# 		raise ValueError('Extension must have either webstore_id or unpacked_path')

# 	# Set statically computable extension metadata
# 	ext['webstore_id'] = ext.get('webstore_id') or ext.get('id')
# 	ext['name'] = ext.get('name') or ext['webstore_id']
# 	ext['webstore_url'] = ext.get('webstore_url') or f'https://chromewebstore.google.com/detail/{ext["webstore_id"]}'
# 	ext['crx_url'] = (
# 		ext.get('crx_url')
# 		or f'https://clients2.google.com/service/update2/crx?response=redirect&prodversion=1230&acceptformat=crx3&x=id%3D{ext["webstore_id"]}%26uc'
# 	)
# 	ext['crx_path'] = ext.get('crx_path') or str(Path(CHROME_EXTENSIONS_DIR) / f'{ext["webstore_id"]}__{ext["name"]}.crx')
# 	ext['unpacked_path'] = ext.get('unpacked_path') or str(Path(CHROME_EXTENSIONS_DIR) / f'{ext["webstore_id"]}__{ext["name"]}')

# 	manifest_path = Path(ext['unpacked_path']) / 'manifest.json'

# 	def read_manifest():
# 		with open(manifest_path) as f:
# 			return json.load(f)

# 	def read_version():
# 		return manifest_path.exists() and read_manifest().get('version')

# 	ext['read_manifest'] = read_manifest
# 	ext['read_version'] = read_version

# 	# if extension is not installed, download and unpack it
# 	if not ext['read_version']():
# 		await install_extension(ext)

# 	# autodetect id from filesystem path (unpacked extensions dont have stable IDs)
# 	ext['id'] = get_extension_id(ext['unpacked_path'])
# 	ext['version'] = ext['read_version']()

# 	if not ext['version']:
# 		logger.warning(f'[❌] Unable to detect ID and version of installed extension {pretty_path(ext["unpacked_path"])}')
# 	else:
# 		logger.info(f'[➕] Installed extension {ext["name"]} ({ext["version"]})...'.ljust(82) + pretty_path(ext['unpacked_path']))

# 	return ext


# async def is_target_extension(target):
# 	target_type = None
# 	target_ctx = None
# 	target_url = None
# 	try:
# 		target_type = await target.type
# 		target_ctx = await target.worker() or await target.page() or None
# 		target_url = await target.url or (await target_ctx.url if target_ctx else None)
# 	except Exception as err:
# 		if 'No target with given id found' in str(err):
# 			# because this runs on initial browser startup, we sometimes race with closing the initial
# 			# new tab page. it will throw a harmless error if we try to check a target that's already closed,
# 			# ignore it and return null since that page is definitely not an extension's bg page anyway
# 			target_type = 'closed'
# 			target_ctx = None
# 			target_url = 'about:closed'
# 		else:
# 			raise err

# 	target_is_bg = target_type in ['service_worker', 'background_page']
# 	target_is_extension = target_url and target_url.startswith('chrome-extension://')
# 	extension_id = target_url.split('://')[1].split('/')[0] if target_is_extension else None
# 	manifest_version = '3' if target_type == 'service_worker' else '2'

# 	return {
# 		'target_type': target_type,
# 		'target_ctx': target_ctx,
# 		'target_url': target_url,
# 		'target_is_bg': target_is_bg,
# 		'target_is_extension': target_is_extension,
# 		'extension_id': extension_id,
# 		'manifest_version': manifest_version,
# 	}


# async def load_extension_from_target(extensions, target):
# 	extension_info = await is_target_extension(target)
# 	target_is_bg = extension_info['target_is_bg']
# 	target_is_extension = extension_info['target_is_extension']
# 	target_type = extension_info['target_type']
# 	target_ctx = extension_info['target_ctx']
# 	target_url = extension_info['target_url']
# 	extension_id = extension_info['extension_id']
# 	manifest_version = extension_info['manifest_version']

# 	if not (target_is_bg and extension_id and target_ctx):
# 		return None

# 	manifest = await target_ctx.evaluate('() => chrome.runtime.getManifest()')

# 	name = manifest.get('name')
# 	version = manifest.get('version')
# 	homepage_url = manifest.get('homepage_url')
# 	options_page = manifest.get('options_page')
# 	options_ui = manifest.get('options_ui', {})

# 	if not version or not extension_id:
# 		return None

# 	options_url = await target_ctx.evaluate(
# 		'(options_page) => chrome.runtime.getURL(options_page)',
# 		options_page or options_ui.get('page') or 'options.html',
# 	)

# 	commands = await target_ctx.evaluate("""
#         async () => {
#             return await new Promise((resolve, reject) => {
#                 if (chrome.commands)
#                     chrome.commands.getAll(resolve)
#                 else
#                     resolve({})
#             })
#         }
#     """)

# 	# logger.debug(f"[+] Found Manifest V{manifest_version} Extension: {extension_id} {name} {target_url} {len(commands)}")

# 	async def dispatch_eval(*args):
# 		return await target_ctx.evaluate(*args)

# 	async def dispatch_popup():
# 		return await target_ctx.evaluate(
# 			"() => chrome.action?.openPopup() || chrome.tabs.create({url: chrome.runtime.getURL('popup.html')})"
# 		)

… (truncated)

```

---

## `browser-use-main\browser_use\browser\profile.py`

```py
import os
import sys
from collections.abc import Iterable
from enum import Enum
from functools import cache
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse
from venv import logger

from playwright._impl._api_structures import (
	ClientCertificate,
	Geolocation,
	HttpCredentials,
	ProxySettings,
	StorageState,
	ViewportSize,
)
from pydantic import AfterValidator, AliasChoices, BaseModel, ConfigDict, Field, model_validator

# fix pydantic error on python 3.11
# PydanticUserError: Please use `typing_extensions.TypedDict` instead of `typing.TypedDict` on Python < 3.12.
# For further information visit https://errors.pydantic.dev/2.10/u/typed-dict-version
if sys.version_info < (3, 12):
	from typing_extensions import TypedDict

	# convert new-style typing.TypedDict used by playwright to old-style typing_extensions.TypedDict used by pydantic
	ClientCertificate = TypedDict('ClientCertificate', ClientCertificate.__annotations__, total=ClientCertificate.__total__)
	Geolocation = TypedDict('Geolocation', Geolocation.__annotations__, total=Geolocation.__total__)
	ProxySettings = TypedDict('ProxySettings', ProxySettings.__annotations__, total=ProxySettings.__total__)
	ViewportSize = TypedDict('ViewportSize', ViewportSize.__annotations__, total=ViewportSize.__total__)
	HttpCredentials = TypedDict('HttpCredentials', HttpCredentials.__annotations__, total=HttpCredentials.__total__)
	StorageState = TypedDict('StorageState', StorageState.__annotations__, total=StorageState.__total__)

IN_DOCKER = os.environ.get('IN_DOCKER', 'false').lower()[0] in 'ty1'
CHROME_DEBUG_PORT = 9242  # use a non-default port to avoid conflicts with other tools / devs using 9222
CHROME_DISABLED_COMPONENTS = [
	# Playwright defaults: https://github.com/microsoft/playwright/blob/41008eeddd020e2dee1c540f7c0cdfa337e99637/packages/playwright-core/src/server/chromium/chromiumSwitches.ts#L76
	# See https:#github.com/microsoft/playwright/pull/10380
	'AcceptCHFrame',
	# See https:#github.com/microsoft/playwright/pull/10679
	'AutoExpandDetailsElement',
	# See https:#github.com/microsoft/playwright/issues/14047
	'AvoidUnnecessaryBeforeUnloadCheckSync',
	# See https:#github.com/microsoft/playwright/pull/12992
	'CertificateTransparencyComponentUpdater',
	'DestroyProfileOnBrowserClose',
	# See https:#github.com/microsoft/playwright/pull/13854
	'DialMediaRouteProvider',
	# Chromium is disabling manifest version 2. Allow testing it as long as Chromium can actually run it.
	# Disabled in https:#chromium-review.googlesource.com/c/chromium/src/+/6265903.
	'ExtensionManifestV2Disabled',
	'GlobalMediaControls',
	# See https:#github.com/microsoft/playwright/pull/27605
	'HttpsUpgrades',
	'ImprovedCookieControls',
	'LazyFrameLoading',
	# Hides the Lens feature in the URL address bar. Its not working in unofficial builds.
	'LensOverlay',
	# See https:#github.com/microsoft/playwright/pull/8162
	'MediaRouter',
	# See https:#github.com/microsoft/playwright/issues/28023
	'PaintHolding',
	# See https:#github.com/microsoft/playwright/issues/32230
	'ThirdPartyStoragePartitioning',
	# See https://github.com/microsoft/playwright/issues/16126
	'Translate',
	'AutomationControlled',
	# Added by us:
	'OptimizationHints',
	'ProcessPerSiteUpToMainFrameThreshold',
	'InterestFeedContentSuggestions',
	'CalculateNativeWinOcclusion',  # chrome normally stops rendering tabs if they are not visible (occluded by a foreground window or other app)
	# 'BackForwardCache',  # agent does actually use back/forward navigation, but we can disable if we ever remove that
	'HeavyAdPrivacyMitigations',
	'PrivacySandboxSettings4',
	'AutofillServerCommunication',
	'CrashReporting',
	'OverscrollHistoryNavigation',
	'InfiniteSessionRestore',
	'ExtensionDisableUnsupportedDeveloper',
]

CHROME_HEADLESS_ARGS = [
	'--headless=new',
]

CHROME_DOCKER_ARGS = [
	'--no-sandbox',
	'--disable-gpu-sandbox',
	'--disable-setuid-sandbox',
	'--disable-dev-shm-usage',
	'--no-xshm',
	'--no-zygote',
	'--single-process',
]

CHROME_DISABLE_SECURITY_ARGS = [
	'--disable-web-security',
	'--disable-site-isolation-trials',
	'--disable-features=IsolateOrigins,site-per-process',
	'--allow-running-insecure-content',
	'--ignore-certificate-errors',
	'--ignore-ssl-errors',
	'--ignore-certificate-errors-spki-list',
]

CHROME_DETERMINISTIC_RENDERING_ARGS = [
	'--deterministic-mode',
	'--js-flags=--random-seed=1157259159',
	'--force-device-scale-factor=2',
	'--enable-webgl',
	# '--disable-skia-runtime-opts',
	# '--disable-2d-canvas-clip-aa',
	'--font-render-hinting=none',
	'--force-color-profile=srgb',
]

CHROME_DEFAULT_ARGS = [
	# provided by playwright by default: https://github.com/microsoft/playwright/blob/41008eeddd020e2dee1c540f7c0cdfa337e99637/packages/playwright-core/src/server/chromium/chromiumSwitches.ts#L76
	# we don't need to include them twice in our own config, but it's harmless
	'--disable-field-trial-config',  # https://source.chromium.org/chromium/chromium/src/+/main:testing/variations/README.md
	'--disable-background-networking',
	'--disable-background-timer-throttling',
	'--disable-backgrounding-occluded-windows',
	'--disable-back-forward-cache',  # Avoids surprises like main request not being intercepted during page.goBack().
	'--disable-breakpad',
	'--disable-client-side-phishing-detection',
	'--disable-component-extensions-with-background-pages',
	'--disable-component-update',  # Avoids unneeded network activity after startup.
	'--no-default-browser-check',
	# '--disable-default-apps',
	'--disable-dev-shm-usage',
	# '--disable-extensions',
	# '--disable-features=' + disabledFeatures(assistantMode).join(','),
	'--allow-pre-commit-input',  # let page JS run a little early before GPU rendering finishes
	'--disable-hang-monitor',
	'--disable-ipc-flooding-protection',
	'--disable-popup-blocking',
	'--disable-prompt-on-repost',
	'--disable-renderer-backgrounding',
	# '--force-color-profile=srgb',  # moved to CHROME_DETERMINISTIC_RENDERING_ARGS
	'--metrics-recording-only',
	'--no-first-run',
	'--password-store=basic',
	'--use-mock-keychain',
	# // See https://chromium-review.googlesource.com/c/chromium/src/+/2436773
	'--no-service-autorun',
	'--export-tagged-pdf',
	# // https://chromium-review.googlesource.com/c/chromium/src/+/4853540
	'--disable-search-engine-choice-screen',
	# // https://issues.chromium.org/41491762
	'--unsafely-disable-devtools-self-xss-warnings',
	'--enable-features=NetworkService,NetworkServiceInProcess',
	'--enable-network-information-downlink-max',
	# added by us:
	'--test-type=gpu',
	'--disable-sync',
	'--allow-legacy-extension-manifests',
	'--allow-pre-commit-input',
	'--disable-blink-features=AutomationControlled',
	'--install-autogenerated-theme=0,0,0',
	'--hide-scrollbars',
	'--log-level=2',
	# '--enable-logging=stderr',
	'--disable-focus-on-load',
	'--disable-window-activation',
	'--generate-pdf-document-outline',
	'--no-pings',
	'--ash-no-nudges',
	'--disable-infobars',
	'--simulate-outdated-no-au="Tue, 31 Dec 2099 23:59:59 GMT"',
	'--hide-crash-restore-bubble',
	'--suppress-message-center-popups',
	'--disable-domain-reliability',
	'--disable-datasaver-prompt',
	'--disable-speech-synthesis-api',
	'--disable-speech-api',
	'--disable-print-preview',
	'--safebrowsing-disable-auto-update',
	'--disable-external-intent-requests',
	'--disable-desktop-notifications',
	'--noerrdialogs',
	'--silent-debugger-extension-api',
	f'--disable-features={",".join(CHROME_DISABLED_COMPONENTS)}',
]


@cache
def get_display_size() -> ViewportSize | None:
	# macOS
	try:
		from AppKit import NSScreen

		screen = NSScreen.mainScreen().frame()
		return ViewportSize(width=int(screen.size.width), height=int(screen.size.height))
	except Exception:
		pass

… (truncated)

```

---

## `browser-use-main\browser_use\browser\session.py`

```py
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Self

import psutil
from patchright.async_api import Playwright as PatchrightPlaywright
from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import BrowserContext as PlaywrightBrowserContext
from playwright.async_api import ElementHandle, FrameLocator, Page, Playwright, async_playwright
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, InstanceOf, PrivateAttr, model_validator

from browser_use.browser.profile import BrowserProfile
from browser_use.browser.views import (
	BrowserError,
	BrowserStateSummary,
	TabInfo,
	URLNotAllowedError,
)
from browser_use.dom.clickable_element_processor.service import ClickableElementProcessor
from browser_use.dom.service import DomService
from browser_use.dom.views import DOMElementNode, SelectorMap
from browser_use.utils import time_execution_async, time_execution_sync

# Check if running in Docker
IN_DOCKER = os.environ.get('IN_DOCKER', 'false').lower()[0] in 'ty1'

logger = logging.getLogger('browser_use.browser.session')


_GLOB_WARNING_SHOWN = False  # used inside _is_url_allowed to avoid spamming the logs with the same warning multiple times


def truncate_url(s: str, max_len: int | None = None) -> str:
	"""Truncate/pretty-print a URL with a maximum length, removing the protocol and www. prefix"""
	s = s.replace('https://', '').replace('http://', '').replace('www.', '')
	if max_len is not None and len(s) > max_len:
		return s[:max_len] + '…'
	return s


def require_initialization(func):
	"""decorator for BrowserSession methods to require the BrowserSession be already active"""

	@wraps(func)
	def wrapper(self, *args, **kwargs):
		if not self.initialized:
			raise RuntimeError('BrowserSession(...).start() must be called first to launch or connect to the browser')
		if not self.agent_current_page or self.agent_current_page.is_closed():
			self.agent_current_page = self.browser_context.pages[0] if self.browser_context.pages else None

		if not self.agent_current_page or self.agent_current_page.is_closed():
			self.create_new_tab()
			assert self.agent_current_page and not self.agent_current_page.is_closed()

		if not hasattr(self, '_cached_browser_state_summary'):
			raise RuntimeError('BrowserSession(...).start() must be called first to initialize the browser session')

		return func(self, *args, **kwargs)

	return wrapper


DEFAULT_BROWSER_PROFILE = BrowserProfile()


@dataclass
class CachedClickableElementHashes:
	"""
	Clickable elements hashes for the last state
	"""

	url: str
	hashes: set[str]


class BrowserSession(BaseModel):
	"""
	Represents an active browser session with a running browser process somewhere.

	Chromium flags should be passed via extra_launch_args.
	Extra Playwright launch options (e.g., handle_sigterm, handle_sigint) can be passed as kwargs to BrowserSession and will be forwarded to the launch() call.
	"""

	model_config = ConfigDict(
		extra='allow',
		validate_assignment=False,
		revalidate_instances='always',
		frozen=False,
		arbitrary_types_allowed=True,
		populate_by_name=True,
	)
	# this class accepts arbitrary extra **kwargs in init because of the extra='allow' pydantic option
	# they are saved on the model, then applied to self.browser_profile via .apply_session_overrides_to_profile()

	# template profile for the BrowserSession, will be copied at init/validation time, and overrides applied to the copy
	browser_profile: InstanceOf[BrowserProfile] = Field(
		default=DEFAULT_BROWSER_PROFILE,
		description='BrowserProfile() instance containing config for the BrowserSession',
		validation_alias=AliasChoices('profile', 'config', 'new_context_config'),  # old names for this field, remove eventually
	)

	# runtime props/state: these can be passed in as props at init, or get auto-setup by BrowserSession.start()
	wss_url: str | None = Field(
		default=None,
		description='WSS URL of the node.js playwright browser server to connect to, outputted by (await chromium.launchServer()).wsEndpoint()',
	)
	cdp_url: str | None = Field(
		default=None,
		description='CDP URL of the browser to connect to, e.g. http://localhost:9222 or ws://127.0.0.1:9222/devtools/browser/387adf4c-243f-4051-a181-46798f4a46f4',
	)
	chrome_pid: int | None = Field(
		default=None, description='pid of the running chrome process to connect to on localhost (optional)'
	)
	playwright: Playwright | PatchrightPlaywright | Playwright | None = Field(
		default=None,
		description='Playwright library object returned by: await (playwright or patchright).async_playwright().start()',
		exclude=True,
	)
	browser: InstanceOf[PlaywrightBrowser] | None = Field(
		default=None,
		description='playwright Browser object to use (optional)',
		validation_alias=AliasChoices('playwright_browser'),
		exclude=True,
	)
	browser_context: InstanceOf[PlaywrightBrowserContext] | None = Field(
		default=None,
		description='playwright BrowserContext object to use (optional)',
		validation_alias=AliasChoices('playwright_browser_context', 'context'),
		exclude=True,
	)
	initialized: bool = Field(
		default=False,
		description='Skip BrowserSession launch/connection setup entirely if True (not recommended)',
		validation_alias=AliasChoices('initialized', 'is_initialized'),
	)

	# runtime state: internally tracked attrs updated by BrowserSession class methods
	agent_current_page: InstanceOf[Page] | None = Field(  # mutated by self.create_new_tab(url)
		default=None,
		description='Foreground Page that the agent is focused on',
		validation_alias=AliasChoices('current_page', 'page'),
		exclude=True,
	)
	human_current_page: InstanceOf[Page] | None = Field(  # mutated by self.setup_foreground_tab_detection()
		default=None,
		description='Foreground Page that the human is focused on',
		exclude=True,
	)

	_cached_browser_state_summary: BrowserStateSummary | None = PrivateAttr(default=None)
	_cached_clickable_element_hashes: CachedClickableElementHashes | None = PrivateAttr(default=None)

	@model_validator(mode='after')
	def apply_session_overrides_to_profile(self) -> Self:
		"""Apply any extra **kwargs passed to BrowserSession(...) as config overrides on top of browser_profile"""
		session_own_fields = type(self).model_fields.keys()

		# get all the extra BrowserProfile kwarg overrides passed to BrowserSession(...) that are not Fields on self
		overrides = self.model_dump(exclude=session_own_fields)

		# FOR REPL DEBUGGING ONLY, NEVER ALLOW CIRCULAR REFERENCES IN REAL CODE:
		# self.browser_profile._in_use_by_session = self

		# replace browser_profile with patched version
		self.browser_profile = self.browser_profile.model_copy(update=overrides)

		# FOR REPL DEBUGGING ONLY, NEVER ALLOW CIRCULAR REFERENCES IN REAL CODE:
		# self.browser_profile._in_use_by_session = self

		return self

	# def __getattr__(self, key: str) -> Any:
	# 	"""
	# 	fall back to getting any attrs from the underlying self.browser_profile when not defined on self.
	# 	(extra kwargs passed e.g. BrowserSession(extra_kwarg=124) on init get saved into self.browser_profile on validation,
	# 	so this also allows you to read those: browser_session.extra_kwarg => browser_session.browser_profile.extra_kwarg)
	# 	"""
	# 	return getattr(self.browser_profile, key)

	async def start(self) -> Self:
		# finish initializing/validate the browser_profile:
		assert isinstance(self.browser_profile, BrowserProfile)
		self.browser_profile.prepare_user_data_dir()  # create/unlock the <user_data_dir>/SingletonLock
		self.browser_profile.detect_display_configuration()  # adjusts config values, must come before launch/connect

		# launch/connect to the browser:
		# setup playwright library client, Browser, and BrowserContext objects
		await self.setup_playwright()
		await self.setup_browser_connection()  # connects to existing br
… (truncated)

```

---

## `browser-use-main\browser_use\browser\views.py`

```py
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from browser_use.dom.history_tree_processor.service import DOMHistoryElement
from browser_use.dom.views import DOMState


# Pydantic
class TabInfo(BaseModel):
	"""Represents information about a browser tab"""

	page_id: int
	url: str
	title: str
	parent_page_id: int | None = None  # parent page that contains this popup or cross-origin iframe


@dataclass
class BrowserStateSummary(DOMState):
	"""The summary of the browser's current state designed for an LLM to process"""

	# provided by DOMState:
	# element_tree: DOMElementNode
	# selector_map: SelectorMap

	url: str
	title: str
	tabs: list[TabInfo]
	screenshot: str | None = None
	pixels_above: int = 0
	pixels_below: int = 0
	browser_errors: list[str] = field(default_factory=list)


@dataclass
class BrowserStateHistory:
	"""The summary of the browser's state at a past point in time to usse in LLM message history"""

	url: str
	title: str
	tabs: list[TabInfo]
	interacted_element: list[DOMHistoryElement | None] | list[None]
	screenshot: str | None = None

	def to_dict(self) -> dict[str, Any]:
		data = {}
		data['tabs'] = [tab.model_dump() for tab in self.tabs]
		data['screenshot'] = self.screenshot
		data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['url'] = self.url
		data['title'] = self.title
		return data


class BrowserError(Exception):
	"""Base class for all browser errors"""


class URLNotAllowedError(BrowserError):
	"""Error raised when a URL is not allowed"""

```

---

## `browser-use-main\browser_use\cli.py`

```py
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
	import click
	from textual import events
	from textual.app import App, ComposeResult
	from textual.binding import Binding
	from textual.containers import Container, HorizontalGroup, VerticalScroll
	from textual.widgets import Footer, Header, Input, Label, Link, RichLog, Static
except ImportError:
	print('⚠️ CLI addon is not installed. Please install it with: `pip install browser-use[cli]` and try again.')
	sys.exit(1)

import langchain_anthropic
import langchain_google_genai
import langchain_openai

try:
	import readline

	READLINE_AVAILABLE = True
except ImportError:
	# readline not available on Windows by default
	READLINE_AVAILABLE = False

from browser_use import Agent, Controller
from browser_use.agent.views import AgentSettings
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.logging_config import addLoggingLevel

# Paths
USER_CONFIG_DIR = Path.home() / '.config' / 'browseruse'
USER_CONFIG_FILE = USER_CONFIG_DIR / 'config.json'
CHROME_PROFILES_DIR = USER_CONFIG_DIR / 'profiles'
USER_DATA_DIR = CHROME_PROFILES_DIR / 'default'

# Default User settings
MAX_HISTORY_LENGTH = 100

# Ensure directories exist
USER_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Logo components with styling for rich panels
BROWSER_LOGO = """
				   [white]   ++++++   +++++++++   [/]                                
				   [white] +++     +++++     +++  [/]                                
				   [white] ++    ++++   ++    ++  [/]                                
				   [white] ++  +++       +++  ++  [/]                                
				   [white]   ++++          +++    [/]                                
				   [white]  +++             +++   [/]                                
				   [white] +++               +++  [/]                                
				   [white] ++   +++      +++  ++  [/]                                
				   [white] ++    ++++   ++    ++  [/]                                
				   [white] +++     ++++++    +++  [/]                                
				   [white]   ++++++    +++++++    [/]                                

[white]██████╗ ██████╗  ██████╗ ██╗    ██╗███████╗███████╗██████╗[/]     [darkorange]██╗   ██╗███████╗███████╗[/]
[white]██╔══██╗██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔════╝██╔══██╗[/]    [darkorange]██║   ██║██╔════╝██╔════╝[/]
[white]██████╔╝██████╔╝██║   ██║██║ █╗ ██║███████╗█████╗  ██████╔╝[/]    [darkorange]██║   ██║███████╗█████╗[/]  
[white]██╔══██╗██╔══██╗██║   ██║██║███╗██║╚════██║██╔══╝  ██╔══██╗[/]    [darkorange]██║   ██║╚════██║██╔══╝[/]  
[white]██████╔╝██║  ██║╚██████╔╝╚███╔███╔╝███████║███████╗██║  ██║[/]    [darkorange]╚██████╔╝███████║███████╗[/]
[white]╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝ ╚══════╝╚══════╝╚═╝  ╚═╝[/]     [darkorange]╚═════╝ ╚══════╝╚══════╝[/]
"""


# Common UI constants
TEXTUAL_BORDER_STYLES = {'logo': 'blue', 'info': 'blue', 'input': 'orange3', 'working': 'yellow', 'completion': 'green'}


def get_default_config() -> dict[str, Any]:
	"""Return default configuration dictionary."""
	return {
		'model': {
			'name': None,
			'temperature': 0.0,
			'api_keys': {
				'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
				'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY', ''),
				'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY', ''),
				'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY', ''),
				'GROK_API_KEY': os.getenv('GROK_API_KEY', ''),
			},
		},
		'agent': {},  # AgentSettings will use defaults
		'browser': {
			'headless': True,
		},
		'browser_context': {
			'keep_alive': True,
			'ignore_https_errors': False,
		},
		'command_history': [],
	}


def load_user_config() -> dict[str, Any]:
	"""Load user configuration from file."""
	if not USER_CONFIG_FILE.exists():
		# Create default config
		config = get_default_config()
		save_user_config(config)
		return config

	try:
		with open(USER_CONFIG_FILE) as f:
			data = json.load(f)
			# Ensure data is a dictionary, not a list
			if isinstance(data, list):
				# If it's a list, it's probably just command history from previous version
				config = get_default_config()
				config['command_history'] = data  # Use the list as command history
				return config
			return data
	except (json.JSONDecodeError, FileNotFoundError):
		# If file is corrupted, start with empty config
		return get_default_config()


def save_user_config(config: dict[str, Any]) -> None:
	"""Save user configuration to file."""
	# Ensure command history doesn't exceed maximum length
	if 'command_history' in config and isinstance(config['command_history'], list):
		if len(config['command_history']) > MAX_HISTORY_LENGTH:
			config['command_history'] = config['command_history'][-MAX_HISTORY_LENGTH:]

	with open(USER_CONFIG_FILE, 'w') as f:
		json.dump(config, f, indent=2)


def update_config_with_click_args(config: dict[str, Any], ctx: click.Context) -> dict[str, Any]:
	"""Update configuration with command-line arguments."""
	# Ensure required sections exist
	if 'model' not in config:
		config['model'] = {}
	if 'browser' not in config:
		config['browser'] = {}
	if 'browser_context' not in config:
		config['browser_context'] = {}

	# Create a merged browser profile config for all browser settings
	browser_profile = config.get('browser', {}) | config.get('browser_context', {})

	# Update configuration with command-line args if provided
	if ctx.params.get('model'):
		config['model']['name'] = ctx.params['model']
	if ctx.params.get('headless') is not None:
		browser_profile['headless'] = ctx.params['headless']
	if ctx.params.get('window_width'):
		browser_profile['window_width'] = ctx.params['window_width']
	if ctx.params.get('window_height'):
		browser_profile['window_height'] = ctx.params['window_height']

	# Update config with the merged profile
	config['browser'] = browser_profile
	# Remove the old split config
	if 'browser_context' in config:
		del config['browser_context']

	return config


def setup_readline_history(history: list[str]) -> None:
	"""Set up readline with command history."""
	if not READLINE_AVAILABLE:
		return

	# Add history items to readline
	for item in history:
		readline.add_history(item)


def get_llm(config: dict[str, Any]):
	"""Get the language model based on config and available API keys."""
	# Set API keys from config if available
	api_keys = config.get('model', {}).get('api_keys', {})
	model_name = config.get('model', {}).get('name')
	temperature = config.get('model', {}).get('temperature', 0.0)

	# Set environment variables if they're in the config but not in the environment
	if api_keys.get('openai') and not os.getenv('OPENAI_API_KEY'):
		os.environ['OPENAI_API_KEY'] = api_keys['openai']
	if api_keys.get('anthropic') and not os.getenv('ANTHROPIC_API_KEY'):
		os.environ['ANTHROPIC_API_KEY'] = api_keys['anthropic']
	if api_keys.get('google') and not os.getenv('GOOGLE_API_KEY'):
		os.environ['GOOGLE_API_KEY'] = api_keys['google']

	if model_name:
… (truncated)

```

---

## `browser-use-main\browser_use\controller\registry\service.py`

```py
import asyncio
import logging
from collections.abc import Callable
from inspect import iscoroutinefunction, signature
from typing import Any, Generic, Optional, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, create_model

from browser_use.browser import BrowserSession
from browser_use.controller.registry.views import (
	ActionModel,
	ActionRegistry,
	RegisteredAction,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
	ControllerRegisteredFunctionsTelemetryEvent,
	RegisteredFunction,
)
from browser_use.utils import time_execution_async

Context = TypeVar('Context')

logger = logging.getLogger(__name__)


class Registry(Generic[Context]):
	"""Service for registering and managing actions"""

	def __init__(self, exclude_actions: list[str] | None = None):
		self.registry = ActionRegistry()
		self.telemetry = ProductTelemetry()
		self.exclude_actions = exclude_actions if exclude_actions is not None else []

	# @time_execution_sync('--create_param_model')
	def _create_param_model(self, function: Callable) -> type[BaseModel]:
		"""Creates a Pydantic model from function signature"""
		sig = signature(function)
		params = {
			name: (param.annotation, ... if param.default == param.empty else param.default)
			for name, param in sig.parameters.items()
			if name != 'browser'
			and name != 'page_extraction_llm'
			and name != 'available_file_paths'
			and name != 'browser_session'
			and name != 'browser_context'
		}
		# TODO: make the types here work
		return create_model(
			f'{function.__name__}_parameters',
			__base__=ActionModel,
			**params,  # type: ignore
		)

	def action(
		self,
		description: str,
		param_model: type[BaseModel] | None = None,
		domains: list[str] | None = None,
		page_filter: Callable[[Any], bool] | None = None,
	):
		"""Decorator for registering actions"""

		def decorator(func: Callable):
			# Skip registration if action is in exclude_actions
			if func.__name__ in self.exclude_actions:
				return func

			# Create param model from function if not provided
			actual_param_model = param_model or self._create_param_model(func)

			# Wrap sync functions to make them async
			if not iscoroutinefunction(func):

				async def async_wrapper(*args, **kwargs):
					return await asyncio.to_thread(func, *args, **kwargs)

				# Copy the signature and other metadata from the original function
				async_wrapper.__signature__ = signature(func)
				async_wrapper.__name__ = func.__name__
				async_wrapper.__annotations__ = func.__annotations__
				wrapped_func = async_wrapper
			else:
				wrapped_func = func

			action = RegisteredAction(
				name=func.__name__,
				description=description,
				function=wrapped_func,
				param_model=actual_param_model,
				domains=domains,
				page_filter=page_filter,
			)
			self.registry.actions[func.__name__] = action
			return func

		return decorator

	@time_execution_async('--execute_action')
	async def execute_action(
		self,
		action_name: str,
		params: dict,
		browser_session: BrowserSession | None = None,
		page_extraction_llm: BaseChatModel | None = None,
		sensitive_data: dict[str, str] | None = None,
		available_file_paths: list[str] | None = None,
		#
		context: Context | None = None,
	) -> Any:
		"""Execute a registered action"""
		if action_name not in self.registry.actions:
			raise ValueError(f'Action {action_name} not found')

		action = self.registry.actions[action_name]
		try:
			# Create the validated Pydantic model
			try:
				validated_params = action.param_model(**params)
			except Exception as e:
				raise ValueError(f'Invalid parameters {params} for action {action_name}: {type(e)}: {e}') from e

			# Check if the first parameter is a Pydantic model
			sig = signature(action.function)
			parameters = list(sig.parameters.values())
			is_pydantic = parameters and issubclass(parameters[0].annotation, BaseModel)
			parameter_names = [param.name for param in parameters]

			if sensitive_data:
				validated_params = self._replace_sensitive_data(validated_params, sensitive_data)

			# Check if the action requires browser
			if (
				'browser_session' in parameter_names or 'browser' in parameter_names or 'browser_context' in parameter_names
			) and not browser_session:
				raise ValueError(f'Action {action_name} requires browser_session but none provided.')
			if 'page_extraction_llm' in parameter_names and not page_extraction_llm:
				raise ValueError(f'Action {action_name} requires page_extraction_llm but none provided.')
			if 'available_file_paths' in parameter_names and not available_file_paths:
				raise ValueError(f'Action {action_name} requires available_file_paths but none provided.')

			if 'context' in parameter_names and not context:
				raise ValueError(f'Action {action_name} requires context but none provided.')

			# Prepare arguments based on parameter type
			extra_args = {}
			if 'context' in parameter_names:
				extra_args['context'] = context
			if 'browser_session' in parameter_names:
				extra_args['browser_session'] = browser_session
			if 'browser' in parameter_names:  # support legacy browser: BrowserContext arg
				logger.debug(
					f'You should update this action {action_name}(browser: BrowserContext)  -> to take {action_name}(browser_session: BrowserSession) instead'
				)
				extra_args['browser'] = browser_session
			if 'browser_context' in parameter_names:  # support legacy browser: BrowserContext arg
				logger.debug(
					f'You should update this action {action_name}(browser_context: BrowserContext)  -> to take {action_name}(browser_session: BrowserSession) instead'
				)
				extra_args['browser_context'] = browser_session
			if 'page_extraction_llm' in parameter_names:
				extra_args['page_extraction_llm'] = page_extraction_llm
			if 'available_file_paths' in parameter_names:
				extra_args['available_file_paths'] = available_file_paths
			if action_name == 'input_text' and sensitive_data:
				extra_args['has_sensitive_data'] = True
			if is_pydantic:
				return await action.function(validated_params, **extra_args)
			return await action.function(**validated_params.model_dump(), **extra_args)

		except Exception as e:
			raise RuntimeError(f'Error executing action {action_name}: {str(e)}') from e

	def _replace_sensitive_data(self, params: BaseModel, sensitive_data: dict[str, str]) -> BaseModel:
		"""Replaces the sensitive data in the params"""
		# if there are any str with <secret>placeholder</secret> in the params, replace them with the actual value from sensitive_data

		import logging
		import re

		logger = logging.getLogger(__name__)
		secret_pattern = re.compile(r'<secret>(.*?)</secret>')

		# Set to track all missing placeholders across the full object
		all_missing_placeholders = set()

		def replace_secrets(value):
			if isinstance(value, str):
				matches = secret_pattern.findall(value)

				for placeholder in matches:
					if placeholder in sensitive_data and sensitive_data[placeholder]:
						value = value.replace(f'<secret>{placeholder}</secret>', sensitive_data[placeholder])
					else:
						# Keep track of missing placeholders
						all_missing_placeholders.add(placeholder)
						# Don't replace the tag, keep it as is

				return value
… (truncated)

```

---

## `browser-use-main\browser_use\controller\registry\views.py`

```py
from collections.abc import Callable

from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict


class RegisteredAction(BaseModel):
	"""Model for a registered action"""

	name: str
	description: str
	function: Callable
	param_model: type[BaseModel]

	# filters: provide specific domains or a function to determine whether the action should be available on the given page or not
	domains: list[str] | None = None  # e.g. ['*.google.com', 'www.bing.com', 'yahoo.*]
	page_filter: Callable[[Page], bool] | None = None

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def prompt_description(self) -> str:
		"""Get a description of the action for the prompt"""
		skip_keys = ['title']
		s = f'{self.description}: \n'
		s += '{' + str(self.name) + ': '
		s += str(
			{
				k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k not in skip_keys}
				for k, v in self.param_model.model_json_schema()['properties'].items()
			}
		)
		s += '}'
		return s


class ActionModel(BaseModel):
	"""Base model for dynamically created action models"""

	# this will have all the registered actions, e.g.
	# click_element = param_model = ClickElementParams
	# done = param_model = None
	#
	model_config = ConfigDict(arbitrary_types_allowed=True)

	def get_index(self) -> int | None:
		"""Get the index of the action"""
		# {'clicked_element': {'index':5}}
		params = self.model_dump(exclude_unset=True).values()
		if not params:
			return None
		for param in params:
			if param is not None and 'index' in param:
				return param['index']
		return None

	def set_index(self, index: int):
		"""Overwrite the index of the action"""
		# Get the action name and params
		action_data = self.model_dump(exclude_unset=True)
		action_name = next(iter(action_data.keys()))
		action_params = getattr(self, action_name)

		# Update the index directly on the model
		if hasattr(action_params, 'index'):
			action_params.index = index


class ActionRegistry(BaseModel):
	"""Model representing the action registry"""

	actions: dict[str, RegisteredAction] = {}

	@staticmethod
	def _match_domains(domains: list[str] | None, url: str) -> bool:
		"""
		Match a list of domain glob patterns against a URL.

		Args:
			domain_patterns: A list of domain patterns that can include glob patterns (* wildcard)
			url: The URL to match against

		Returns:
			True if the URL's domain matches the pattern, False otherwise
		"""

		if domains is None or not url:
			return True

		import fnmatch
		from urllib.parse import urlparse

		# Parse the URL to get the domain
		try:
			parsed_url = urlparse(url)
			if not parsed_url.netloc:
				return False

			domain = parsed_url.netloc
			# Remove port if present
			if ':' in domain:
				domain = domain.split(':')[0]

			for domain_pattern in domains:
				if fnmatch.fnmatch(domain, domain_pattern):  # Perform glob *.matching.*
					return True
			return False
		except Exception:
			return False

	@staticmethod
	def _match_page_filter(page_filter: Callable[[Page], bool] | None, page: Page) -> bool:
		"""Match a page filter against a page"""
		if page_filter is None:
			return True
		return page_filter(page)

	def get_prompt_description(self, page: Page | None = None) -> str:
		"""Get a description of all actions for the prompt

		Args:
			page: If provided, filter actions by page using page_filter and domains.

		Returns:
			A string description of available actions.
			- If page is None: return only actions with no page_filter and no domains (for system prompt)
			- If page is provided: return only filtered actions that match the current page (excluding unfiltered actions)
		"""
		if page is None:
			# For system prompt (no page provided), include only actions with no filters
			return '\n'.join(
				action.prompt_description()
				for action in self.actions.values()
				if action.page_filter is None and action.domains is None
			)

		# only include filtered actions for the current page
		filtered_actions = []
		for action in self.actions.values():
			if not (action.domains or action.page_filter):
				# skip actions with no filters, they are already included in the system prompt
				continue

			domain_is_allowed = self._match_domains(action.domains, page.url)
			page_is_allowed = self._match_page_filter(action.page_filter, page)

			if domain_is_allowed and page_is_allowed:
				filtered_actions.append(action)

		return '\n'.join(action.prompt_description() for action in filtered_actions)

```

---

## `browser-use-main\browser_use\controller\service.py`

```py
import asyncio
import enum
import json
import logging
import re
from typing import Generic, TypeVar, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from playwright.async_api import ElementHandle, Page

# from lmnr.sdk.laminar import Laminar
from pydantic import BaseModel

from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser import BrowserSession
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
	ClickElementAction,
	CloseTabAction,
	DoneAction,
	DragDropAction,
	GoToUrlAction,
	InputTextAction,
	NoParamsAction,
	OpenTabAction,
	Position,
	ScrollAction,
	SearchGoogleAction,
	SendKeysAction,
	SwitchTabAction,
)
from browser_use.utils import time_execution_sync

logger = logging.getLogger(__name__)


Context = TypeVar('Context')


class Controller(Generic[Context]):
	def __init__(
		self,
		exclude_actions: list[str] = [],
		output_model: type[BaseModel] | None = None,
	):
		self.registry = Registry[Context](exclude_actions)

		"""Register all default browser actions"""

		if output_model is not None:
			# Create a new model that extends the output model with success parameter
			class ExtendedOutputModel(BaseModel):  # type: ignore
				success: bool = True
				data: output_model  # type: ignore

			@self.registry.action(
				'Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached',
				param_model=ExtendedOutputModel,
			)
			async def done(params: ExtendedOutputModel):
				# Exclude success from the output JSON since it's an internal parameter
				output_dict = params.data.model_dump()

				# Enums are not serializable, convert to string
				for key, value in output_dict.items():
					if isinstance(value, enum.Enum):
						output_dict[key] = value.value

				return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
		else:

			@self.registry.action(
				'Complete task - with return text and if the task is finished (success=True) or not yet  completely finished (success=False), because last step is reached',
				param_model=DoneAction,
			)
			async def done(params: DoneAction):
				return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

		# Basic Navigation Actions
		@self.registry.action(
			'Search the query in Google in the current tab, the query should be a search query like humans search in Google, concrete and not vague or super long. More the single most important items. ',
			param_model=SearchGoogleAction,
		)
		async def search_google(params: SearchGoogleAction, browser_session: BrowserSession):
			search_url = f'https://www.google.com/search?q={params.query}&udm=14'

			page = await browser_session.get_current_page()
			if page:
				await page.goto(search_url)
				await page.wait_for_load_state()
			else:
				page = await browser_session.create_new_tab(search_url)
			msg = f'🔍  Searched for "{params.query}" in Google'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action('Navigate to URL in the current tab', param_model=GoToUrlAction)
		async def go_to_url(params: GoToUrlAction, browser_session: BrowserSession):
			page = await browser_session.get_current_page()
			if page:
				await page.goto(params.url)
				await page.wait_for_load_state()
			else:
				page = await browser_session.create_new_tab(params.url)
			msg = f'🔗  Navigated to {params.url}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action('Go back', param_model=NoParamsAction)
		async def go_back(_: NoParamsAction, browser_session: BrowserSession):
			await browser_session.go_back()
			msg = '🔙  Navigated back'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# wait for x seconds
		@self.registry.action('Wait for x seconds default 3')
		async def wait(seconds: int = 3):
			msg = f'🕒  Waiting for {seconds} seconds'
			logger.info(msg)
			await asyncio.sleep(seconds)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# Element Interaction Actions
		@self.registry.action('Click element by index', param_model=ClickElementAction)
		async def click_element_by_index(params: ClickElementAction, browser_session: BrowserSession):
			# Browser is now a BrowserSession itself

			if params.index not in await browser_session.get_selector_map():
				raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

			element_node = await browser_session.get_dom_element_by_index(params.index)
			initial_pages = len(browser_session.tabs)

			# if element has file uploader then dont click
			if await browser_session.is_file_uploader(element_node):
				msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)

			msg = None

			try:
				download_path = await browser_session._click_element_node(element_node)
				if download_path:
					msg = f'💾  Downloaded file to {download_path}'
				else:
					msg = f'🖱️  Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

				logger.info(msg)
				logger.debug(f'Element xpath: {element_node.xpath}')
				if len(browser_session.tabs) > initial_pages:
					new_tab_msg = 'New tab opened - switching to it'
					msg += f' - {new_tab_msg}'
					logger.info(new_tab_msg)
					await browser_session.switch_to_tab(-1)
				return ActionResult(extracted_content=msg, include_in_memory=True)
			except Exception as e:
				logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
				return ActionResult(error=str(e))

		@self.registry.action(
			'Input text into a input interactive element',
			param_model=InputTextAction,
		)
		async def input_text(params: InputTextAction, browser_session: BrowserSession, has_sensitive_data: bool = False):
			if params.index not in await browser_session.get_selector_map():
				raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

			element_node = await browser_session.get_dom_element_by_index(params.index)
			await browser_session._input_text_element_node(element_node, params.text)
			if not has_sensitive_data:
				msg = f'⌨️  Input {params.text} into index {params.index}'
			else:
				msg = f'⌨️  Input sensitive data into index {params.index}'
			logger.info(msg)
			logger.debug(f'Element xpath: {element_node.xpath}')
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# Save PDF
		@self.registry.action(
			'Save the current page as a PDF file',
		)
		async def save_pdf(browser_session: BrowserSession):
			page = await browser_session.get_current_page()
			short_url = re.sub(r'^https?://(?:www\.)?|/$', '', page.url)
			slug = re.sub(r'[^a-zA-Z0-9]+', '-', short_url).strip('-').lower()
			sanitized_filename = f'{slug}.pdf'

			await page.emulate_media(media='screen')
			await page.pdf(path=sanitized_filename, format='A4', print_background=False)
			msg = f'Saving page with URL {page.url} as PDF to ./{sanitized_filename}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# Tab Management Actions
		@self.registry.action('Switch tab', param_model=SwitchTabAction)
		async def switch_tab(params: SwitchTabAction, browser_session: BrowserSession):
			await browser_session.switch_to_tab(params.page_id)
… (truncated)

```

---

## `browser-use-main\browser_use\controller\views.py`

```py
from pydantic import BaseModel, ConfigDict, Field, model_validator


# Action Input Models
class SearchGoogleAction(BaseModel):
	query: str


class GoToUrlAction(BaseModel):
	url: str


class ClickElementAction(BaseModel):
	index: int
	xpath: str | None = None


class InputTextAction(BaseModel):
	index: int
	text: str
	xpath: str | None = None


class DoneAction(BaseModel):
	text: str
	success: bool


class SwitchTabAction(BaseModel):
	page_id: int


class OpenTabAction(BaseModel):
	url: str


class CloseTabAction(BaseModel):
	page_id: int


class ScrollAction(BaseModel):
	amount: int | None = None  # The number of pixels to scroll. If None, scroll down/up one page


class SendKeysAction(BaseModel):
	keys: str


class ExtractPageContentAction(BaseModel):
	value: str


class NoParamsAction(BaseModel):
	"""
	Accepts absolutely anything in the incoming data
	and discards it, so the final parsed model is empty.
	"""

	model_config = ConfigDict(extra='allow')

	@model_validator(mode='before')
	def ignore_all_inputs(cls, values):
		# No matter what the user sends, discard it and return empty.
		return {}


class Position(BaseModel):
	x: int
	y: int


class DragDropAction(BaseModel):
	# Element-based approach
	element_source: str | None = Field(None, description='CSS selector or XPath of the element to drag from')
	element_target: str | None = Field(None, description='CSS selector or XPath of the element to drop onto')
	element_source_offset: Position | None = Field(
		None, description='Precise position within the source element to start drag (in pixels from top-left corner)'
	)
	element_target_offset: Position | None = Field(
		None, description='Precise position within the target element to drop (in pixels from top-left corner)'
	)

	# Coordinate-based approach (used if selectors not provided)
	coord_source_x: int | None = Field(None, description='Absolute X coordinate on page to start drag from (in pixels)')
	coord_source_y: int | None = Field(None, description='Absolute Y coordinate on page to start drag from (in pixels)')
	coord_target_x: int | None = Field(None, description='Absolute X coordinate on page to drop at (in pixels)')
	coord_target_y: int | None = Field(None, description='Absolute Y coordinate on page to drop at (in pixels)')

	# Common options
	steps: int | None = Field(10, description='Number of intermediate points for smoother movement (5-20 recommended)')
	delay_ms: int | None = Field(5, description='Delay in milliseconds between steps (0 for fastest, 10-20 for more natural)')

```

---

## `browser-use-main\browser_use\dom\__init__.py`

```py

```

---

## `browser-use-main\browser_use\dom\buildDomTree.js`

```js
(
  args = {
    doHighlightElements: true,
    focusHighlightIndex: -1,
    viewportExpansion: 0,
    debugMode: false,
  }
) => {
  const { doHighlightElements, focusHighlightIndex, viewportExpansion, debugMode } = args;
  let highlightIndex = 0; // Reset highlight index

  // Add timing stack to handle recursion
  const TIMING_STACK = {
    nodeProcessing: [],
    treeTraversal: [],
    highlighting: [],
    current: null
  };

  function pushTiming(type) {
    TIMING_STACK[type] = TIMING_STACK[type] || [];
    TIMING_STACK[type].push(performance.now());
  }

  function popTiming(type) {
    const start = TIMING_STACK[type].pop();
    const duration = performance.now() - start;
    return duration;
  }

  // Only initialize performance tracking if in debug mode
  const PERF_METRICS = debugMode ? {
    buildDomTreeCalls: 0,
    timings: {
      buildDomTree: 0,
      highlightElement: 0,
      isInteractiveElement: 0,
      isElementVisible: 0,
      isTopElement: 0,
      isInExpandedViewport: 0,
      isTextNodeVisible: 0,
      getEffectiveScroll: 0,
    },
    cacheMetrics: {
      boundingRectCacheHits: 0,
      boundingRectCacheMisses: 0,
      computedStyleCacheHits: 0,
      computedStyleCacheMisses: 0,
      getBoundingClientRectTime: 0,
      getComputedStyleTime: 0,
      boundingRectHitRate: 0,
      computedStyleHitRate: 0,
      overallHitRate: 0,
      clientRectsCacheHits: 0,
      clientRectsCacheMisses: 0,
    },
    nodeMetrics: {
      totalNodes: 0,
      processedNodes: 0,
      skippedNodes: 0,
    },
    buildDomTreeBreakdown: {
      totalTime: 0,
      totalSelfTime: 0,
      buildDomTreeCalls: 0,
      domOperations: {
        getBoundingClientRect: 0,
        getComputedStyle: 0,
      },
      domOperationCounts: {
        getBoundingClientRect: 0,
        getComputedStyle: 0,
      }
    }
  } : null;

  // Simple timing helper that only runs in debug mode
  function measureTime(fn) {
    if (!debugMode) return fn;
    return function (...args) {
      const start = performance.now();
      const result = fn.apply(this, args);
      const duration = performance.now() - start;
      return result;
    };
  }

  // Helper to measure DOM operations
  function measureDomOperation(operation, name) {
    if (!debugMode) return operation();

    const start = performance.now();
    const result = operation();
    const duration = performance.now() - start;

    if (PERF_METRICS && name in PERF_METRICS.buildDomTreeBreakdown.domOperations) {
      PERF_METRICS.buildDomTreeBreakdown.domOperations[name] += duration;
      PERF_METRICS.buildDomTreeBreakdown.domOperationCounts[name]++;
    }

    return result;
  }

  // Add caching mechanisms at the top level
  const DOM_CACHE = {
    boundingRects: new WeakMap(),
    clientRects: new WeakMap(),
    computedStyles: new WeakMap(),
    clearCache: () => {
      DOM_CACHE.boundingRects = new WeakMap();
      DOM_CACHE.clientRects = new WeakMap();
      DOM_CACHE.computedStyles = new WeakMap();
    }
  };

  // Cache helper functions
  function getCachedBoundingRect(element) {
    if (!element) return null;

    if (DOM_CACHE.boundingRects.has(element)) {
      if (debugMode && PERF_METRICS) {
        PERF_METRICS.cacheMetrics.boundingRectCacheHits++;
      }
      return DOM_CACHE.boundingRects.get(element);
    }

    if (debugMode && PERF_METRICS) {
      PERF_METRICS.cacheMetrics.boundingRectCacheMisses++;
    }

    let rect;
    if (debugMode) {
      const start = performance.now();
      rect = element.getBoundingClientRect();
      const duration = performance.now() - start;
      if (PERF_METRICS) {
        PERF_METRICS.buildDomTreeBreakdown.domOperations.getBoundingClientRect += duration;
        PERF_METRICS.buildDomTreeBreakdown.domOperationCounts.getBoundingClientRect++;
      }
    } else {
      rect = element.getBoundingClientRect();
    }

    if (rect) {
      DOM_CACHE.boundingRects.set(element, rect);
    }
    return rect;
  }

  function getCachedComputedStyle(element) {
    if (!element) return null;

    if (DOM_CACHE.computedStyles.has(element)) {
      if (debugMode && PERF_METRICS) {
        PERF_METRICS.cacheMetrics.computedStyleCacheHits++;
      }
      return DOM_CACHE.computedStyles.get(element);
    }

    if (debugMode && PERF_METRICS) {
      PERF_METRICS.cacheMetrics.computedStyleCacheMisses++;
    }

    let style;
    if (debugMode) {
      const start = performance.now();
      style = window.getComputedStyle(element);
      const duration = performance.now() - start;
      if (PERF_METRICS) {
        PERF_METRICS.buildDomTreeBreakdown.domOperations.getComputedStyle += duration;
        PERF_METRICS.buildDomTreeBreakdown.domOperationCounts.getComputedStyle++;
      }
    } else {
      style = window.getComputedStyle(element);
    }

    if (style) {
      DOM_CACHE.computedStyles.set(element, style);
    }
    return style;
  }

  // Add a new function to get cached client rects
  function getCachedClientRects(element) {
    if (!element) return null;
    
    if (DOM_CACHE.clientRects.has(element)) {
      if (debugMode && PERF_METRICS) {
        PERF_METRICS.cacheMetrics.clientRectsCacheHits++;
      }
      return DOM_CACHE.clientRects.get(element);
    }
    
    if (debugMode && PERF_METRICS) {
      PERF_METRICS.cacheMetrics.clientRectsCacheMisses++;
    }
    
    const rects = element.getClientRects();
    
    if (rects) {
… (truncated)

```

---

## `browser-use-main\browser_use\dom\clickable_element_processor\service.py`

```py
import hashlib

from browser_use.dom.views import DOMElementNode


class ClickableElementProcessor:
	@staticmethod
	def get_clickable_elements_hashes(dom_element: DOMElementNode) -> set[str]:
		"""Get all clickable elements in the DOM tree"""
		clickable_elements = ClickableElementProcessor.get_clickable_elements(dom_element)
		return {ClickableElementProcessor.hash_dom_element(element) for element in clickable_elements}

	@staticmethod
	def get_clickable_elements(dom_element: DOMElementNode) -> list[DOMElementNode]:
		"""Get all clickable elements in the DOM tree"""
		clickable_elements = list()
		for child in dom_element.children:
			if isinstance(child, DOMElementNode):
				if child.highlight_index:
					clickable_elements.append(child)

				clickable_elements.extend(ClickableElementProcessor.get_clickable_elements(child))

		return list(clickable_elements)

	@staticmethod
	def hash_dom_element(dom_element: DOMElementNode) -> str:
		parent_branch_path = ClickableElementProcessor._get_parent_branch_path(dom_element)
		branch_path_hash = ClickableElementProcessor._parent_branch_path_hash(parent_branch_path)
		attributes_hash = ClickableElementProcessor._attributes_hash(dom_element.attributes)
		xpath_hash = ClickableElementProcessor._xpath_hash(dom_element.xpath)
		# text_hash = DomTreeProcessor._text_hash(dom_element)

		return ClickableElementProcessor._hash_string(f'{branch_path_hash}-{attributes_hash}-{xpath_hash}')

	@staticmethod
	def _get_parent_branch_path(dom_element: DOMElementNode) -> list[str]:
		parents: list[DOMElementNode] = []
		current_element: DOMElementNode = dom_element
		while current_element.parent is not None:
			parents.append(current_element)
			current_element = current_element.parent

		parents.reverse()

		return [parent.tag_name for parent in parents]

	@staticmethod
	def _parent_branch_path_hash(parent_branch_path: list[str]) -> str:
		parent_branch_path_string = '/'.join(parent_branch_path)
		return hashlib.sha256(parent_branch_path_string.encode()).hexdigest()

	@staticmethod
	def _attributes_hash(attributes: dict[str, str]) -> str:
		attributes_string = ''.join(f'{key}={value}' for key, value in attributes.items())
		return ClickableElementProcessor._hash_string(attributes_string)

	@staticmethod
	def _xpath_hash(xpath: str) -> str:
		return ClickableElementProcessor._hash_string(xpath)

	@staticmethod
	def _text_hash(dom_element: DOMElementNode) -> str:
		""" """
		text_string = dom_element.get_all_text_till_next_clickable_element()
		return ClickableElementProcessor._hash_string(text_string)

	@staticmethod
	def _hash_string(string: str) -> str:
		return hashlib.sha256(string.encode()).hexdigest()

```

---

## `browser-use-main\browser_use\dom\history_tree_processor\service.py`

```py
import hashlib

from browser_use.dom.history_tree_processor.view import DOMHistoryElement, HashedDomElement
from browser_use.dom.views import DOMElementNode


class HistoryTreeProcessor:
	""" "
	Operations on the DOM elements

	@dev be careful - text nodes can change even if elements stay the same
	"""

	@staticmethod
	def convert_dom_element_to_history_element(dom_element: DOMElementNode) -> DOMHistoryElement:
		from browser_use.browser.context import BrowserContext

		parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
		css_selector = BrowserContext._enhanced_css_selector_for_element(dom_element)
		return DOMHistoryElement(
			dom_element.tag_name,
			dom_element.xpath,
			dom_element.highlight_index,
			parent_branch_path,
			dom_element.attributes,
			dom_element.shadow_root,
			css_selector=css_selector,
			page_coordinates=dom_element.page_coordinates,
			viewport_coordinates=dom_element.viewport_coordinates,
			viewport_info=dom_element.viewport_info,
		)

	@staticmethod
	def find_history_element_in_tree(dom_history_element: DOMHistoryElement, tree: DOMElementNode) -> DOMElementNode | None:
		hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)

		def process_node(node: DOMElementNode):
			if node.highlight_index is not None:
				hashed_node = HistoryTreeProcessor._hash_dom_element(node)
				if hashed_node == hashed_dom_history_element:
					return node
			for child in node.children:
				if isinstance(child, DOMElementNode):
					result = process_node(child)
					if result is not None:
						return result
			return None

		return process_node(tree)

	@staticmethod
	def compare_history_element_and_dom_element(dom_history_element: DOMHistoryElement, dom_element: DOMElementNode) -> bool:
		hashed_dom_history_element = HistoryTreeProcessor._hash_dom_history_element(dom_history_element)
		hashed_dom_element = HistoryTreeProcessor._hash_dom_element(dom_element)

		return hashed_dom_history_element == hashed_dom_element

	@staticmethod
	def _hash_dom_history_element(dom_history_element: DOMHistoryElement) -> HashedDomElement:
		branch_path_hash = HistoryTreeProcessor._parent_branch_path_hash(dom_history_element.entire_parent_branch_path)
		attributes_hash = HistoryTreeProcessor._attributes_hash(dom_history_element.attributes)
		xpath_hash = HistoryTreeProcessor._xpath_hash(dom_history_element.xpath)

		return HashedDomElement(branch_path_hash, attributes_hash, xpath_hash)

	@staticmethod
	def _hash_dom_element(dom_element: DOMElementNode) -> HashedDomElement:
		parent_branch_path = HistoryTreeProcessor._get_parent_branch_path(dom_element)
		branch_path_hash = HistoryTreeProcessor._parent_branch_path_hash(parent_branch_path)
		attributes_hash = HistoryTreeProcessor._attributes_hash(dom_element.attributes)
		xpath_hash = HistoryTreeProcessor._xpath_hash(dom_element.xpath)
		# text_hash = DomTreeProcessor._text_hash(dom_element)

		return HashedDomElement(branch_path_hash, attributes_hash, xpath_hash)

	@staticmethod
	def _get_parent_branch_path(dom_element: DOMElementNode) -> list[str]:
		parents: list[DOMElementNode] = []
		current_element: DOMElementNode = dom_element
		while current_element.parent is not None:
			parents.append(current_element)
			current_element = current_element.parent

		parents.reverse()

		return [parent.tag_name for parent in parents]

	@staticmethod
	def _parent_branch_path_hash(parent_branch_path: list[str]) -> str:
		parent_branch_path_string = '/'.join(parent_branch_path)
		return hashlib.sha256(parent_branch_path_string.encode()).hexdigest()

	@staticmethod
	def _attributes_hash(attributes: dict[str, str]) -> str:
		attributes_string = ''.join(f'{key}={value}' for key, value in attributes.items())
		return hashlib.sha256(attributes_string.encode()).hexdigest()

	@staticmethod
	def _xpath_hash(xpath: str) -> str:
		return hashlib.sha256(xpath.encode()).hexdigest()

	@staticmethod
	def _text_hash(dom_element: DOMElementNode) -> str:
		""" """
		text_string = dom_element.get_all_text_till_next_clickable_element()
		return hashlib.sha256(text_string.encode()).hexdigest()

```

---

## `browser-use-main\browser_use\dom\history_tree_processor\view.py`

```py
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class HashedDomElement:
	"""
	Hash of the dom element to be used as a unique identifier
	"""

	branch_path_hash: str
	attributes_hash: str
	xpath_hash: str
	# text_hash: str


class Coordinates(BaseModel):
	x: int
	y: int


class CoordinateSet(BaseModel):
	top_left: Coordinates
	top_right: Coordinates
	bottom_left: Coordinates
	bottom_right: Coordinates
	center: Coordinates
	width: int
	height: int


class ViewportInfo(BaseModel):
	scroll_x: int
	scroll_y: int
	width: int
	height: int


@dataclass
class DOMHistoryElement:
	tag_name: str
	xpath: str
	highlight_index: int | None
	entire_parent_branch_path: list[str]
	attributes: dict[str, str]
	shadow_root: bool = False
	css_selector: str | None = None
	page_coordinates: CoordinateSet | None = None
	viewport_coordinates: CoordinateSet | None = None
	viewport_info: ViewportInfo | None = None

	def to_dict(self) -> dict:
		page_coordinates = self.page_coordinates.model_dump() if self.page_coordinates else None
		viewport_coordinates = self.viewport_coordinates.model_dump() if self.viewport_coordinates else None
		viewport_info = self.viewport_info.model_dump() if self.viewport_info else None

		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'highlight_index': self.highlight_index,
			'entire_parent_branch_path': self.entire_parent_branch_path,
			'attributes': self.attributes,
			'shadow_root': self.shadow_root,
			'css_selector': self.css_selector,
			'page_coordinates': page_coordinates,
			'viewport_coordinates': viewport_coordinates,
			'viewport_info': viewport_info,
		}

```

---

## `browser-use-main\browser_use\dom\service.py`

```py
import json
import logging
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
	from playwright.async_api import Page

from browser_use.dom.views import (
	DOMBaseNode,
	DOMElementNode,
	DOMState,
	DOMTextNode,
	SelectorMap,
)
from browser_use.utils import time_execution_async

logger = logging.getLogger(__name__)


@dataclass
class ViewportInfo:
	width: int
	height: int


class DomService:
	def __init__(self, page: 'Page'):
		self.page = page
		self.xpath_cache = {}

		self.js_code = resources.files('browser_use.dom').joinpath('buildDomTree.js').read_text()

	# region - Clickable elements
	@time_execution_async('--get_clickable_elements')
	async def get_clickable_elements(
		self,
		highlight_elements: bool = True,
		focus_element: int = -1,
		viewport_expansion: int = 0,
	) -> DOMState:
		element_tree, selector_map = await self._build_dom_tree(highlight_elements, focus_element, viewport_expansion)
		return DOMState(element_tree=element_tree, selector_map=selector_map)

	@time_execution_async('--get_cross_origin_iframes')
	async def get_cross_origin_iframes(self) -> list[str]:
		# invisible cross-origin iframes are used for ads and tracking, dont open those
		hidden_frame_urls = await self.page.locator('iframe').filter(visible=False).evaluate_all('e => e.map(e => e.src)')

		is_ad_url = lambda url: any(
			domain in urlparse(url).netloc for domain in ('doubleclick.net', 'adroll.com', 'googletagmanager.com')
		)

		return [
			frame.url
			for frame in self.page.frames
			if urlparse(frame.url).netloc  # exclude data:urls and about:blank
			and urlparse(frame.url).netloc != urlparse(self.page.url).netloc  # exclude same-origin iframes
			and frame.url not in hidden_frame_urls  # exclude hidden frames
			and not is_ad_url(frame.url)  # exclude most common ad network tracker frame URLs
		]

	@time_execution_async('--build_dom_tree')
	async def _build_dom_tree(
		self,
		highlight_elements: bool,
		focus_element: int,
		viewport_expansion: int,
	) -> tuple[DOMElementNode, SelectorMap]:
		if await self.page.evaluate('1+1') != 2:
			raise ValueError('The page cannot evaluate javascript code properly')

		if self.page.url == 'about:blank':
			# short-circuit if the page is a new empty tab for speed, no need to inject buildDomTree.js
			return (
				DOMElementNode(
					tag_name='body',
					xpath='',
					attributes={},
					children=[],
					is_visible=False,
					parent=None,
				),
				{},
			)

		# NOTE: We execute JS code in the browser to extract important DOM information.
		#       The returned hash map contains information about the DOM tree and the
		#       relationship between the DOM elements.
		debug_mode = logger.getEffectiveLevel() == logging.DEBUG
		args = {
			'doHighlightElements': highlight_elements,
			'focusHighlightIndex': focus_element,
			'viewportExpansion': viewport_expansion,
			'debugMode': debug_mode,
		}

		try:
			eval_page: dict = await self.page.evaluate(self.js_code, args)
		except Exception as e:
			logger.error('Error evaluating JavaScript: %s', e)
			raise

		# Only log performance metrics in debug mode
		if debug_mode and 'perfMetrics' in eval_page:
			logger.debug(
				'DOM Tree Building Performance Metrics for: %s\n%s',
				self.page.url,
				json.dumps(eval_page['perfMetrics'], indent=2),
			)

		return await self._construct_dom_tree(eval_page)

	@time_execution_async('--construct_dom_tree')
	async def _construct_dom_tree(
		self,
		eval_page: dict,
	) -> tuple[DOMElementNode, SelectorMap]:
		js_node_map = eval_page['map']
		js_root_id = eval_page['rootId']

		selector_map = {}
		node_map = {}

		for id, node_data in js_node_map.items():
			node, children_ids = self._parse_node(node_data)
			if node is None:
				continue

			node_map[id] = node

			if isinstance(node, DOMElementNode) and node.highlight_index is not None:
				selector_map[node.highlight_index] = node

			# NOTE: We know that we are building the tree bottom up
			#       and all children are already processed.
			if isinstance(node, DOMElementNode):
				for child_id in children_ids:
					if child_id not in node_map:
						continue

					child_node = node_map[child_id]

					child_node.parent = node
					node.children.append(child_node)

		html_to_dict = node_map[str(js_root_id)]

		del node_map
		del js_node_map
		del js_root_id

		if html_to_dict is None or not isinstance(html_to_dict, DOMElementNode):
			raise ValueError('Failed to parse HTML to dictionary')

		return html_to_dict, selector_map

	def _parse_node(
		self,
		node_data: dict,
	) -> tuple[DOMBaseNode | None, list[int]]:
		if not node_data:
			return None, []

		# Process text nodes immediately
		if node_data.get('type') == 'TEXT_NODE':
			text_node = DOMTextNode(
				text=node_data['text'],
				is_visible=node_data['isVisible'],
				parent=None,
			)
			return text_node, []

		# Process coordinates if they exist for element nodes

		viewport_info = None

		if 'viewport' in node_data:
			viewport_info = ViewportInfo(
				width=node_data['viewport']['width'],
				height=node_data['viewport']['height'],
			)

		element_node = DOMElementNode(
			tag_name=node_data['tagName'],
			xpath=node_data['xpath'],
			attributes=node_data.get('attributes', {}),
			children=[],
			is_visible=node_data.get('isVisible', False),
			is_interactive=node_data.get('isInteractive', False),
			is_top_element=node_data.get('isTopElement', False),
			is_in_viewport=node_data.get('isInViewport', False),
			highlight_index=node_data.get('highlightIndex'),
			shadow_root=node_data.get('shadowRoot', False),
			parent=None,
			viewport_info=viewport_info,
		)

… (truncated)

```

---

## `browser-use-main\browser_use\dom\tests\test_accessibility_playground.py`

```py
"""
Accessibility Tree Playground for browser-use

- Launches a browser and navigates to a target URL (default: amazon.com)
- Extracts both the full and interesting-only accessibility trees using Playwright
- Prints and saves both trees to JSON files
- Recursively prints relevant info for each node (role, name, value, description, focusable, focused, checked, selected, disabled, children count)
- Explains the difference between the accessibility tree and the DOM tree
- Notes on React/Vue/SPA apps
- Easy to modify for your own experiments

Run with: python browser_use/dom/tests/test_accessibility_playground.py
"""

import asyncio

from playwright.async_api import async_playwright

# Change this to any site you want to test


# Helper to recursively print relevant info from the accessibility tree
def print_ax_tree(node, depth=0):
	if not node:
		return
	indent = '  ' * depth
	info = [
		f'role={node.get("role")!r}',
		f'name={node.get("name")!r}' if node.get('name') else None,
		f'value={node.get("value")!r}' if node.get('value') else None,
		f'desc={node.get("description")!r}' if node.get('description') else None,
		f'focusable={node.get("focusable")!r}' if 'focusable' in node else None,
		f'focused={node.get("focused")!r}' if 'focused' in node else None,
		f'checked={node.get("checked")!r}' if 'checked' in node else None,
		f'selected={node.get("selected")!r}' if 'selected' in node else None,
		f'disabled={node.get("disabled")!r}' if 'disabled' in node else None,
		f'children={len(node.get("children", []))}' if node.get('children') else None,
	]
	print('--------------------------------')
	print(indent + ', '.join([x for x in info if x]))
	for child in node.get('children', []):
		print_ax_tree(child, depth + 1)


# Helper to print all available accessibility node attributes
# Prints all key-value pairs for each node (except 'children'), then recurses into children
def print_all_fields(node, depth=0):
	if not node:
		return
	indent = '  ' * depth
	for k, v in node.items():
		if k != 'children':
			print(f'{indent}{k}: {v!r}')
	if 'children' in node:
		print(f'{indent}children: {len(node["children"])}')
		for child in node['children']:
			print_all_fields(child, depth + 1)


def flatten_ax_tree(node, lines):
	if not node:
		return
	role = node.get('role', '')
	name = node.get('name', '')
	lines.append(f'{role} {name}')
	for child in node.get('children', []):
		flatten_ax_tree(child, lines)


async def get_ax_tree(TARGET_URL):
	async with async_playwright() as p:
		browser = await p.chromium.launch(headless=True)
		page = await browser.new_page()
		print(f'Navigating to {TARGET_URL}')
		await page.goto(TARGET_URL, wait_until='domcontentloaded')

		ax_tree_interesting = await page.accessibility.snapshot(interesting_only=True)
		lines = []
		flatten_ax_tree(ax_tree_interesting, lines)
		print(lines)
		print(f'length of ax_tree_interesting: {len(lines)}')

		await browser.close()


if __name__ == '__main__':
	TARGET_URL = [
		# 'https://amazon.com/',
		# 'https://www.google.com/',
		# 'https://www.facebook.com/',
		# 'https://platform.openai.com/tokenizer',
		'https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/input/checkbox',
	]
	for url in TARGET_URL:
		asyncio.run(get_ax_tree(url))

```

---

## `browser-use-main\browser_use\dom\views.py`

```py
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from browser_use.dom.history_tree_processor.view import CoordinateSet, HashedDomElement, ViewportInfo
from browser_use.utils import time_execution_sync

# Avoid circular import issues
if TYPE_CHECKING:
	from .views import DOMElementNode


@dataclass(frozen=False)
class DOMBaseNode:
	is_visible: bool
	# Use None as default and set parent later to avoid circular reference issues
	parent: Optional['DOMElementNode']

	def __json__(self) -> dict:
		raise NotImplementedError('DOMBaseNode is an abstract class')


@dataclass(frozen=False)
class DOMTextNode(DOMBaseNode):
	text: str
	type: str = 'TEXT_NODE'

	def has_parent_with_highlight_index(self) -> bool:
		current = self.parent
		while current is not None:
			# stop if the element has a highlight index (will be handled separately)
			if current.highlight_index is not None:
				return True

			current = current.parent
		return False

	def is_parent_in_viewport(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_in_viewport

	def is_parent_top_element(self) -> bool:
		if self.parent is None:
			return False
		return self.parent.is_top_element

	def __json__(self) -> dict:
		return {
			'text': self.text,
			'type': self.type,
		}


@dataclass(frozen=False)
class DOMElementNode(DOMBaseNode):
	"""
	xpath: the xpath of the element from the last root node (shadow root or iframe OR document if no shadow root or iframe).
	To properly reference the element we need to recursively switch the root node until we find the element (work you way up the tree with `.parent`)
	"""

	tag_name: str
	xpath: str
	attributes: dict[str, str]
	children: list[DOMBaseNode]
	is_interactive: bool = False
	is_top_element: bool = False
	is_in_viewport: bool = False
	shadow_root: bool = False
	highlight_index: int | None = None
	viewport_coordinates: CoordinateSet | None = None
	page_coordinates: CoordinateSet | None = None
	viewport_info: ViewportInfo | None = None

	"""
	### State injected by the browser context.

	The idea is that the clickable elements are sometimes persistent from the previous page -> tells the model which objects are new/_how_ the state has changed
	"""
	is_new: bool | None = None

	def __json__(self) -> dict:
		return {
			'tag_name': self.tag_name,
			'xpath': self.xpath,
			'attributes': self.attributes,
			'is_visible': self.is_visible,
			'is_interactive': self.is_interactive,
			'is_top_element': self.is_top_element,
			'is_in_viewport': self.is_in_viewport,
			'shadow_root': self.shadow_root,
			'highlight_index': self.highlight_index,
			'viewport_coordinates': self.viewport_coordinates,
			'page_coordinates': self.page_coordinates,
			'children': [child.__json__() for child in self.children],
		}

	def __repr__(self) -> str:
		tag_str = f'<{self.tag_name}'

		# Add attributes
		for key, value in self.attributes.items():
			tag_str += f' {key}="{value}"'
		tag_str += '>'

		# Add extra info
		extras = []
		if self.is_interactive:
			extras.append('interactive')
		if self.is_top_element:
			extras.append('top')
		if self.shadow_root:
			extras.append('shadow-root')
		if self.highlight_index is not None:
			extras.append(f'highlight:{self.highlight_index}')
		if self.is_in_viewport:
			extras.append('in-viewport')

		if extras:
			tag_str += f' [{", ".join(extras)}]'

		return tag_str

	@cached_property
	def hash(self) -> HashedDomElement:
		from browser_use.dom.history_tree_processor.service import (
			HistoryTreeProcessor,
		)

		return HistoryTreeProcessor._hash_dom_element(self)

	def get_all_text_till_next_clickable_element(self, max_depth: int = -1) -> str:
		text_parts = []

		def collect_text(node: DOMBaseNode, current_depth: int) -> None:
			if max_depth != -1 and current_depth > max_depth:
				return

			# Skip this branch if we hit a highlighted element (except for the current node)
			if isinstance(node, DOMElementNode) and node != self and node.highlight_index is not None:
				return

			if isinstance(node, DOMTextNode):
				text_parts.append(node.text)
			elif isinstance(node, DOMElementNode):
				for child in node.children:
					collect_text(child, current_depth + 1)

		collect_text(self, 0)
		return '\n'.join(text_parts).strip()

	@time_execution_sync('--clickable_elements_to_string')
	def clickable_elements_to_string(self, include_attributes: list[str] | None = None) -> str:
		"""Convert the processed DOM content to HTML."""
		formatted_text = []

		def process_node(node: DOMBaseNode, depth: int) -> None:
			next_depth = int(depth)
			depth_str = depth * '\t'

			if isinstance(node, DOMElementNode):
				# Add element with highlight_index
				if node.highlight_index is not None:
					next_depth += 1

					text = node.get_all_text_till_next_clickable_element()
					attributes_html_str = ''
					if include_attributes:
						attributes_to_include = {
							key: str(value) for key, value in node.attributes.items() if key in include_attributes
						}

						# Easy LLM optimizations
						# if tag == role attribute, don't include it
						if node.tag_name == attributes_to_include.get('role'):
							del attributes_to_include['role']

						# if aria-label == text of the node, don't include it
						if (
							attributes_to_include.get('aria-label')
							and attributes_to_include.get('aria-label', '').strip() == text.strip()
						):
							del attributes_to_include['aria-label']

						# if placeholder == text of the node, don't include it
						if (
							attributes_to_include.get('placeholder')
							and attributes_to_include.get('placeholder', '').strip() == text.strip()
						):
							del attributes_to_include['placeholder']

						if attributes_to_include:
							# Format as key1='value1' key2='value2'
							attributes_html_str = ' '.join(f"{key}='{value}'" for key, value in attributes_to_include.items())

					# Build the line
					if node.is_new:
						highlight_indicator = f'*[{node.highlight_index}]*'
					else:
						highlight_indicator = f'[{node.highlight_index}]'
… (truncated)

```

---

## `browser-use-main\browser_use\exceptions.py`

```py
class LLMException(Exception):
	def __init__(self, status_code, message):
		self.status_code = status_code
		self.message = message
		super().__init__(f'Error {status_code}: {message}')

```

---

## `browser-use-main\browser_use\logging_config.py`

```py
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def addLoggingLevel(levelName, levelNum, methodName=None):
	"""
	Comprehensively adds a new logging level to the `logging` module and the
	currently configured logging class.

	`levelName` becomes an attribute of the `logging` module with the value
	`levelNum`. `methodName` becomes a convenience method for both `logging`
	itself and the class returned by `logging.getLoggerClass()` (usually just
	`logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
	used.

	To avoid accidental clobberings of existing attributes, this method will
	raise an `AttributeError` if the level name is already an attribute of the
	`logging` module or if the method name is already present

	Example
	-------
	>>> addLoggingLevel('TRACE', logging.DEBUG - 5)
	>>> logging.getLogger(__name__).setLevel('TRACE')
	>>> logging.getLogger(__name__).trace('that worked')
	>>> logging.trace('so did this')
	>>> logging.TRACE
	5

	"""
	if not methodName:
		methodName = levelName.lower()

	if hasattr(logging, levelName):
		raise AttributeError(f'{levelName} already defined in logging module')
	if hasattr(logging, methodName):
		raise AttributeError(f'{methodName} already defined in logging module')
	if hasattr(logging.getLoggerClass(), methodName):
		raise AttributeError(f'{methodName} already defined in logger class')

	# This method was inspired by the answers to Stack Overflow post
	# http://stackoverflow.com/q/2183233/2988730, especially
	# http://stackoverflow.com/a/13638084/2988730
	def logForLevel(self, message, *args, **kwargs):
		if self.isEnabledFor(levelNum):
			self._log(levelNum, message, args, **kwargs)

	def logToRoot(message, *args, **kwargs):
		logging.log(levelNum, message, *args, **kwargs)

	logging.addLevelName(levelNum, levelName)
	setattr(logging, levelName, levelNum)
	setattr(logging.getLoggerClass(), methodName, logForLevel)
	setattr(logging, methodName, logToRoot)


def setup_logging():
	# Try to add RESULT level, but ignore if it already exists
	try:
		addLoggingLevel('RESULT', 35)  # This allows ERROR, FATAL and CRITICAL
	except AttributeError:
		pass  # Level already exists, which is fine

	log_type = os.getenv('BROWSER_USE_LOGGING_LEVEL', 'info').lower()

	# Check if handlers are already set up
	if logging.getLogger().hasHandlers():
		return

	# Clear existing handlers
	root = logging.getLogger()
	root.handlers = []

	class BrowserUseFormatter(logging.Formatter):
		def format(self, record):
			if isinstance(record.name, str) and record.name.startswith('browser_use.'):
				record.name = record.name.split('.')[-2]
			return super().format(record)

	# Setup single handler for all loggers
	console = logging.StreamHandler(sys.stdout)

	# adittional setLevel here to filter logs
	if log_type == 'result':
		console.setLevel('RESULT')
		console.setFormatter(BrowserUseFormatter('%(message)s'))
	else:
		console.setFormatter(BrowserUseFormatter('%(levelname)-8s [%(name)s] %(message)s'))

	# Configure root logger only
	root.addHandler(console)

	# switch cases for log_type
	if log_type == 'result':
		root.setLevel('RESULT')  # string usage to avoid syntax error
	elif log_type == 'debug':
		root.setLevel(logging.DEBUG)
	else:
		root.setLevel(logging.INFO)

	# Configure browser_use logger
	browser_use_logger = logging.getLogger('browser_use')
	browser_use_logger.propagate = False  # Don't propagate to root logger
	browser_use_logger.addHandler(console)
	browser_use_logger.setLevel(root.level)  # Set same level as root logger

	logger = logging.getLogger('browser_use')
	# logger.info('BrowserUse logging setup complete with level %s', log_type)
	# Silence or adjust third-party loggers
	third_party_loggers = [
		'WDM',
		'httpx',
		'selenium',
		'playwright',
		'urllib3',
		'asyncio',
		'langchain',
		'openai',
		'httpcore',
		'charset_normalizer',
		'anthropic._base_client',
		'PIL.PngImagePlugin',
		'trafilatura.htmlprocessing',
		'trafilatura',
		'mem0',
		'mem0.vector_stores.faiss',
		'mem0.vector_stores',
		'mem0.memory',
	]
	for logger_name in third_party_loggers:
		third_party = logging.getLogger(logger_name)
		third_party.setLevel(logging.ERROR)
		third_party.propagate = False

```

---

## `browser-use-main\browser_use\README.md`

```md
# Codebase Structure

> The code structure inspired by https://github.com/Netflix/dispatch.

Very good structure on how to make a scalable codebase is also in [this repo](https://github.com/zhanymkanov/fastapi-best-practices).

Just a brief document about how we should structure our backend codebase.

## Code Structure

```markdown
src/
/<service name>/
models.py
services.py
prompts.py
views.py
utils.py
routers.py

    	/_<subservice name>/
```

### Service.py

Always a single file, except if it becomes too long - more than ~500 lines, split it into \_subservices

### Views.py

Always split the views into two parts

```python
# All
...

# Requests
...

# Responses
...
```

If too long → split into multiple files

### Prompts.py

Single file; if too long → split into multiple files (one prompt per file or so)

### Routers.py

Never split into more than one file

```

---

## `browser-use-main\browser_use\telemetry\__init__.py`

```py
"""
Telemetry for Browser Use.
"""

from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import BaseTelemetryEvent, ControllerRegisteredFunctionsTelemetryEvent

__all__ = ['BaseTelemetryEvent', 'ControllerRegisteredFunctionsTelemetryEvent', 'ProductTelemetry']

```

---

## `browser-use-main\browser_use\telemetry\service.py`

```py
import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from posthog import Posthog

from browser_use.telemetry.views import BaseTelemetryEvent
from browser_use.utils import singleton

load_dotenv()


logger = logging.getLogger(__name__)


POSTHOG_EVENT_SETTINGS = {
	'process_person_profile': True,
}


def xdg_cache_home() -> Path:
	default = Path.home() / '.cache'
	env_var = os.getenv('XDG_CACHE_HOME')
	if env_var and (path := Path(env_var)).is_absolute():
		return path
	return default


@singleton
class ProductTelemetry:
	"""
	Service for capturing anonymized telemetry data.

	If the environment variable `ANONYMIZED_TELEMETRY=False`, anonymized telemetry will be disabled.
	"""

	USER_ID_PATH = str(xdg_cache_home() / 'browser_use' / 'telemetry_user_id')
	PROJECT_API_KEY = 'phc_F8JMNjW1i2KbGUTaW1unnDdLSPCoyc52SGRU0JecaUh'
	HOST = 'https://eu.i.posthog.com'
	UNKNOWN_USER_ID = 'UNKNOWN'

	_curr_user_id = None

	def __init__(self) -> None:
		telemetry_disabled = os.getenv('ANONYMIZED_TELEMETRY', 'true').lower() == 'false'
		self.debug_logging = os.getenv('BROWSER_USE_LOGGING_LEVEL', 'info').lower() == 'debug'

		if telemetry_disabled:
			self._posthog_client = None
		else:
			logger.info(
				'Anonymized telemetry enabled. See https://docs.browser-use.com/development/telemetry for more information.'
			)
			self._posthog_client = Posthog(
				project_api_key=self.PROJECT_API_KEY,
				host=self.HOST,
				disable_geoip=False,
				enable_exception_autocapture=True,
			)

			# Silence posthog's logging
			if not self.debug_logging:
				posthog_logger = logging.getLogger('posthog')
				posthog_logger.disabled = True

		if self._posthog_client is None:
			logger.debug('Telemetry disabled')

	def capture(self, event: BaseTelemetryEvent) -> None:
		if self._posthog_client is None:
			return

		if self.debug_logging:
			logger.debug(f'Telemetry event: {event.name} {event.properties}')
		self._direct_capture(event)

	def _direct_capture(self, event: BaseTelemetryEvent) -> None:
		"""
		Should not be thread blocking because posthog magically handles it
		"""
		if self._posthog_client is None:
			return

		try:
			self._posthog_client.capture(
				self.user_id,
				event.name,
				{**event.properties, **POSTHOG_EVENT_SETTINGS},
			)
		except Exception as e:
			logger.error(f'Failed to send telemetry event {event.name}: {e}')

	def flush(self) -> None:
		if self._posthog_client:
			try:
				self._posthog_client.flush()
				logger.debug('PostHog client telemetry queue flushed.')
			except Exception as e:
				logger.error(f'Failed to flush PostHog client: {e}')
		else:
			logger.debug('PostHog client not available, skipping flush.')

	@property
	def user_id(self) -> str:
		if self._curr_user_id:
			return self._curr_user_id

		# File access may fail due to permissions or other reasons. We don't want to
		# crash so we catch all exceptions.
		try:
			if not os.path.exists(self.USER_ID_PATH):
				os.makedirs(os.path.dirname(self.USER_ID_PATH), exist_ok=True)
				with open(self.USER_ID_PATH, 'w') as f:
					new_user_id = str(uuid.uuid4())
					f.write(new_user_id)
				self._curr_user_id = new_user_id
			else:
				with open(self.USER_ID_PATH) as f:
					self._curr_user_id = f.read()
		except Exception:
			self._curr_user_id = 'UNKNOWN_USER_ID'
		return self._curr_user_id

```

---

## `browser-use-main\browser_use\telemetry\views.py`

```py
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BaseTelemetryEvent(ABC):
	@property
	@abstractmethod
	def name(self) -> str:
		pass

	@property
	def properties(self) -> dict[str, Any]:
		return {k: v for k, v in asdict(self).items() if k != 'name'}


@dataclass
class RegisteredFunction:
	name: str
	params: dict[str, Any]


@dataclass
class ControllerRegisteredFunctionsTelemetryEvent(BaseTelemetryEvent):
	registered_functions: list[RegisteredFunction]
	name: str = 'controller_registered_functions'


@dataclass
class AgentTelemetryEvent(BaseTelemetryEvent):
	# start details
	task: str
	model: str
	model_provider: str
	planner_llm: str | None
	max_steps: int
	max_actions_per_step: int
	use_vision: bool
	use_validation: bool
	version: str
	source: str
	# step details
	action_errors: Sequence[str | None]
	action_history: Sequence[list[dict] | None]
	urls_visited: Sequence[str | None]
	# end details
	steps: int
	total_input_tokens: int
	total_duration_seconds: float
	success: bool | None
	final_result_response: str | None
	error_message: str | None

	name: str = 'agent_event'

```

---

## `browser-use-main\browser_use\utils.py`

```py
import asyncio
import logging
import os
import platform
import signal
import time
from collections.abc import Callable, Coroutine
from functools import wraps
from sys import stderr
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

# Global flag to prevent duplicate exit messages
_exiting = False

# Define generic type variables for return type and parameters
R = TypeVar('R')
P = ParamSpec('P')


class SignalHandler:
	"""
	A modular and reusable signal handling system for managing SIGINT (Ctrl+C), SIGTERM,
	and other signals in asyncio applications.

	This class provides:
	- Configurable signal handling for SIGINT and SIGTERM
	- Support for custom pause/resume callbacks
	- Management of event loop state across signals
	- Standardized handling of first and second Ctrl+C presses
	- Cross-platform compatibility (with simplified behavior on Windows)
	"""

	def __init__(
		self,
		loop: asyncio.AbstractEventLoop | None = None,
		pause_callback: Callable[[], None] | None = None,
		resume_callback: Callable[[], None] | None = None,
		custom_exit_callback: Callable[[], None] | None = None,
		exit_on_second_int: bool = True,
		interruptible_task_patterns: list[str] = None,
	):
		"""
		Initialize the signal handler.

		Args:
			loop: The asyncio event loop to use. Defaults to current event loop.
			pause_callback: Function to call when system is paused (first Ctrl+C)
			resume_callback: Function to call when system is resumed
			custom_exit_callback: Function to call on exit (second Ctrl+C or SIGTERM)
			exit_on_second_int: Whether to exit on second SIGINT (Ctrl+C)
			interruptible_task_patterns: List of patterns to match task names that should be
										 canceled on first Ctrl+C (default: ['step', 'multi_act', 'get_next_action'])
		"""
		self.loop = loop or asyncio.get_event_loop()
		self.pause_callback = pause_callback
		self.resume_callback = resume_callback
		self.custom_exit_callback = custom_exit_callback
		self.exit_on_second_int = exit_on_second_int
		self.interruptible_task_patterns = interruptible_task_patterns or ['step', 'multi_act', 'get_next_action']
		self.is_windows = platform.system() == 'Windows'

		# Initialize loop state attributes
		self._initialize_loop_state()

		# Store original signal handlers to restore them later if needed
		self.original_sigint_handler = None
		self.original_sigterm_handler = None

	def _initialize_loop_state(self) -> None:
		"""Initialize loop state attributes used for signal handling."""
		setattr(self.loop, 'ctrl_c_pressed', False)
		setattr(self.loop, 'waiting_for_input', False)

	def register(self) -> None:
		"""Register signal handlers for SIGINT and SIGTERM."""
		try:
			if self.is_windows:
				# On Windows, use simple signal handling with immediate exit on Ctrl+C
				def windows_handler(sig, frame):
					print('\n\n🛑 Got Ctrl+C. Exiting immediately on Windows...\n', file=stderr)
					# Run the custom exit callback if provided
					if self.custom_exit_callback:
						self.custom_exit_callback()
					os._exit(0)

				self.original_sigint_handler = signal.signal(signal.SIGINT, windows_handler)
			else:
				# On Unix-like systems, use asyncio's signal handling for smoother experience
				self.original_sigint_handler = self.loop.add_signal_handler(signal.SIGINT, lambda: self.sigint_handler())
				self.original_sigterm_handler = self.loop.add_signal_handler(signal.SIGTERM, lambda: self.sigterm_handler())

		except Exception:
			# there are situations where signal handlers are not supported, e.g.
			# - when running in a thread other than the main thread
			# - some operating systems
			# - inside jupyter notebooks
			pass

	def unregister(self) -> None:
		"""Unregister signal handlers and restore original handlers if possible."""
		try:
			if self.is_windows:
				# On Windows, just restore the original SIGINT handler
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
			else:
				# On Unix-like systems, use asyncio's signal handler removal
				self.loop.remove_signal_handler(signal.SIGINT)
				self.loop.remove_signal_handler(signal.SIGTERM)

				# Restore original handlers if available
				if self.original_sigint_handler:
					signal.signal(signal.SIGINT, self.original_sigint_handler)
				if self.original_sigterm_handler:
					signal.signal(signal.SIGTERM, self.original_sigterm_handler)
		except Exception as e:
			logger.warning(f'Error while unregistering signal handlers: {e}')

	def _handle_second_ctrl_c(self) -> None:
		"""
		Handle a second Ctrl+C press by performing cleanup and exiting.
		This is shared logic used by both sigint_handler and wait_for_resume.
		"""
		global _exiting

		if not _exiting:
			_exiting = True

			# Call custom exit callback if provided
			if self.custom_exit_callback:
				try:
					self.custom_exit_callback()
				except Exception as e:
					logger.error(f'Error in exit callback: {e}')

		# Force immediate exit - more reliable than sys.exit()
		print('\n\n🛑  Got second Ctrl+C. Exiting immediately...\n', file=stderr)

		# Reset terminal to a clean state by sending multiple escape sequences
		# Order matters for terminal resets - we try different approaches

		# Reset terminal modes for both stdout and stderr
		print('\033[?25h', end='', flush=True, file=stderr)  # Show cursor
		print('\033[?25h', end='', flush=True)  # Show cursor

		# Reset text attributes and terminal modes
		print('\033[0m', end='', flush=True, file=stderr)  # Reset text attributes
		print('\033[0m', end='', flush=True)  # Reset text attributes

		# Disable special input modes that may cause arrow keys to output control chars
		print('\033[?1l', end='', flush=True, file=stderr)  # Reset cursor keys to normal mode
		print('\033[?1l', end='', flush=True)  # Reset cursor keys to normal mode

		# Disable bracketed paste mode
		print('\033[?2004l', end='', flush=True, file=stderr)
		print('\033[?2004l', end='', flush=True)

		# Carriage return helps ensure a clean line
		print('\r', end='', flush=True, file=stderr)
		print('\r', end='', flush=True)

		os._exit(0)

	def sigint_handler(self) -> None:
		"""
		SIGINT (Ctrl+C) handler.

		First Ctrl+C: Cancel current step and pause.
		Second Ctrl+C: Exit immediately if exit_on_second_int is True.
		"""
		global _exiting

		if _exiting:
			# Already exiting, force exit immediately
			os._exit(0)

		if getattr(self.loop, 'ctrl_c_pressed', False):
			# If we're in the waiting for input state, let the pause method handle it
			if getattr(self.loop, 'waiting_for_input', False):
				return

			# Second Ctrl+C - exit immediately if configured to do so
			if self.exit_on_second_int:
				self._handle_second_ctrl_c()

		# Mark that Ctrl+C was pressed
		self.loop.ctrl_c_pressed = True

		# Cancel current tasks that should be interruptible - this is crucial for immediate pausing
		self._cancel_interruptible_tasks()

		# Call pause callback if provided - this sets the paused flag
		if self.pause_callback:
			try:
				self.pause_callback()
			except Exception as e:
				logger.error(f'Error in pause callback: {e}')

… (truncated)

```

---

## `browser-use-main\docs\mint.json`

```json
{
  "$schema": "https://mintlify.com/schema.json",
  "name": "Browser Use",
  "logo": {
    "dark": "/logo/dark.svg",
    "light": "/logo/light.svg",
    "href": "https://browser-use.com"
  },
  "favicon": "/favicon.svg",
  "colors": {
    "primary": "#F97316",
    "light": "#FFF7ED",
    "dark": "#C2410C",
    "anchors": {
      "from": "#F97316",
      "to": "#FB923C"
    },
    "background": {
      "dark": "#0D0A09"
    }
  },
  "feedback": {
    "thumbsRating": true,
    "raiseIssue": true,
    "suggestEdit": true
  },
  "topbarLinks": [
    {
      "name": "Github",
      "url": "https://github.com/browser-use/browser-use"
    },
    {
      "name": "Twitter",
      "url": "https://x.com/gregpr07"
    }
  ],
  "topbarCtaButton": {
    "name": "Join Discord",
    "url": "https://link.browser-use.com/discord"
  },
  "tabs": [
    {
      "name": "Cloud API",
      "url": "cloud",
      "openapi": "https://api.browser-use.com/openapi.json"
    }
  ],
  "navigation": [
    {
      "group": "Get Started",
      "pages": ["introduction", "quickstart"]
    },
    {
      "group": "Customize",
      "pages": [
        "customize/supported-models",
        "customize/agent-settings",
        "customize/browser-settings",
        "customize/real-browser",
        "customize/output-format",
        "customize/system-prompt",
        "customize/sensitive-data",
        "customize/custom-functions",
        "customize/hooks"
      ]
    },
    {
      "group": "Development",
      "pages": [
        "development/contribution-guide",
        "development/local-setup",
        "development/telemetry",
        "development/observability",
        "development/evaluations",
        "development/roadmap"
      ]
    },
    {
      "group": "Cloud API",
      "pages": ["cloud/quickstart", "cloud/implementation", "cloud/webhooks"]
    }
  ],
  "footerSocials": {
    "x": "https://x.com/gregpr07",
    "github": "https://github.com/browser-use/browser-use",
    "linkedin": "https://linkedin.com/company/browser-use"
  }
}

```

---

## `browser-use-main\docs\README.md`

```md
# Docs

The official documentation for Browser Use. The docs are published to [Browser Use Docs](https://docs.browser-use.com).

### Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mintlify) to preview the documentation changes locally. To install, use the following command

```
npm i -g mintlify
```

Run the following command at the root of your documentation (where mint.json is)

```
mintlify dev
```

```

---

## `browser-use-main\eval\claude-3.5.py`

```py
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatAnthropic(
		model_name='claude-3-5-sonnet-20240620',
		temperature=0.0,
		timeout=100,
		stop=None,
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\claude-3.6.py`

```py
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatAnthropic(
		model_name='claude-3-5-sonnet-20241022',
		temperature=0.0,
		timeout=100,
		stop=None,
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\claude-3.7.py`

```py
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatAnthropic(
		model_name='claude-3-7-sonnet-20250219',
		temperature=0.0,
		timeout=100,
		stop=None,
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\deepseek-r1.py`

```py
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent, Browser

load_dotenv()

api_key_deepseek = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key_deepseek:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		base_url='https://api.deepseek.com/v1',
		model='deepseek-reasoner',
		api_key=SecretStr(api_key_deepseek),
	)
	agent = Agent(task=task, llm=llm, use_vision=False, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\deepseek.py`

```py
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent, Browser

load_dotenv()

api_key_deepseek = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key_deepseek:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		base_url='https://api.deepseek.com/v1',
		model='deepseek-chat',
		api_key=SecretStr(api_key_deepseek),
	)
	agent = Agent(task=task, llm=llm, use_vision=False, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gemini-1.5-flash.py`

```py
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, Browser

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY', '')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', api_key=SecretStr(api_key))
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gemini-2.0-flash.py`

```py
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, Browser

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY', '')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gemini-2.5-preview.py`

```py
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, Browser

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY', '')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatGoogleGenerativeAI(model='gemini-2.5-pro-preview-03-25', api_key=SecretStr(api_key))
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gpt-4.1.py`

```py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		model='gpt-4.1-2025-04-14',
		temperature=0.0,
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gpt-4o-no-boundingbox.py`

```py
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	browser.config.new_context_config.highlight_elements = False
	llm = ChatOpenAI(
		model='gpt-4o',
		temperature=0.0,
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result


if __name__ == '__main__':
	task = 'Open 1 random Wikipedia pages in new tab'
	result = asyncio.run(run_agent(task))

```

---

## `browser-use-main\eval\gpt-4o-no-vision.py`

```py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		model='gpt-4o',
		temperature=0.0,
	)
	agent = Agent(task=task, llm=llm, use_vision=False, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gpt-4o-viewport-0.py`

```py
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		model='gpt-4o',
		temperature=0.0,
	)
	browser.config.new_context_config.viewport_expansion = 0
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result


if __name__ == '__main__':
	task = 'Go to https://www.google.com and search for "python" and click on the first result'
	result = asyncio.run(run_agent(task))
	print(result)

```

---

## `browser-use-main\eval\gpt-4o.py`

```py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		model='gpt-4o',
		temperature=0.0,
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\gpt-o4-mini.py`

```py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

load_dotenv()


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	llm = ChatOpenAI(
		model='o4-mini-2025-04-16',
	)
	agent = Agent(task=task, llm=llm, browser=browser)
	result = await agent.run(max_steps=max_steps)
	return result

```

---

## `browser-use-main\eval\grok.py`

```py
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent, Browser

load_dotenv()

api_key = os.getenv('GROK_API_KEY', '')
if not api_key:
	raise ValueError('GROK_API_KEY is not set')


async def run_agent(task: str, browser: Browser | None = None, max_steps: int = 38):
	browser = browser or Browser()
	agent = Agent(
		task=task,
		use_vision=False,
		llm=ChatOpenAI(model='grok-3-beta', base_url='https://api.x.ai/v1', api_key=SecretStr(api_key)),
		browser=browser,
	)

	await agent.run()

```

---

## `browser-use-main\eval\service.py`

```py
# ==============================================================================================================
# Documentation for this evaluation file.
# The import


# Here is the command to run the evaluation:
# python eval/service.py --parallel_runs 5 --parallel_evaluations 5 --max-steps 25 --start 0 --end 100 --model gpt-4o
# options:
# --parallel_runs: Number of parallel tasks to run
# --max-steps: Maximum steps per task
# --start: Start index
# --end: End index (exclusive)
# --headless: Run in headless mode

# Here is the command to run the evaluation only:
# python eval/service.py --evaluate-only
# options:
# --parallel_evaluations: Number of parallel evaluations to run

# ==============================================================================================================


# ==============================================================================================================
# This is the LLM as a judge evaluation system from the OSU-NLP Group paper
# Any adaptiations made should be explicitly stated here:
# Adaptations:
# We are using our langchain wrapper for the OpenAI API
# This means we changed model.generate to model.invoke. The behavior of the model should be identical.
# Added a Online_Mind2Web_eval_with_retry wrapper with retry logic in case of API rate limiting or other issues.


# @article{xue2025illusionprogressassessingcurrent,
#       title={An Illusion of Progress? Assessing the Current State of Web Agents},
#       author={Tianci Xue and Weijian Qi and Tianneng Shi and Chan Hee Song and Boyu Gou and Dawn Song and Huan Sun and Yu Su},
#       year={2025},
#       eprint={2504.01382},
#       archivePrefix={arXiv},
#       primaryClass={cs.AI},
#       url={https://arxiv.org/abs/2504.01382},
# }

# @inproceedings{deng2023mind2web,
#  author = {Deng, Xiang and Gu, Yu and Zheng, Boyuan and Chen, Shijie and Stevens, Sam and Wang, Boshi and Sun, Huan and Su, Yu},
#  booktitle = {Advances in Neural Information Processing Systems},
#  editor = {A. Oh and T. Naumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
#  pages = {28091--28114},
#  publisher = {Curran Associates, Inc.},
#  title = {Mind2Web: Towards a Generalist Agent for the Web},
#  url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/5950bf290a1570ea401bf98882128160-Paper-Datasets_and_Benchmarks.pdf},
#  volume = {36},
#  year = {2023}
# }
# ==============================================================================================================
import asyncio
import base64
import io
import logging
import re
import shutil

import anyio
from PIL import Image

MAX_IMAGE = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def encode_image(image):
	"""Convert a PIL image to base64 string."""
	if image.mode == 'RGBA':
		image = image.convert('RGB')
	buffered = io.BytesIO()
	image.save(buffered, format='JPEG')
	return base64.b64encode(buffered.getvalue()).decode('utf-8')


async def identify_key_points(task, model):
	system_msg = """You are an expert tasked with analyzing a given task to identify the key points explicitly stated in the task description.

**Objective**: Carefully analyze the task description and extract the critical elements explicitly mentioned in the task for achieving its goal.

**Instructions**:
1. Read the task description carefully.
2. Identify and extract **key points** directly stated in the task description.
   - A **key point** is a critical element, condition, or step explicitly mentioned in the task description.
   - Do not infer or add any unstated elements.
   - Words such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" must go through the sort function(e.g., the key point should be "Filter by highest").

**Respond with**:
- **Key Points**: A numbered list of the explicit key points for completing this task, one per line, without explanations or additional details."""
	prompt = """Task: {task}"""
	text = prompt.format(task=task)
	messages = [
		{'role': 'system', 'content': system_msg},
		{
			'role': 'user',
			'content': [{'type': 'text', 'text': text}],
		},
	]
	response = await asyncio.to_thread(model.invoke, messages)
	return response.content


async def judge_image(task, image_path, key_points, model):
	system_msg = """You are an expert evaluator tasked with determining whether an image contains information about the necessary steps to complete a task.

**Objective**: Analyze the provided image and decide if it shows essential steps or evidence required for completing the task. Use your reasoning to explain your decision before assigning a score.

**Instructions**:
1. Provide a detailed description of the image, including its contents, visible elements, text (if any), and any notable features.

2. Carefully examine the image and evaluate whether it contains necessary steps or evidence crucial to task completion:  
- Identify key points that could be relevant to task completion, such as actions, progress indicators, tool usage, applied filters, or step-by-step instructions.  
- Does the image show actions, progress indicators, or critical information directly related to completing the task?  
- Is this information indispensable for understanding or ensuring task success?
- If the image contains partial but relevant information, consider its usefulness rather than dismissing it outright.

3. Provide your response in the following format:  
- **Reasoning**: Explain your thought process and observations. Mention specific elements in the image that indicate necessary steps, evidence, or lack thereof.  
- **Score**: Assign a score based on the reasoning, using the following scale:  
    - **1**: The image does not contain any necessary steps or relevant information.  
    - **2**: The image contains minimal or ambiguous information, unlikely to be essential.  
    - **3**: The image includes some relevant steps or hints but lacks clarity or completeness.  
    - **4**: The image contains important steps or evidence that are highly relevant but not fully comprehensive.  
    - **5**: The image clearly displays necessary steps or evidence crucial for completing the task.

Respond with:  
1. **Reasoning**: [Your explanation]  
2. **Score**: [1-5]"""

	jpg_base64_str = encode_image(Image.open(image_path))

	prompt = """**Task**: {task}

**Key Points for Task Completion**: {key_points}

The snapshot of the web page is shown in the image."""
	text = prompt.format(task=task, key_points=key_points)

	messages = [
		{'role': 'system', 'content': system_msg},
		{
			'role': 'user',
			'content': [
				{'type': 'text', 'text': text},
				{
					'type': 'image_url',
					'image_url': {'url': f'data:image/jpeg;base64,{jpg_base64_str}', 'detail': 'high'},
				},
			],
		},
	]
	response = await asyncio.to_thread(model.invoke, messages)
	return response.content


async def Online_Mind2Web_eval(task, last_actions, images_path, model, score_threshold):
	system_msg = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest,"
… (truncated)

```

---

## `browser-use-main\examples\browser\real_browser.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser, BrowserConfig

browser = Browser(
	config=BrowserConfig(
		# NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
		browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	)
)


async def main():
	agent = Agent(
		task='In docs.google.com write my Papa a quick letter',
		llm=ChatOpenAI(model='gpt-4o'),
		browser=browser,
	)

	await agent.run()
	await browser.close()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\browser\stealth.py`

```py
import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from imgcat import imgcat
from langchain_openai import ChatOpenAI
from patchright.async_api import async_playwright as async_patchright

from browser_use.browser import BrowserSession

llm = ChatOpenAI(model='gpt-4o')

terminal_width, terminal_height = shutil.get_terminal_size((80, 20))


async def main():
	# Default Playwright Chromium Browser
	normal_browser_session = BrowserSession(
		# executable_path=<defaults to playwright builtin browser stored in ms-cache directory>,
		user_data_dir=None,
		headless=False,
		# deterministic_rendering=False,
		# disable_security=False,
	)
	await normal_browser_session.start()
	await normal_browser_session.create_new_tab('https://abrahamjuliot.github.io/creepjs/')
	await asyncio.sleep(5)
	await (await normal_browser_session.get_current_page()).screenshot(path='normal_browser.png')
	imgcat(Path('normal_browser.png').read_bytes(), height=max(terminal_height - 15, 40))
	await normal_browser_session.close()

	print('\n\nPATCHRIGHT STEALTH BROWSER:')
	patchright_browser_session = BrowserSession(
		# cdp_url='wss://browser.zenrows.com?apikey=your-api-key-here&proxy_region=na',
		#                or try anchor browser, browserless, steel.dev, browserbase, oxylabs, brightdata, etc.
		playwright=await async_patchright().start(),
		user_data_dir='~/.config/browseruse/profiles/stealth',
		headless=False,
		disable_security=False,
		deterministic_rendering=False,
	)
	await patchright_browser_session.start()
	await patchright_browser_session.create_new_tab('https://abrahamjuliot.github.io/creepjs/')
	await asyncio.sleep(5)
	await (await patchright_browser_session.get_current_page()).screenshot(path='patchright_browser.png')
	imgcat(Path('patchright_browser.png').read_bytes(), height=max(terminal_height - 15, 40))
	await patchright_browser_session.close()

	# Brave Browser
	if Path('/Applications/Brave Browser.app/Contents/MacOS/Brave Browser').is_file():
		print('\n\nBRAVE BROWSER:')
		brave_browser_session = BrowserSession(
			executable_path='/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
			headless=False,
			disable_security=False,
			user_data_dir='~/.config/browseruse/profiles/brave',
			deterministic_rendering=False,
		)
		await brave_browser_session.start()
		await brave_browser_session.create_new_tab('https://abrahamjuliot.github.io/creepjs/')
		await asyncio.sleep(5)
		await (await brave_browser_session.get_current_page()).screenshot(path='brave_browser.png')
		imgcat(Path('brave_browser.png').read_bytes(), height=max(terminal_height - 15, 40))
		await brave_browser_session.close()

	if Path('/Applications/Brave Browser.app/Contents/MacOS/Brave Browser').is_file():
		print('\n\nBRAVE + PATCHRIGHT STEALTH BROWSER:')
		brave_patchright_browser_session = BrowserSession(
			executable_path='/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
			playwright=await async_patchright().start(),
			headless=False,
			disable_security=False,
			user_data_dir=None,
			deterministic_rendering=False,
		)
		await brave_patchright_browser_session.start()
		await brave_patchright_browser_session.create_new_tab('https://abrahamjuliot.github.io/creepjs/')
		await asyncio.sleep(5)
		await (await brave_patchright_browser_session.get_current_page()).screenshot(path='brave_patchright_browser.png')
		imgcat(Path('brave_patchright_browser.png').read_bytes(), height=max(terminal_height - 15, 40))

		input('Press [Enter] to close the browser...')
		await brave_patchright_browser_session.close()

	# print()
	# agent = Agent(
	# 	task="""
	#         Go to https://abrahamjuliot.github.io/creepjs/ and verify that the detection score is >50%.
	#     """,
	# 	llm=llm,
	# 	browser_session=browser_session,
	# )
	# await agent.run()

	# input('Press Enter to close the browser...')

	# agent = Agent(
	# 	task="""
	#         Go to https://bot-detector.rebrowser.net/ and verify that all the bot checks are passed.
	#     """,
	# 	llm=llm,
	# 	browser_session=browser_session,
	# )
	# await agent.run()
	# input('Press Enter to continue to the next test...')

	# agent = Agent(
	# 	task="""
	#         Go to https://www.webflow.com/ and verify that the page is not blocked by a bot check.
	#     """,
	# 	llm=llm,
	# 	browser_session=browser_session,
	# )
	# await agent.run()
	# input('Press Enter to continue to the next test...')

	# agent = Agent(
	# 	task="""
	#         Go to https://www.okta.com/ and verify that the page is not blocked by a bot check.
	#     """,
	# 	llm=llm,
	# 	browser_session=browser_session,
	# )
	# await agent.run()

	# agent = Agent(
	# 	task="""
	#         Go to https://nowsecure.nl/ check the "I'm not a robot" checkbox.
	#     """,
	# 	llm=llm,
	# 	browser_session=browser_session,
	# )
	# await agent.run()

	# input('Press Enter to close the browser...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\browser\using_cdp.py`

```py
"""
Simple demonstration of the CDP feature.

To test this locally, follow these steps:
1. Create a shortcut for the executable Chrome file.
2. Add the following argument to the shortcut:
   - On Windows: `--remote-debugging-port=9222`
3. Open a web browser and navigate to `http://localhost:9222/json/version` to verify that the Remote Debugging Protocol (CDP) is running.
4. Launch this example.

@dev You need to set the `GOOGLE_API_KEY` environment variable before proceeding.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

browser = Browser(
	config=BrowserConfig(
		headless=False,
		cdp_url='http://localhost:9222',
	)
)
controller = Controller()


async def main():
	task = 'In docs.google.com write my Papa a quick thank you for everything letter \n - Magnus'
	task += ' and save the document as pdf'
	model = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(str(api_key)))
	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser=browser,
	)

	await agent.run()
	await browser.close()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\action_filters.py`

```py
"""
Action filters (domains and page_filter) let you limit actions available to the Agent on a step-by-step/page-by-page basis.

@registry.action(..., domains=['*'], page_filter=lambda page: return True)
async def some_action(browser_session: BrowserSession):
    ...

This helps prevent the LLM from deciding to use an action that is not compatible with the current page.
It helps limit decision fatique by scoping actions only to pages where they make sense.
It also helps prevent mis-triggering stateful actions or actions that could break other programs or leak secrets.

For example:
    - only run on certain domains @registry.action(..., domains=['example.com', '*.example.com', 'example.co.*']) (supports globs, but no regex)
    - only fill in a password on a specific login page url
    - only run if this action has not run before on this page (e.g. by looking up the url in a file on disk)

During each step, the agent recalculates the actions available specifically for that page, and informs the LLM.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from playwright.async_api import Page

from browser_use.agent.service import Agent, Controller
from browser_use.browser import BrowserSession

# Initialize controller and registry
controller = Controller()
registry = controller.registry


# Action will only be available to Agent on Google domains because of the domain filter
@registry.action(description='Trigger disco mode', domains=['google.com', '*.google.com'])
async def disco_mode(browser_session: BrowserSession):
	page = await browser_session.get_current_page()
	await page.evaluate("""() => { 
        // define the wiggle animation
        document.styleSheets[0].insertRule('@keyframes wiggle { 0% { transform: rotate(0deg); } 50% { transform: rotate(10deg); } 100% { transform: rotate(0deg); } }');
        
        document.querySelectorAll("*").forEach(element => {
            element.style.animation = "wiggle 0.5s infinite";
        });
    }""")


# you can create a custom page filter function that determines if the action should be available for a given page
def is_login_page(page: Page) -> bool:
	return 'login' in page.url.lower() or 'signin' in page.url.lower()


# then use it in the action decorator to limit the action to only be available on pages where the filter returns True
@registry.action(description='Use the force, luke', page_filter=is_login_page)
async def use_the_force(browser_session: BrowserSession):
	# this will only ever run on pages that matched the filter
	page = await browser_session.get_current_page()
	assert is_login_page(page)

	await page.evaluate("""() => { document.querySelector('body').innerHTML = 'These are not the droids you are looking for';}""")


async def main():
	"""Main function to run the example"""
	browser_session = BrowserSession()
	await browser_session.start()
	llm = ChatOpenAI(model_name='gpt-4o')

	# Create the agent
	agent = Agent(  # disco mode will not be triggered on apple.com because the LLM won't be able to see that action available, it should work on Google.com though.
		task="""
            Go to apple.com and trigger disco mode (if dont know how to do that, then just move on).
            Then go to google.com and trigger disco mode.
            After that, go to the Google login page and Use the force, luke.
        """,
		llm=llm,
		browser_session=browser_session,
		controller=controller,
	)

	# Run the agent
	await agent.run(max_steps=10)

	# Cleanup
	await browser_session.stop()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\advanced_search.py`

```py
import asyncio
import http
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.profile import BrowserProfile

logger = logging.getLogger(__name__)


class Person(BaseModel):
	name: str
	email: str | None = None


class PersonList(BaseModel):
	people: list[Person]


SERP_API_KEY = os.getenv('SERPER_API_KEY')
if not SERP_API_KEY:
	raise ValueError('SERPER_API_KEY is not set')

controller = Controller(exclude_actions=['search_google'], output_model=PersonList)


@controller.registry.action('Search the web for a specific query')
async def search_web(query: str):
	# do a serp search for the query
	conn = http.client.HTTPSConnection('google.serper.dev')
	payload = json.dumps({'q': query})
	headers = {'X-API-KEY': SERP_API_KEY, 'Content-Type': 'application/json'}
	conn.request('POST', '/search', payload, headers)
	res = conn.getresponse()
	data = res.read()
	serp_data = json.loads(data.decode('utf-8'))

	# exclude searchParameters and credits
	serp_data = {k: v for k, v in serp_data.items() if k not in ['searchParameters', 'credits']}

	# print the original data
	logger.debug(json.dumps(serp_data, indent=2))

	# to string
	serp_data_str = json.dumps(serp_data)

	return ActionResult(extracted_content=serp_data_str, include_in_memory=False)


names = [
	'Ruedi Aebersold',
	'Bernd Bodenmiller',
	'Eugene Demler',
	'Erich Fischer',
	'Pietro Gambardella',
	'Matthias Huss',
	'Reto Knutti',
	'Maksym Kovalenko',
	'Antonio Lanzavecchia',
	'Maria Lukatskaya',
	'Jochen Markard',
	'Javier Pérez-Ramírez',
	'Federica Sallusto',
	'Gisbert Schneider',
	'Sonia I. Seneviratne',
	'Michael Siegrist',
	'Johan Six',
	'Tanja Stadler',
	'Shinichi Sunagawa',
	'Michael Bruce Zimmermann',
]


async def main():
	task = 'use search_web with "find email address of the following ETH professor:" for each of the following persons in a list of actions. Finally return the list with name and email if provided - do always 5 at once'
	task += '\n' + '\n'.join(names)
	model = ChatOpenAI(model='gpt-4o')
	browser_profile = BrowserProfile()
	agent = Agent(task=task, llm=model, controller=controller, browser_profile=browser_profile)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: PersonList = PersonList.model_validate_json(result)

		for person in parsed.people:
			print(f'{person.name} - {person.email}')
	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\clipboard.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import pyperclip
from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserProfile, BrowserSession

browser_profile = BrowserProfile(
	headless=False,
)
controller = Controller()


@controller.registry.action('Copy text to clipboard')
def copy_to_clipboard(text: str):
	pyperclip.copy(text)
	return ActionResult(extracted_content=text)


@controller.registry.action('Paste text from clipboard')
async def paste_from_clipboard(browser_session: BrowserSession):
	text = pyperclip.paste()
	# send text to browser
	page = await browser_session.get_current_page()
	await page.keyboard.type(text)

	return ActionResult(extracted_content=text)


async def main():
	task = 'Copy the text "Hello, world!" to the clipboard, then go to google.com and paste the text'
	model = ChatOpenAI(model='gpt-4o')
	browser_session = BrowserSession(browser_profile=browser_profile)
	await browser_session.start()
	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser_session=browser_session,
	)

	await agent.run()
	await browser_session.stop()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\custom_hooks_before_after_step.py`

```py
"""
Description: These Python modules are designed to capture detailed
browser usage datafor analysis, with both server and client
components working together to record and store the information.

Author: Carlos A. Planchón
https://github.com/carlosplanchon/

Adapt this code to your needs.

Feedback is appreciated!
"""

#####################
#                   #
#   --- UTILS ---   #
#                   #
#####################

import base64


def b64_to_png(b64_string: str, output_file):
	"""
	Convert a Base64-encoded string to a PNG file.

	:param b64_string: A string containing Base64-encoded data
	:param output_file: The path to the output PNG file
	"""
	with open(output_file, 'wb') as f:
		f.write(base64.b64decode(b64_string))


###################################################################
#                                                                 #
#   --- FASTAPI API TO RECORD AND SAVE Browser-Use ACTIVITY ---   #
#                                                                 #
###################################################################

# Save to api.py and run with `python api.py`

# ! pip install uvicorn
# ! pip install fastapi
# ! pip install prettyprinter

import json
from pathlib import Path

import prettyprinter
from fastapi import FastAPI, Request

prettyprinter.install_extras()

app = FastAPI()


@app.post('/post_agent_history_step')
async def post_agent_history_step(request: Request):
	data = await request.json()
	prettyprinter.cpprint(data)

	# Ensure the "recordings" folder exists using pathlib
	recordings_folder = Path('recordings')
	recordings_folder.mkdir(exist_ok=True)

	# Determine the next file number by examining existing .json files
	existing_numbers = []
	for item in recordings_folder.iterdir():
		if item.is_file() and item.suffix == '.json':
			try:
				file_num = int(item.stem)
				existing_numbers.append(file_num)
			except ValueError:
				# In case the file name isn't just a number
				...

	if existing_numbers:
		next_number = max(existing_numbers) + 1
	else:
		next_number = 1

	# Construct the file path
	file_path = recordings_folder / f'{next_number}.json'

	# Save the JSON data to the file
	with file_path.open('w') as f:
		json.dump(data, f, indent=2)

	return {'status': 'ok', 'message': f'Saved to {file_path}'}


if __name__ == '__main__':
	import uvicorn

	uvicorn.run(app, host='0.0.0.0', port=9000)


##############################################################
#                                                            #
#   --- CLIENT TO RECORD AND SAVE Browser-Use ACTIVITY ---   #
#                                                            #
##############################################################

"""
pyobjtojson:

A Python library to safely and recursively serialize any Python object
(including Pydantic models and dataclasses) into JSON-ready structures,
gracefully handling circular references.
"""

# ! pip install -U pyobjtojson
# ! pip install -U prettyprinter

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import requests
from langchain_openai import ChatOpenAI
from pyobjtojson import obj_to_json

from browser_use import Agent

# import prettyprinter
# prettyprinter.install_extras()


def send_agent_history_step(data):
	url = 'http://127.0.0.1:9000/post_agent_history_step'
	response = requests.post(url, json=data)
	return response.json()


async def record_activity(agent_obj):
	website_html = None
	website_screenshot = None
	urls_json_last_elem = None
	model_thoughts_last_elem = None
	model_outputs_json_last_elem = None
	model_actions_json_last_elem = None
	extracted_content_json_last_elem = None

	print('--- ON_STEP_START HOOK ---')
	website_html: str = await agent_obj.browser_context.get_page_html()
	website_screenshot: str = await agent_obj.browser_context.take_screenshot()

	print('--> History:')
	if hasattr(agent_obj, 'state'):
		history = agent_obj.state.history
	else:
		history = None

	model_thoughts = obj_to_json(obj=history.model_thoughts(), check_circular=False)

	# print("--- MODEL THOUGHTS ---")
	if len(model_thoughts) > 0:
		model_thoughts_last_elem = model_thoughts[-1]
		# prettyprinter.cpprint(model_thoughts_last_elem)

	# print("--- MODEL OUTPUT ACTION ---")
	model_outputs = agent_obj.state.history.model_outputs()
	model_outputs_json = obj_to_json(obj=model_outputs, check_circular=False)

	if len(model_outputs_json) > 0:
		model_outputs_json_last_elem = model_outputs_json[-1]
		# prettyprinter.cpprint(model_outputs_json_last_elem)

	# print("--- MODEL INTERACTED ELEM ---")
	model_actions = agent_obj.state.history.model_actions()
	model_actions_json = obj_to_json(obj=model_actions, check_circular=False)

	if len(model_actions_json) > 0:
		model_actions_json_last_elem = model_actions_json[-1]
		# prettyprinter.cpprint(model_actions_json_last_elem)

	# print("--- EXTRACTED CONTENT ---")
	extracted_content = agent_obj.state.history.extracted_content()
	extracted_content_json = obj_to_json(obj=extracted_content, check_circular=False)
	if len(extracted_content_json) > 0:
		extracted_content_json_last_elem = extracted_content_json[-1]
		# prettyprinter.cpprint(extracted_content_json_last_elem)

	# print("--- URLS ---")
	urls = agent_obj.state.history.urls()
	# prettyprinter.cpprint(urls)
	urls_json = obj_to_json(obj=urls, check_circular=False)

	if len(urls_json) > 0:
		urls_json_last_elem = urls_json[-1]
		# prettyprinter.cpprint(urls_json_last_elem)

	model_step_summary = {
		'website_html': website_html,
… (truncated)

```

---

## `browser-use-main\examples\custom-functions\file_upload.py`

```py
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import anyio
from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserProfile, BrowserSession

logger = logging.getLogger(__name__)

# Initialize browser and controller
browser_profile = BrowserProfile(
	headless=False,
	browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
)
controller = Controller()


@controller.action(
	'Upload file to interactive element with file path ',
)
async def upload_file(index: int, path: str, browser_session: BrowserSession, available_file_paths: list[str]):
	if path not in available_file_paths:
		return ActionResult(error=f'File path {path} is not available')

	if not os.path.exists(path):
		return ActionResult(error=f'File {path} does not exist')

	dom_el = await browser_session.get_dom_element_by_index(index)

	file_upload_dom_el = dom_el.get_file_upload_element()

	if file_upload_dom_el is None:
		msg = f'No file upload element found at index {index}'
		logger.info(msg)
		return ActionResult(error=msg)

	file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)

	if file_upload_el is None:
		msg = f'No file upload element found at index {index}'
		logger.info(msg)
		return ActionResult(error=msg)

	try:
		await file_upload_el.set_input_files(path)
		msg = f'Successfully uploaded file to index {index}'
		logger.info(msg)
		return ActionResult(extracted_content=msg, include_in_memory=True)
	except Exception as e:
		msg = f'Failed to upload file to index {index}: {str(e)}'
		logger.info(msg)
		return ActionResult(error=msg)


@controller.action('Read the file content of a file given a path')
async def read_file(path: str, available_file_paths: list[str]):
	if path not in available_file_paths:
		return ActionResult(error=f'File path {path} is not available')

	async with await anyio.open_file(path, 'r') as f:
		content = await f.read()
	msg = f'File content: {content}'
	logger.info(msg)
	return ActionResult(extracted_content=msg, include_in_memory=True)


def create_file(file_type: str = 'txt'):
	with open(f'tmp.{file_type}', 'w') as f:
		f.write('test')
	file_path = Path.cwd() / f'tmp.{file_type}'
	logger.info(f'Created file: {file_path}')
	return str(file_path)


async def main():
	task = 'Go to https://kzmpmkh2zfk1ojnpxfn1.lite.vusercontent.net/ and - read the file content and upload them to fields'

	available_file_paths = [create_file('txt'), create_file('pdf'), create_file('csv')]

	model = ChatOpenAI(model='gpt-4o')
	browser_session = BrowserSession(browser_profile=browser_profile)
	await browser_session.start()
	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser_session=browser_session,
		available_file_paths=available_file_paths,
	)

	await agent.run()

	await browser_session.stop()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\hover_element.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserProfile, BrowserSession


class HoverAction(BaseModel):
	index: int | None = None
	xpath: str | None = None
	selector: str | None = None


browser_profile = BrowserProfile(
	headless=False,
)
controller = Controller()


@controller.registry.action(
	'Hover over an element',
	param_model=HoverAction,  # Define this model with at least "index: int" field
)
async def hover_element(params: HoverAction, browser_session: BrowserSession):
	"""
	Hovers over the element specified by its index from the cached selector map or by XPath.
	"""
	if params.xpath:
		# Use XPath to locate the element
		element_handle = await browser_session.get_locate_element_by_xpath(params.xpath)
		if element_handle is None:
			raise Exception(f'Failed to locate element with XPath {params.xpath}')
	elif params.selector:
		# Use CSS selector to locate the element
		element_handle = await browser_session.get_locate_element_by_css_selector(params.selector)
		if element_handle is None:
			raise Exception(f'Failed to locate element with CSS Selector {params.selector}')
	elif params.index is not None:
		# Use index to locate the element
		selector_map = await browser_session.get_selector_map()
		if params.index not in selector_map:
			raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')
		element_node = selector_map[params.index]
		element_handle = await browser_session.get_locate_element(element_node)
		if element_handle is None:
			raise Exception(f'Failed to locate element with index {params.index}')
	else:
		raise Exception('Either index or xpath must be provided')

	try:
		await element_handle.hover()
		msg = (
			f'🖱️ Hovered over element at index {params.index}'
			if params.index is not None
			else f'🖱️ Hovered over element with XPath {params.xpath}'
		)
		return ActionResult(extracted_content=msg, include_in_memory=True)
	except Exception as e:
		err_msg = f'❌ Failed to hover over element: {str(e)}'
		raise Exception(err_msg)


async def main():
	task = 'Open https://testpages.eviltester.com/styled/csspseudo/css-hover.html and hover the element with the css selector #hoverdivpara, then click on "Can you click me?"'
	# task = 'Open https://testpages.eviltester.com/styled/csspseudo/css-hover.html and hover the element with the xpath //*[@id="hoverdivpara"], then click on "Can you click me?"'
	model = ChatOpenAI(model='gpt-4o')
	browser_session = BrowserSession(browser_profile=browser_profile)
	await browser_session.start()
	agent = Agent(
		task=task,
		llm=model,
		controller=controller,
		browser_session=browser_session,
	)

	await agent.run()
	await browser_session.stop()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\notification.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import ActionResult, Agent, Controller

controller = Controller()


@controller.registry.action('Done with task ')
async def done(text: str):
	import yagmail

	# To send emails use
	# STEP 1: go to https://support.google.com/accounts/answer/185833
	# STEP 2: Create an app password (you can't use here your normal gmail password)
	# STEP 3: Use the app password in the code below for the password
	yag = yagmail.SMTP('your_email@gmail.com', 'your_app_password')
	yag.send(
		to='recipient@example.com',
		subject='Test Email',
		contents=f'result\n: {text}',
	)

	return ActionResult(is_done=True, extracted_content='Email sent!')


async def main():
	task = 'go to brower-use.com and then done'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\onepassword_2fa.py`

```py
import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from onepassword.client import Client  # pip install onepassword-sdk

from browser_use import ActionResult, Agent, Controller

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OP_SERVICE_ACCOUNT_TOKEN = os.getenv('OP_SERVICE_ACCOUNT_TOKEN')
OP_ITEM_ID = os.getenv('OP_ITEM_ID')  # Go to 1Password, right click on the item, click "Copy Secret Reference"


controller = Controller()


@controller.registry.action('Get 2FA code from 1Password for Google Account', domains=['*.google.com', 'google.com'])
async def get_1password_2fa() -> ActionResult:
	"""
	Custom action to retrieve 2FA/MFA code from 1Password using onepassword.client SDK.
	"""
	client = await Client.authenticate(
		# setup instructions: https://github.com/1Password/onepassword-sdk-python/#-get-started
		auth=OP_SERVICE_ACCOUNT_TOKEN,
		integration_name='Browser-Use',
		integration_version='v1.0.0',
	)

	mfa_code = await client.secrets.resolve(f'op://Private/{OP_ITEM_ID}/One-time passcode')

	return ActionResult(extracted_content=mfa_code)


async def main():
	# Example task using the 1Password 2FA action
	task = 'Go to account.google.com, enter username and password, then if prompted for 2FA code, get 2FA code from 1Password for and enter it'

	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	result = await agent.run()
	print(f'Task completed with result: {result}')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\custom-functions\save_to_file_hugging_face.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

# Initialize controller first
controller = Controller()


class Model(BaseModel):
	title: str
	url: str
	likes: int
	license: str


class Models(BaseModel):
	models: list[Model]


@controller.action('Save models', param_model=Models)
def save_models(params: Models):
	with open('models.txt', 'a') as f:
		for model in params.models:
			f.write(f'{model.title} ({model.url}): {model.likes} likes, {model.license}\n')


# video: https://preview.screen.studio/share/EtOhIk0P
async def main():
	task = 'Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.'

	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\click_fallback_options.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from aiohttp import web  # make sure to install aiohttp: pip install aiohttp
from langchain_openai import ChatOpenAI

# from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, Controller

# Define a simple HTML page
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Custom Select Div</title>
  <style>
    .custom-select {
      position: relative;
      width: 200px;
      font-family: Arial, sans-serif;
      margin-bottom: 20px;
    }

    .select-display {
      padding: 10px;
      border: 1px solid #ccc;
      background-color: #fff;
      cursor: pointer;
    }

    .select-options {
      position: absolute;
      top: 100%;
      left: 0;
      right: 0;
      border: 1px solid #ccc;
      border-top: none;
      background-color: #fff;
      display: none;
      max-height: 150px;
      overflow-y: auto;
      z-index: 100;
    }

    .select-option {
      padding: 10px;
      cursor: pointer;
    }

    .select-option:hover {
      background-color: #f0f0f0;
    }
  </style>
</head>
<body>
  <div class="custom-select">
    <div class="select-display">Select a fruit</div>
    <div class="select-options">
      <div class="select-option" data-value="option1">Apples</div>
      <div class="select-option" data-value="option2">Oranges</div>
      <div class="select-option" data-value="option3">Pineapples</div>
    </div>
  </div>

  <div class="custom-select">
    <div class="select-display">Select a fruit</div>
    <div class="select-options">
      <div class="select-option" data-value="option1">Apples</div>
      <div class="select-option" data-value="option2">Oranges</div>
      <div class="select-option" data-value="option3">Pineapples</div>
    </div>
  </div>
  
  <div class="custom-select">
    <div class="select-display">Select a fruit</div>
    <div class="select-options">
      <div class="select-option" data-value="option1">Apples</div>
      <div class="select-option" data-value="option2">Oranges</div>
      <div class="select-option" data-value="option3">Pineapples</div>
    </div>
  </div>
  
  <div class="custom-select">
    <div class="select-display">Select a fruit</div>
    <div class="select-options">
      <div class="select-option" data-value="option1">Apples</div>
      <div class="select-option" data-value="option2">Oranges</div>
      <div class="select-option" data-value="option3">Pineapples</div>
    </div>
  </div>

  <label for="cars">Choose a car:</label>
  <select name="cars" id="cars">
    <option value="volvo">Volvo</option>
    <option value="bmw">BMW</option>
    <option value="mercedes">Mercedes</option>
    <option value="audi">Audi</option>
  </select>

  <button onclick="alert('I told you!')">Don't click me</button>

  <script>
    document.querySelectorAll('.custom-select').forEach(customSelect => {
      const selectDisplay = customSelect.querySelector('.select-display');
      const selectOptions = customSelect.querySelector('.select-options');
      const options = customSelect.querySelectorAll('.select-option');

      selectDisplay.addEventListener('click', (e) => {
        // Close all other dropdowns
        document.querySelectorAll('.select-options').forEach(opt => {
          if (opt !== selectOptions) opt.style.display = 'none';
        });

        // Toggle current dropdown
        const isVisible = selectOptions.style.display === 'block';
        selectOptions.style.display = isVisible ? 'none' : 'block';

        e.stopPropagation();
      });

      options.forEach(option => {
        option.addEventListener('click', () => {
          selectDisplay.textContent = option.textContent;
          selectDisplay.dataset.value = option.getAttribute('data-value');
          selectOptions.style.display = 'none';
        });
      });
    });

    // Close all dropdowns if clicking outside
    document.addEventListener('click', () => {
      document.querySelectorAll('.select-options').forEach(opt => {
        opt.style.display = 'none';
      });
    });
  </script>
</body>
</html>

"""


# aiohttp request handler to serve the HTML content
async def handle_root(request):
	return web.Response(text=HTML_CONTENT, content_type='text/html')


# Function to run the HTTP server
async def run_http_server():
	app = web.Application()
	app.router.add_get('/', handle_root)
	runner = web.AppRunner(app)
	await runner.setup()
	site = web.TCPSite(runner, 'localhost', 8000)
	await site.start()
	print('HTTP server running on http://localhost:8000')
	# Keep the server running indefinitely.
	await asyncio.Event().wait()


# Your agent tasks and other logic
controller = Controller()


async def main():
	# Start the HTTP server in the background.
	server_task = asyncio.create_task(run_http_server())

	# Example tasks for the agent.
	xpath_task = 'Open http://localhost:8000/, click element with the xpath "/html/body/div/div[1]" and then click on Oranges'
	css_selector_task = 'Open http://localhost:8000/, click element with the selector div.select-display and then click on apples'
	text_task = 'Open http://localhost:8000/, click the third element with the text "Select a fruit" and then click on Apples, then click the second element with the text "Select a fruit" and then click on Oranges'
	select_task = 'Open http://localhost:8000/, choose the car BMW'
	button_task = 'Open http://localhost:8000/, click on the button'

	llm = ChatOpenAI(model='gpt-4o')
	# llm = ChatGoogleGenerativeAI(
	#     model="gemini-2.0-flash-lite",
	# )

	# Run different agent tasks.
	for task in [xpath_task, css_selector_task, text_task, select_task, button_task]:
		agent = Agent(
			task=task,
			llm=llm,
			controller=controller,
		)
		await agent.run()

	# Wait for user input before shutting down.
	input('Press Enter to close...')
… (truncated)

```

---

## `browser-use-main\examples\features\cross_origin_iframes.py`

```py
"""
Example of how it supports cross-origin iframes.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')


browser = Browser(
	config=BrowserConfig(
		browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	)
)
controller = Controller()


async def main():
	agent = Agent(
		task='Click "Go cross-site (simple page)" button on https://csreis.github.io/tests/cross-site-iframe.html then tell me the text within',
		llm=ChatOpenAI(model='gpt-4o', temperature=0.0),
		controller=controller,
		browser=browser,
	)

	await agent.run()
	await browser.close()

	input('Press Enter to close...')


if __name__ == '__main__':
	try:
		asyncio.run(main())
	except Exception as e:
		print(e)

```

---

## `browser-use-main\examples\features\custom_output.py`

```py
"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent, Controller


class Post(BaseModel):
	post_title: str
	post_url: str
	num_comments: int
	hours_since_post: int


class Posts(BaseModel):
	posts: list[Post]


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

---

## `browser-use-main\examples\features\custom_system_prompt.py`

```py
import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

extend_system_message = (
	'REMEMBER the most important RULE: ALWAYS open first a new tab and go first to url wikipedia.com no matter the task!!!'
)

# or use override_system_message to completely override the system prompt


async def main():
	task = "do google search to find images of Elon Musk's wife"
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, extend_system_message=extend_system_message)

	print(
		json.dumps(
			agent.message_manager.system_prompt.model_dump(exclude_unset=True),
			indent=4,
		)
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\custom_user_agent.py`

```py
import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.controller.service import Controller


def get_llm(provider: str):
	if provider == 'anthropic':
		return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
	elif provider == 'openai':
		return ChatOpenAI(model='gpt-4o', temperature=0.0)

	else:
		raise ValueError(f'Unsupported provider: {provider}')


# NOTE: This example is to find your current user agent string to use it in the browser_context
task = 'go to https://whatismyuseragent.com and find the current user agent string '


controller = Controller()


parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, help='The query to process', default=task)
parser.add_argument(
	'--provider',
	type=str,
	choices=['openai', 'anthropic'],
	default='openai',
	help='The model provider to use (default: openai)',
)

args = parser.parse_args()

llm = get_llm(args.provider)

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		user_agent='foobarfoo',
		user_data_dir='~/.config/browseruse/profiles/default',
	)
)

agent = Agent(
	task=args.query,
	llm=llm,
	controller=controller,
	browser_session=browser_session,
	use_vision=True,
	max_actions_per_step=1,
)


async def main():
	await agent.run(max_steps=25)

	input('Press Enter to close the browser...')
	await browser_session.close()


asyncio.run(main())

```

---

## `browser-use-main\examples\features\download_file.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		save_downloads_path=os.path.join(os.path.expanduser('~'), 'downloads'),
		user_data_dir='~/.config/browseruse/profiles/default',
	)
)


async def run_download():
	agent = Agent(
		task=('Go to "https://file-examples.com/" and download the smallest doc file.'),
		llm=llm,
		max_actions_per_step=8,
		use_vision=True,
		browser_session=browser_session,
	)
	await agent.run(max_steps=25)
	await browser_session.close()


if __name__ == '__main__':
	asyncio.run(run_download())

```

---

## `browser-use-main\examples\features\drag_drop.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))


task_1 = """
Navigate to: https://sortablejs.github.io/Sortable/. 
Then scroll down to the first examplw with title "Simple list example". 
Drag the element with name "item 1" to below the element with name "item 3".
"""


task_2 = """
Navigate to: https://excalidraw.com/.
Click on the pencil icon (with index 40).
Then draw a triangle in the canvas.
Draw the triangle starting from coordinate (400,400).
You can use the drag and drop action to draw the triangle.
"""


async def run_search():
	agent = Agent(
		task=task_1,
		llm=llm,
		max_actions_per_step=1,
		use_vision=True,
	)

	await agent.run(max_steps=25)


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\features\follow_up_tasks.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser import BrowserProfile, BrowserSession

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)
# Get your chrome path
browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		keep_alive=True,
		user_data_dir='~/.config/browseruse/profiles/default',
	),
)

controller = Controller()


task = 'Find the founders of browser-use and draft them a short personalized message'

agent = Agent(task=task, llm=llm, controller=controller, browser_session=browser_session)


async def main():
	await agent.run()

	# new_task = input('Type in a new task: ')
	new_task = 'Find an image of the founders'

	agent.add_new_task(new_task)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\initial_actions.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o')

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


async def main():
	await agent.run(max_steps=10)


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\multi-tab_handling.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

# video: https://preview.screen.studio/share/clenCmS6
llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='open 3 tabs with elon musk, trump, and steve jobs, then go back to the first and stop',
	llm=llm,
)


async def main():
	await agent.run()


asyncio.run(main())

```

---

## `browser-use-main\examples\features\multiple_agents_same_browser.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser


# Video: https://preview.screen.studio/share/8Elaq9sm
async def main():
	# Persist the browser state across agents

	browser = Browser()
	async with await browser.new_context() as context:
		model = ChatOpenAI(model='gpt-4o')
		current_agent = None

		async def get_input():
			return await asyncio.get_event_loop().run_in_executor(
				None, lambda: input('Enter task (p: pause current agent, r: resume, b: break): ')
			)

		while True:
			task = await get_input()

			if task.lower() == 'p':
				# Pause the current agent if one exists
				if current_agent:
					current_agent.pause()
				continue
			elif task.lower() == 'r':
				# Resume the current agent if one exists
				if current_agent:
					current_agent.resume()
				continue
			elif task.lower() == 'b':
				# Break the current agent's execution if one exists
				if current_agent:
					current_agent.stop()
					current_agent = None
				continue

			# If there's a current agent running, pause it before starting new one
			if current_agent:
				current_agent.pause()

			# Create and run new agent with the task
			current_agent = Agent(
				task=task,
				llm=model,
				browser_context=context,
			)

			# Run the agent asynchronously without blocking
			asyncio.create_task(current_agent.run())


asyncio.run(main())

# Now aad the cheapest to the cart

```

---

## `browser-use-main\examples\features\outsource_state.py`

```py
"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import anyio
from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.agent.views import AgentState
from browser_use.browser.browser import Browser, BrowserConfig


async def main():
	task = 'Go to hackernews show hn and give me the first  5 posts'

	browser = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)

	browser_context = await browser.new_context()

	agent_state = AgentState()

	for i in range(10):
		agent = Agent(
			task=task,
			llm=ChatOpenAI(model='gpt-4o'),
			browser=browser,
			browser_context=browser_context,
			injected_agent_state=agent_state,
			page_extraction_llm=ChatOpenAI(model='gpt-4o-mini'),
		)

		done, valid = await agent.take_step()
		print(f'Step {i}: Done: {done}, Valid: {valid}')

		if done and valid:
			break

		agent_state.history.history = []

		# Save state to file
		async with await anyio.open_file('agent_state.json', 'w') as f:
			serialized = agent_state.model_dump_json(exclude={'history'})
			await f.write(serialized)

		# Load state back from file
		async with await anyio.open_file('agent_state.json', 'r') as f:
			loaded_json = await f.read()
			agent_state = AgentState.model_validate_json(loaded_json)

		break


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\parallel_agents.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use.agent.service import Agent
from browser_use.browser import BrowserProfile, BrowserSession

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		disable_security=True,
		headless=False,
		save_recording_path='./tmp/recordings',
		user_data_dir='~/.config/browseruse/profiles/default',
	)
)
llm = ChatOpenAI(model='gpt-4o')


async def main():
	agents = [
		Agent(task=task, llm=llm, browser_session=browser_session)
		for task in [
			'Search Google for weather in Tokyo',
			'Check Reddit front page title',
			'Look up Bitcoin price on Coinbase',
			'Find NASA image of the day',
			# 'Check top story on CNN',
			# 'Search latest SpaceX launch date',
			# 'Look up population of Paris',
			# 'Find current time in Sydney',
			# 'Check who won last Super Bowl',
			# 'Search trending topics on Twitter',
		]
	]

	await asyncio.gather(*[agent.run() for agent in agents])

	agentX = Agent(
		task='Go to apple.com and return the title of the page',
		llm=llm,
		browser_session=browser_session,
	)
	await agentX.run()

	await browser_session.close()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\pause_agent.py`

```py
import asyncio
import os
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent


class AgentController:
	def __init__(self):
		llm = ChatOpenAI(model='gpt-4o')
		self.agent = Agent(
			task='open in one action https://www.google.com, https://www.wikipedia.org, https://www.youtube.com, https://www.github.com, https://amazon.com',
			llm=llm,
		)
		self.running = False

	async def run_agent(self):
		"""Run the agent"""
		self.running = True
		await self.agent.run()

	def start(self):
		"""Start the agent in a separate thread"""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		loop.run_until_complete(self.run_agent())

	def pause(self):
		"""Pause the agent"""
		self.agent.pause()

	def resume(self):
		"""Resume the agent"""
		self.agent.resume()

	def stop(self):
		"""Stop the agent"""
		self.agent.stop()
		self.running = False


def print_menu():
	print('\nAgent Control Menu:')
	print('1. Start')
	print('2. Pause')
	print('3. Resume')
	print('4. Stop')
	print('5. Exit')


async def main():
	controller = AgentController()
	agent_thread = None

	while True:
		print_menu()
		try:
			choice = input('Enter your choice (1-5): ')
		except KeyboardInterrupt:
			choice = '5'

		if choice == '1' and not agent_thread:
			print('Starting agent...')
			agent_thread = threading.Thread(target=controller.start)
			agent_thread.start()

		elif choice == '2':
			print('Pausing agent...')
			controller.pause()

		elif choice == '3':
			print('Resuming agent...')
			controller.resume()

		elif choice == '4':
			print('Stopping agent...')
			controller.stop()
			if agent_thread:
				agent_thread.join()
				agent_thread = None

		elif choice == '5':
			print('Exiting...')
			if controller.running:
				controller.stop()
				if agent_thread:
					agent_thread.join()
			break

		await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\planner.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
planner_llm = ChatOpenAI(
	model='o3-mini',
)
task = 'your task'


agent = Agent(task=task, llm=llm, planner_llm=planner_llm, use_vision_for_planner=False, planner_interval=1)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\playwright_script_generation.py`

```py
import asyncio
import os
import sys
from pathlib import Path

# Ensure the project root is in the Python path if running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser, BrowserConfig

# Define the task for the agent
TASK_DESCRIPTION = """
1. Go to amazon.com
2. Search for 'i7 14700k'
4. If there is an 'Add to Cart' button, open the product page and then click add to cart.
5. the open the shopping cart page /cart button/ go to cart button.
6. Scroll down to the bottom of the cart page.
7. Scroll up to the top of the cart page.
8. Finish the task.
"""

# Define the path where the Playwright script will be saved
SCRIPT_DIR = Path('./playwright_scripts')
SCRIPT_PATH = SCRIPT_DIR / 'playwright_amazon_cart_script.py'


# Helper function to stream output from the subprocess
async def stream_output(stream, prefix):
	if stream is None:
		print(f'{prefix}: (No stream available)')
		return
	while True:
		line = await stream.readline()
		if not line:
			break
		print(f'{prefix}: {line.decode().rstrip()}', flush=True)


async def main():
	# Initialize the language model
	llm = ChatOpenAI(model='gpt-4.1', temperature=0.0)

	# Configure the browser
	# Use headless=False if you want to watch the agent visually
	browser_config = BrowserConfig(headless=False)
	browser = Browser(config=browser_config)

	# Configure the agent
	# The 'save_playwright_script_path' argument tells the agent where to save the script
	agent = Agent(
		task=TASK_DESCRIPTION,
		llm=llm,
		browser=browser,
		save_playwright_script_path=str(SCRIPT_PATH),  # Pass the path as a string
	)

	print('Running the agent to generate the Playwright script...')
	history = None  # Initialize history to None
	try:
		history = await agent.run()
		print('Agent finished running.')

		if history and history.is_successful():
			print(f'Agent completed the task successfully. Final result: {history.final_result()}')
		elif history:
			print('Agent finished, but the task might not be fully successful.')
			if history.has_errors():
				print(f'Errors encountered: {history.errors()}')
		else:
			print('Agent run did not return a history object.')

	except Exception as e:
		print(f'An error occurred during the agent run: {e}')
		# Ensure browser is closed even if agent run fails
		if browser:
			await browser.close()
		return  # Exit if agent failed

	# --- Execute the Generated Playwright Script ---
	print(f'\nChecking if Playwright script was generated at: {SCRIPT_PATH}')
	if SCRIPT_PATH.exists():
		print('Playwright script found. Attempting to execute...')
		try:
			# Ensure the script directory exists before running
			SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

			# Execute the generated script using asyncio.create_subprocess_exec
			process = await asyncio.create_subprocess_exec(
				sys.executable,
				str(SCRIPT_PATH),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
				cwd=Path.cwd(),  # Run from the current working directory
			)

			print('\n--- Playwright Script Execution ---')
			# Create tasks to stream stdout and stderr concurrently
			stdout_task = asyncio.create_task(stream_output(process.stdout, 'stdout'))
			stderr_task = asyncio.create_task(stream_output(process.stderr, 'stderr'))

			# Wait for both stream tasks and the process to finish
			await asyncio.gather(stdout_task, stderr_task)
			returncode = await process.wait()
			print('-------------------------------------')

			if returncode == 0:
				print('\n✅ Playwright script executed successfully!')
			else:
				print(f'\n⚠️ Playwright script finished with exit code {returncode}.')

		except Exception as e:
			print(f'\n❌ An error occurred while executing the Playwright script: {e}')
	else:
		print(f'\n❌ Playwright script not found at {SCRIPT_PATH}. Generation might have failed.')

	# Close the browser used by the agent (if not already closed by agent.run error handling)
	# Note: The generated script manages its own browser instance.
	if browser:
		await browser.close()
		print("Agent's browser closed.")


if __name__ == '__main__':
	# Ensure the script directory is clean before running (optional)
	if SCRIPT_PATH.exists():
		SCRIPT_PATH.unlink()
		print(f'Removed existing script: {SCRIPT_PATH}')

	# Run the main async function
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\restrict_urls.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
task = (
	"go to google.com and search for openai.com and click on the first link then extract content and scroll down - what's there?"
)

allowed_domains = ['google.com']

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		allowed_domains=allowed_domains,
		user_data_dir='~/.config/browseruse/profiles/default',
	),
)

agent = Agent(
	task=task,
	llm=llm,
	browser_session=browser_session,
)


async def main():
	await agent.run(max_steps=25)

	input('Press Enter to close the browser...')
	await browser_session.close()


asyncio.run(main())

```

---

## `browser-use-main\examples\features\result_processing.py`

```py
import asyncio
import os
import sys
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser import BrowserProfile, BrowserSession

llm = ChatOpenAI(model='gpt-4o')


async def main():
	async with BrowserSession(
		browser_profile=BrowserProfile(
			headless=False,
			disable_security=True,
			trace_path='./tmp/result_processing',
			no_viewport=False,
			window_width=1280,
			window_height=1000,
			user_data_dir='~/.config/browseruse/profiles/default',
		)
	) as browser_session:
		agent = Agent(
			task="go to google.com and type 'OpenAI' click search and give me the first url",
			llm=llm,
			browser_session=browser_session,
		)
		history: AgentHistoryList = await agent.run(max_steps=3)

		print('Final Result:')
		pprint(history.final_result(), indent=4)

		print('\nErrors:')
		pprint(history.errors(), indent=4)

		# e.g. xPaths the model clicked on
		print('\nModel Outputs:')
		pprint(history.model_actions(), indent=4)

		print('\nThoughts:')
		pprint(history.model_thoughts(), indent=4)


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\save_trace.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use.agent.service import Agent
from browser_use.browser import BrowserProfile, BrowserSession

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)


async def main():
	browser_session = BrowserSession(
		browser_profile=BrowserProfile(
			trace_path='./tmp/traces/',
			user_data_dir='~/.config/browseruse/profiles/default',
		)
	)

	async with browser_session:
		agent = Agent(
			task='Go to hackernews, then go to apple.com and return all titles of open tabs',
			llm=llm,
			browser_session=browser_session,
		)
		await agent.run()


asyncio.run(main())

```

---

## `browser-use-main\examples\features\sensitive_data.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)
# the model will see x_name and x_password, but never the actual values.
sensitive_data = {'x_name': 'my_x_name', 'x_password': 'my_x_password'}
task = 'go to x.com and login with x_name and x_password then find interesting posts and like them'

agent = Agent(task=task, llm=llm, sensitive_data=sensitive_data)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\small_model_for_extraction.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)
small_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)
task = 'Find the founders of browser-use in ycombinator, extract all links and open the links one by one'
agent = Agent(task=task, llm=llm, page_extraction_llm=small_llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\task_with_memory.py`

```py
import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import anyio
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent, Browser, BrowserConfig, Controller

links = [
	'https://docs.mem0.ai/components/llms/models/litellm',
	'https://docs.mem0.ai/components/llms/models/mistral_AI',
	'https://docs.mem0.ai/components/llms/models/ollama',
	'https://docs.mem0.ai/components/llms/models/openai',
	'https://docs.mem0.ai/components/llms/models/together',
	'https://docs.mem0.ai/components/llms/models/xAI',
	'https://docs.mem0.ai/components/llms/overview',
	'https://docs.mem0.ai/components/vectordbs/config',
	'https://docs.mem0.ai/components/vectordbs/dbs/azure_ai_search',
	'https://docs.mem0.ai/components/vectordbs/dbs/chroma',
	'https://docs.mem0.ai/components/vectordbs/dbs/elasticsearch',
	'https://docs.mem0.ai/components/vectordbs/dbs/milvus',
	'https://docs.mem0.ai/components/vectordbs/dbs/opensearch',
	'https://docs.mem0.ai/components/vectordbs/dbs/pgvector',
	'https://docs.mem0.ai/components/vectordbs/dbs/pinecone',
	'https://docs.mem0.ai/components/vectordbs/dbs/qdrant',
	'https://docs.mem0.ai/components/vectordbs/dbs/redis',
	'https://docs.mem0.ai/components/vectordbs/dbs/supabase',
	'https://docs.mem0.ai/components/vectordbs/dbs/vertex_ai_vector_search',
	'https://docs.mem0.ai/components/vectordbs/dbs/weaviate',
	'https://docs.mem0.ai/components/vectordbs/overview',
	'https://docs.mem0.ai/contributing/development',
	'https://docs.mem0.ai/contributing/documentation',
	'https://docs.mem0.ai/core-concepts/memory-operations',
	'https://docs.mem0.ai/core-concepts/memory-types',
]


class Link(BaseModel):
	url: str
	title: str
	summary: str


class Links(BaseModel):
	links: list[Link]


initial_actions = [
	{'open_tab': {'url': 'https://docs.mem0.ai/'}},
]
controller = Controller(output_model=Links)
task_description = f"""
Visit all the links provided in {links} and summarize the content of the page with url and title. There are {len(links)} links to visit. Make sure to visit all the links. Return a json with the following format: [{{url: <url>, title: <title>, summary: <summary>}}].

Guidelines:
1. Strictly stay on the domain https://docs.mem0.ai
2. Do not visit any other websites.
3. Ignore the links that are hashed (#) or javascript (:), or mailto, or tel, or other protocols
4. Don't visit any other url other than the ones provided above.
5. Capture the unique urls which are not already visited.
6. If you visit any page that doesn't have host name docs.mem0.ai, then do not visit it and come back to the page with host name docs.mem0.ai.
"""


async def main(max_steps=500):
	config = BrowserConfig(headless=True)
	browser = Browser(config=config)

	agent = Agent(
		task=task_description,
		llm=ChatOpenAI(model='gpt-4o-mini'),
		controller=controller,
		initial_actions=initial_actions,
		enable_memory=True,
		browser=browser,
	)
	history = await agent.run(max_steps=max_steps)
	result = history.final_result()
	parsed_result = []
	if result:
		parsed: Links = Links.model_validate_json(result)
		print(f'Total parsed links: {len(parsed.links)}')
		for link in parsed.links:
			parsed_result.append({'title': link.title, 'url': link.url, 'summary': link.summary})
	else:
		print('No result')

	async with await anyio.open_file('result.json', 'w+') as f:
		await f.write(json.dumps(parsed_result, indent=4))


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\features\validate_output.py`

```py
"""
Demonstrate output validator.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import ActionResult, Agent, Controller

controller = Controller()


class DoneResult(BaseModel):
	title: str
	comments: str
	hours_since_start: int


# we overwrite done() in this example to demonstrate the validator
@controller.registry.action('Done with task', param_model=DoneResult)
async def done(params: DoneResult):
	result = ActionResult(is_done=True, extracted_content=params.model_dump_json())
	print(result)
	# NOTE: this is clearly wrong - to demonstrate the validator
	return 'blablabla'


async def main():
	task = 'Go to hackernews hn and give me the top 1 post'
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller, validate_output=True)
	# NOTE: this should fail to demonstrate the validator
	await agent.run(max_steps=5)


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\integrations\discord\discord_api.py`

```py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv

load_dotenv()

import discord
from discord.ext import commands
from langchain_core.language_models.chat_models import BaseChatModel

from browser_use import BrowserConfig
from browser_use.agent.service import Agent, Browser


class DiscordBot(commands.Bot):
	"""Discord bot implementation for Browser-Use tasks.

	This bot allows users to run browser automation tasks through Discord messages.
	Processes tasks asynchronously and sends the result back to the user in response to the message.
	Messages must start with the configured prefix (default: "$bu") followed by the task description.

	Args:
	    llm (BaseChatModel): Language model instance to use for task processing
	    prefix (str, optional): Command prefix for triggering browser tasks. Defaults to "$bu"
	    ack (bool, optional): Whether to acknowledge task receipt with a message. Defaults to False
	    browser_config (BrowserConfig, optional): Browser configuration settings.
	        Defaults to headless mode

	Usage:
	    ```python
	    from langchain_openai import ChatOpenAI

	    llm = ChatOpenAI()
	    bot = DiscordBot(llm=llm, prefix='$bu', ack=True)
	    bot.run('YOUR_DISCORD_TOKEN')
	    ```

	Discord Usage:
	    Send messages starting with the prefix:
	    "$bu search for python tutorials"
	"""

	def __init__(
		self,
		llm: BaseChatModel,
		prefix: str = '$bu',
		ack: bool = False,
		browser_config: BrowserConfig = BrowserConfig(headless=True),
	):
		self.llm = llm
		self.prefix = prefix.strip()
		self.ack = ack
		self.browser_config = browser_config

		# Define intents.
		intents = discord.Intents.default()
		intents.message_content = True  # Enable message content intent
		intents.members = True  # Enable members intent for user info

		# Initialize the bot with a command prefix and intents.
		super().__init__(command_prefix='!', intents=intents)  # You may not need prefix, just here for flexibility

		# self.tree = app_commands.CommandTree(self) # Initialize command tree for slash commands.

	async def on_ready(self):
		"""Called when the bot is ready."""
		try:
			print(f'We have logged in as {self.user}')
			cmds = await self.tree.sync()  # Sync the command tree with discord

		except Exception as e:
			print(f'Error during bot startup: {e}')

	async def on_message(self, message):
		"""Called when a message is received."""
		try:
			if message.author == self.user:  # Ignore the bot's messages
				return
			if message.content.strip().startswith(f'{self.prefix} '):
				if self.ack:
					try:
						await message.reply(
							'Starting browser use task...',
							mention_author=True,  # Don't ping the user
						)
					except Exception as e:
						print(f'Error sending start message: {e}')

				try:
					agent_message = await self.run_agent(message.content.replace(f'{self.prefix} ', '').strip())
					await message.channel.send(content=f'{agent_message}', reference=message, mention_author=True)
				except Exception as e:
					await message.channel.send(
						content=f'Error during task execution: {str(e)}',
						reference=message,
						mention_author=True,
					)

		except Exception as e:
			print(f'Error in message handling: {e}')

	#    await self.process_commands(message)  # Needed to process bot commands

	async def run_agent(self, task: str) -> str:
		try:
			browser = Browser(config=self.browser_config)
			agent = Agent(task=(task), llm=self.llm, browser=browser)
			result = await agent.run()

			agent_message = None
			if result.is_done():
				agent_message = result.history[-1].result[0].extracted_content

			if agent_message is None:
				agent_message = 'Oops! Something went wrong while running Browser-Use.'

			return agent_message

		except Exception as e:
			raise Exception(f'Browser-use task failed: {str(e)}')

```

---

## `browser-use-main\examples\integrations\discord\discord_example.py`

```py
"""
This examples requires you to have a Discord bot token and the bot already added to a server.

Five Steps to create and invite a Discord bot:

1. Create a Discord Application:
    *   Go to the Discord Developer Portal: https://discord.com/developers/applications
    *   Log in to the Discord website.
    *   Click on "New Application".
    *   Give the application a name and click "Create".
2. Configure the Bot:
    *   Navigate to the "Bot" tab on the left side of the screen.
    *   Make sure "Public Bot" is ticked if you want others to invite your bot.
	*	Generate your bot token by clicking on "Reset Token", Copy the token and save it securely.
        *   Do not share the bot token. Treat it like a password. If the token is leaked, regenerate it.
3. Enable Privileged Intents:
    *   Scroll down to the "Privileged Gateway Intents" section.
    *   Enable the necessary intents (e.g., "Server Members Intent" and "Message Content Intent").
   -->  Note: Enabling privileged intents for bots in over 100 guilds requires bot verification. You may need to contact Discord support to enable privileged intents for verified bots.
4. Generate Invite URL:
    *   Go to "OAuth2" tab and "OAuth2 URL Generator" section.
    *   Under "scopes", tick the "bot" checkbox.
    *   Tick the permissions required for your bot to function under “Bot Permissions”.
		*	e.g. "Send Messages", "Send Messages in Threads", "Read Message History",  "Mention Everyone".
    *   Copy the generated URL under the "GENERATED URL" section at the bottom.
5. Invite the Bot:
    *   Paste the URL into your browser.
    *   Choose a server to invite the bot to.
    *   Click “Authorize”.
   -->  Note: The person adding the bot needs "Manage Server" permissions.
6. Run the code below to start the bot with your bot token.
7. Write e.g. "/bu what's the weather in Tokyo?" to start a browser-use task and get a response inside the Discord channel.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import BrowserConfig
from examples.integrations.discord.discord_api import DiscordBot

# load credentials from environment variables
bot_token = os.getenv('DISCORD_BOT_TOKEN')
if not bot_token:
	raise ValueError('Discord bot token not found in .env file.')

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

bot = DiscordBot(
	llm=llm,  # required; instance of BaseChatModel
	prefix='$bu',  # optional; prefix of messages to trigger browser-use, defaults to "$bu"
	ack=True,  # optional; whether to acknowledge task receipt with a message, defaults to False
	browser_config=BrowserConfig(
		headless=False
	),  # optional; useful for changing headless mode or other browser configs, defaults to headless mode
)

bot.run(
	token=bot_token,  # required; Discord bot token
)

```

---

## `browser-use-main\examples\integrations\slack\README.md`

```md
# Slack Integration

Steps to create and configure a Slack bot:

1. Create a Slack App:
    *   Go to the Slack API: https://api.slack.com/apps
    *   Click on "Create New App".
    *   Choose "From scratch" and give your app a name and select the workspace.
    *   Provide a name and description for your bot (these are required fields).
2. Configure the Bot:
    *   Navigate to the "OAuth & Permissions" tab on the left side of the screen.
    *   Under "Scopes", add the necessary bot token scopes (add these "chat:write", "channels:history", "im:history").
3. Enable Event Subscriptions:
    *   Navigate to the "Event Subscriptions" tab.
    *   Enable events and add the necessary bot events (add these "message.channels", "message.im").
    *   Add your request URL (you can use ngrok to expose your local server if needed). [See how to set up ngrok](#installing-and-starting-ngrok).
    *   **Note:** The URL provided by ngrok is ephemeral and will change each time ngrok is started. You will need to update the request URL in the bot's settings each time you restart ngrok. [See how to update the request URL](#updating-the-request-url-in-bots-settings).
4. Add the bot to your Slack workspace:
    *   Navigate to the "OAuth & Permissions" tab.
    *   Under "OAuth Tokens for Your Workspace", click on "Install App to Workspace".
    *   Follow the prompts to authorize the app and add it to your workspace.
5. Set up environment variables:
    *   Obtain the `SLACK_SIGNING_SECRET`:
        *   Go to the Slack API: https://api.slack.com/apps
        *   Select your app.
        *   Navigate to the "Basic Information" tab.
        *   Copy the "Signing Secret".
    *   Obtain the `SLACK_BOT_TOKEN`:
        *   Go to the Slack API: https://api.slack.com/apps
        *   Select your app.
        *   Navigate to the "OAuth & Permissions" tab.
        *   Copy the "Bot User OAuth Token".
    *   Create a `.env` file in the root directory of your project and add the following lines:
        ```env
        SLACK_SIGNING_SECRET=your-signing-secret
        SLACK_BOT_TOKEN=your-bot-token
        ```
6. Invite the bot to a channel:
    *   Use the `/invite @your-bot-name` command in the Slack channel where you want the bot to be active.
7. Run the code in `examples/slack_example.py` to start the bot with your bot token and signing secret.
8. Write e.g. "$bu what's the weather in Tokyo?" to start a browser-use task and get a response inside the Slack channel.

## Installing and Starting ngrok

To expose your local server to the internet, you can use ngrok. Follow these steps to install and start ngrok:

1. Download ngrok from the official website: https://ngrok.com/download
2. Create a free account and follow the official steps to install ngrok.
3. Start ngrok by running the following command in your terminal:
    ```sh
    ngrok http 3000
    ```
    Replace `3000` with the port number your local server is running on.

## Updating the Request URL in Bot's Settings

If you need to update the request URL (e.g., when the ngrok URL changes), follow these steps:

1. Go to the Slack API: https://api.slack.com/apps
2. Select your app.
3. Navigate to the "Event Subscriptions" tab.
4. Update the "Request URL" field with the new ngrok URL. The URL should be something like: `https://<ngrok-id>.ngrok-free.app/slack/events`
5. Save the changes.

## Installing Required Packages

To run this example, you need to install the following packages:

- `fastapi`
- `uvicorn`
- `slack_sdk`

You can install these packages using pip:

```sh
pip install fastapi uvicorn slack_sdk

```

---

## `browser-use-main\examples\integrations\slack\slack_api.py`

```py
import logging
import os
import sys
from typing import Annotated

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, Request
from langchain_core.language_models.chat_models import BaseChatModel
from slack_sdk.errors import SlackApiError
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web.async_client import AsyncWebClient

from browser_use import BrowserConfig
from browser_use.agent.service import Agent, Browser
from browser_use.logging_config import setup_logging

setup_logging()
logger = logging.getLogger('slack')

app = FastAPI()


class SlackBot:
	def __init__(
		self,
		llm: BaseChatModel,
		bot_token: str,
		signing_secret: str,
		ack: bool = False,
		browser_config: BrowserConfig = BrowserConfig(headless=True),
	):
		if not bot_token or not signing_secret:
			raise ValueError('Bot token and signing secret must be provided')

		self.llm = llm
		self.ack = ack
		self.browser_config = browser_config
		self.client = AsyncWebClient(token=bot_token)
		self.signature_verifier = SignatureVerifier(signing_secret)
		self.processed_events = set()
		logger.info('SlackBot initialized')

	async def handle_event(self, event, event_id):
		try:
			logger.info(f'Received event id: {event_id}')
			if not event_id:
				logger.warning('Event ID missing in event data')
				return

			if event_id in self.processed_events:
				logger.info(f'Event {event_id} already processed')
				return
			self.processed_events.add(event_id)

			if 'subtype' in event and event['subtype'] == 'bot_message':
				return

			text = event.get('text')
			user_id = event.get('user')
			if text and text.startswith('$bu '):
				task = text[len('$bu ') :].strip()
				if self.ack:
					try:
						await self.send_message(
							event['channel'], f'<@{user_id}> Starting browser use task...', thread_ts=event.get('ts')
						)
					except Exception as e:
						logger.error(f'Error sending start message: {e}')

				try:
					agent_message = await self.run_agent(task)
					await self.send_message(event['channel'], f'<@{user_id}> {agent_message}', thread_ts=event.get('ts'))
				except Exception as e:
					await self.send_message(event['channel'], f'Error during task execution: {str(e)}', thread_ts=event.get('ts'))
		except Exception as e:
			logger.error(f'Error in handle_event: {str(e)}')

	async def run_agent(self, task: str) -> str:
		try:
			browser = Browser(config=self.browser_config)
			agent = Agent(task=task, llm=self.llm, browser=browser)
			result = await agent.run()

			agent_message = None
			if result.is_done():
				agent_message = result.history[-1].result[0].extracted_content

			if agent_message is None:
				agent_message = 'Oops! Something went wrong while running Browser-Use.'

			return agent_message

		except Exception as e:
			logger.error(f'Error during task execution: {str(e)}')
			return f'Error during task execution: {str(e)}'

	async def send_message(self, channel, text, thread_ts=None):
		try:
			await self.client.chat_postMessage(channel=channel, text=text, thread_ts=thread_ts)
		except SlackApiError as e:
			logger.error(f'Error sending message: {e.response["error"]}')


@app.post('/slack/events')
async def slack_events(request: Request, slack_bot: Annotated[SlackBot, Depends()]):
	try:
		if not slack_bot.signature_verifier.is_valid_request(await request.body(), dict(request.headers)):
			logger.warning('Request verification failed')
			raise HTTPException(status_code=400, detail='Request verification failed')

		event_data = await request.json()
		logger.info(f'Received event data: {event_data}')
		if 'challenge' in event_data:
			return {'challenge': event_data['challenge']}

		if 'event' in event_data:
			try:
				await slack_bot.handle_event(event_data.get('event'), event_data.get('event_id'))
			except Exception as e:
				logger.error(f'Error handling event: {str(e)}')

		return {}
	except Exception as e:
		logger.error(f'Error in slack_events: {str(e)}')
		raise HTTPException(status_code=500, detail='Internal Server Error')

```

---

## `browser-use-main\examples\integrations\slack\slack_example.py`

```py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import BrowserConfig
from examples.integrations.slack.slack_api import SlackBot, app

# load credentials from environment variables
bot_token = os.getenv('SLACK_BOT_TOKEN')
if not bot_token:
	raise ValueError('Slack bot token not found in .env file.')

signing_secret = os.getenv('SLACK_SIGNING_SECRET')
if not signing_secret:
	raise ValueError('Slack signing secret not found in .env file.')

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

slack_bot = SlackBot(
	llm=llm,  # required; instance of BaseChatModel
	bot_token=bot_token,  # required; Slack bot token
	signing_secret=signing_secret,  # required; Slack signing secret
	ack=True,  # optional; whether to acknowledge task receipt with a message, defaults to False
	browser_config=BrowserConfig(
		headless=True
	),  # optional; useful for changing headless mode or other browser configs, defaults to headless mode
)

app.dependency_overrides[SlackBot] = lambda: slack_bot

if __name__ == '__main__':
	import uvicorn

	uvicorn.run('integrations.slack.slack_api:app', host='0.0.0.0', port=3000)

```

---

## `browser-use-main\examples\models\_ollama.py`

```py
# Optional: Disable telemetry
# os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Optional: Set the OLLAMA host to a remote server
# os.environ["OLLAMA_HOST"] = "http://x.x.x.x:11434"

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama

from browser_use import Agent
from browser_use.agent.views import AgentHistoryList


async def run_search() -> AgentHistoryList:
	agent = Agent(
		task="Search for a 'browser use' post on the r/LocalLLaMA subreddit and open it.",
		llm=ChatOllama(
			model='qwen2.5:32b-instruct-q4_K_M',
			num_ctx=32000,
		),
	)

	result = await agent.run()
	return result


async def main():
	result = await run_search()
	print('\n\n', result)


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\models\azure_openai.py`

```py
"""
Simple try of the agent.

@dev You need to add AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI

from browser_use import Agent

# Retrieve Azure-specific environment variables
azure_openai_api_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

if not azure_openai_api_key or not azure_openai_endpoint:
	raise ValueError('AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT is not set')

# Initialize the Azure OpenAI client
llm = AzureChatOpenAI(
	model_name='gpt-4o',
	openai_api_key=azure_openai_api_key,
	azure_endpoint=azure_openai_endpoint,  # Corrected to use azure_endpoint instead of openai_api_base
	deployment_name='gpt-4o',  # Use deployment_name for Azure models
	api_version='2024-08-01-preview',  # Explicitly set the API version here
)

agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
	enable_memory=True,
)


async def main():
	await agent.run(max_steps=10)
	input('Press Enter to continue...')


asyncio.run(main())

```

---

## `browser-use-main\examples\models\bedrock_claude.py`

```py
"""
Automated news analysis and sentiment scoring using Bedrock.

Ensure you have browser-use installed with `examples` extra, i.e. `uv install 'browser-use[examples]'`

@dev Ensure AWS environment variables are set correctly for Bedrock access.
"""

import argparse
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller


def get_llm():
	config = Config(retries={'max_attempts': 10, 'mode': 'adaptive'})
	bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1', config=config)

	return ChatBedrockConverse(
		model_id='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
		temperature=0.0,
		max_tokens=None,
		client=bedrock_client,
	)


# Define the task for the agent
task = (
	"Visit cnn.com, navigate to the 'World News' section, and identify the latest headline. "
	'Open the first article and summarize its content in 3-4 sentences. '
	'Additionally, analyze the sentiment of the article (positive, neutral, or negative) '
	'and provide a confidence score for the sentiment. Present the result in a tabular format.'
)

parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, help='The query for the agent to execute', default=task)
args = parser.parse_args()

llm = get_llm()

browser = Browser(
	config=BrowserConfig(
		# browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	)
)

agent = Agent(
	task=args.query,
	llm=llm,
	controller=Controller(),
	browser=browser,
	validate_output=True,
)


async def main():
	await agent.run(max_steps=30)
	await browser.close()


asyncio.run(main())

```

---

## `browser-use-main\examples\models\claude-3.7-sonnet.py`

```py
"""
Simple script that runs the task of opening amazon and searching.
@dev Ensure we have a `ANTHROPIC_API_KEY` variable in our `.env` file.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_anthropic import ChatAnthropic

from browser_use import Agent

llm = ChatAnthropic(model_name='claude-3-7-sonnet-20250219', temperature=0.0, timeout=30, stop=None)

agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)


async def main():
	await agent.run(max_steps=10)


asyncio.run(main())

```

---

## `browser-use-main\examples\models\deepseek-r1.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from browser_use import Agent

api_key = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=('go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result'),
		llm=ChatDeepSeek(
			base_url='https://api.deepseek.com/v1',
			model='deepseek-reasoner',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
		max_failures=2,
		max_actions_per_step=1,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\models\deepseek.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_deepseek import ChatDeepSeek
from pydantic import SecretStr

from browser_use import Agent

api_key = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key:
	raise ValueError('DEEPSEEK_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=(
			'1. Go to https://www.reddit.com/r/LocalLLaMA '
			"2. Search for 'browser use' in the search bar"
			'3. Click on first result'
			'4. Return the first comment'
		),
		llm=ChatDeepSeek(
			base_url='https://api.deepseek.com/v1',
			model='deepseek-chat',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\models\gemini.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		viewport_expansion=0,
		user_data_dir='~/.config/browseruse/profiles/default',
	)
)


async def run_search():
	agent = Agent(
		task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
		llm=llm,
		max_actions_per_step=4,
		browser_session=browser_session,
	)

	await agent.run(max_steps=25)


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\models\gpt-4o.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

llm = ChatOpenAI(model='gpt-4o')
agent = Agent(
	task='Go to amazon.com, search for laptop, sort by best rating, and give me the price of the first result',
	llm=llm,
)


async def main():
	await agent.run(max_steps=10)
	input('Press Enter to continue...')


asyncio.run(main())

```

---

## `browser-use-main\examples\models\grok.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

api_key = os.getenv('GROK_API_KEY', '')
if not api_key:
	raise ValueError('GROK_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=(
			'1. Go to https://www.amazon.com'
			'2. Search for "wireless headphones"'
			'3. Filter by "Highest customer rating"'
			'4. Return the title and price of the first product'
		),
		llm=ChatOpenAI(
			base_url='https://api.x.ai/v1',
			model='grok-3-beta',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\models\novita.py`

```py
"""
Simple try of the agent.

@dev You need to add NOVITA_API_KEY to your environment variables.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

api_key = os.getenv('NOVITA_API_KEY', '')
if not api_key:
	raise ValueError('NOVITA_API_KEY is not set')


async def run_search():
	agent = Agent(
		task=(
			'1. Go to https://www.reddit.com/r/LocalLLaMA '
			"2. Search for 'browser use' in the search bar"
			'3. Click on first result'
			'4. Return the first comment'
		),
		llm=ChatOpenAI(
			base_url='https://api.novita.ai/v3/openai',
			model='deepseek/deepseek-v3-0324',
			api_key=SecretStr(api_key),
		),
		use_vision=False,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\models\qwen.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_ollama import ChatOllama

from browser_use import Agent


async def run_search():
	agent = Agent(
		task=(
			"1. Go to https://www.reddit.com/r/LocalLLaMA2. Search for 'browser use' in the search bar3. Click search4. Call done"
		),
		llm=ChatOllama(
			# model='qwen2.5:32b-instruct-q4_K_M',
			# model='qwen2.5:14b',
			model='qwen2.5:latest',
			num_ctx=128000,
		),
		max_actions_per_step=1,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())

```

---

## `browser-use-main\examples\models\README.md`

```md
# Gemini
Detailed video on how to integrate browser-use with Gemini: https://www.youtube.com/watch?v=JluZiWBV_Tc

```

---

## `browser-use-main\examples\simple.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

# Initialize the model
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)
task = 'Go to kayak.com and find the cheapest one-way flight from Zurich to San Francisco in 3 weeks.'
agent = Agent(task=task, llm=llm)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\ui\command_line.py`

```py
"""
To Use It:

Example 1: Using OpenAI (default), with default task: 'go to reddit and search for posts about browser-use'
python command_line.py

Example 2: Using OpenAI with a Custom Query
python command_line.py --query "go to google and search for browser-use"

Example 3: Using Anthropic's Claude Model with a Custom Query
python command_line.py --query "find latest Python tutorials on Medium" --provider anthropic

"""

import argparse
import asyncio
import os
import sys

# Ensure local repository (browser_use) is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller


def get_llm(provider: str):
	if provider == 'anthropic':
		from langchain_anthropic import ChatAnthropic

		api_key = os.getenv('ANTHROPIC_API_KEY')
		if not api_key:
			raise ValueError('Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.')

		return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
	elif provider == 'openai':
		from langchain_openai import ChatOpenAI

		api_key = os.getenv('OPENAI_API_KEY')
		if not api_key:
			raise ValueError('Error: OPENAI_API_KEY is not set. Please provide a valid API key.')

		return ChatOpenAI(model='gpt-4o', temperature=0.0)

	else:
		raise ValueError(f'Unsupported provider: {provider}')


def parse_arguments():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description='Automate browser tasks using an LLM agent.')
	parser.add_argument(
		'--query', type=str, help='The query to process', default='go to reddit and search for posts about browser-use'
	)
	parser.add_argument(
		'--provider',
		type=str,
		choices=['openai', 'anthropic'],
		default='openai',
		help='The model provider to use (default: openai)',
	)
	return parser.parse_args()


def initialize_agent(query: str, provider: str):
	"""Initialize the browser agent with the given query and provider."""
	llm = get_llm(provider)
	controller = Controller()
	browser = Browser(config=BrowserConfig())

	return Agent(
		task=query,
		llm=llm,
		controller=controller,
		browser=browser,
		use_vision=True,
		max_actions_per_step=1,
	), browser


async def main():
	"""Main async function to run the agent."""
	args = parse_arguments()
	agent, browser = initialize_agent(args.query, args.provider)

	await agent.run(max_steps=25)

	input('Press Enter to close the browser...')
	await browser.close()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\ui\gradio_demo.py`

```py
import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

# Third-party imports
import gradio as gr
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Local module imports
from browser_use import Agent


@dataclass
class ActionResult:
	is_done: bool
	extracted_content: str | None
	error: str | None
	include_in_memory: bool


@dataclass
class AgentHistoryList:
	all_results: list[ActionResult]
	all_model_outputs: list[dict]


def parse_agent_history(history_str: str) -> None:
	console = Console()

	# Split the content into sections based on ActionResult entries
	sections = history_str.split('ActionResult(')

	for i, section in enumerate(sections[1:], 1):  # Skip first empty section
		# Extract relevant information
		content = ''
		if 'extracted_content=' in section:
			content = section.split('extracted_content=')[1].split(',')[0].strip("'")

		if content:
			header = Text(f'Step {i}', style='bold blue')
			panel = Panel(content, title=header, border_style='blue')
			console.print(panel)
			console.print()


async def run_browser_task(
	task: str,
	api_key: str,
	model: str = 'gpt-4o',
	headless: bool = True,
) -> str:
	if not api_key.strip():
		return 'Please provide an API key'

	os.environ['OPENAI_API_KEY'] = api_key

	try:
		agent = Agent(
			task=task,
			llm=ChatOpenAI(model='gpt-4o'),
		)
		result = await agent.run()
		#  TODO: The result cloud be parsed better
		return result
	except Exception as e:
		return f'Error: {str(e)}'


def create_ui():
	with gr.Blocks(title='Browser Use GUI') as interface:
		gr.Markdown('# Browser Use Task Automation')

		with gr.Row():
			with gr.Column():
				api_key = gr.Textbox(label='OpenAI API Key', placeholder='sk-...', type='password')
				task = gr.Textbox(
					label='Task Description',
					placeholder='E.g., Find flights from New York to London for next week',
					lines=3,
				)
				model = gr.Dropdown(choices=['gpt-4', 'gpt-3.5-turbo'], label='Model', value='gpt-4')
				headless = gr.Checkbox(label='Run Headless', value=True)
				submit_btn = gr.Button('Run Task')

			with gr.Column():
				output = gr.Textbox(label='Output', lines=10, interactive=False)

		submit_btn.click(
			fn=lambda *args: asyncio.run(run_browser_task(*args)),
			inputs=[task, api_key, model, headless],
			outputs=output,
		)

	return interface


if __name__ == '__main__':
	demo = create_ui()
	demo.launch()

```

---

## `browser-use-main\examples\ui\README.md`

```md
# **User Interfaces of Browser-Use**

| **File Name**          | **User Interface** | **Description**                           | **Example Usage**                         |
|------------------------|-------------------|-------------------------------------------|-------------------------------------------|
| `command_line.py`      | **Terminal**      | Parses arguments for command-line execution. | `python command_line.py`                  |
| `gradio_demo.py`       | **Gradio**        | Provides a Gradio-based interactive UI.  | `python gradio_demo.py`                   |
| `streamlit_demo.py`    | **Streamlit**     | Runs a Streamlit-based web interface.    | `python -m streamlit run streamlit_demo.py` |

```

---

## `browser-use-main\examples\ui\streamlit_demo.py`

```py
"""
To use it, you'll need to install streamlit, and run with:

python -m streamlit run streamlit_demo.py

"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller

if os.name == 'nt':
	asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# Function to get the LLM based on provider
def get_llm(provider: str):
	if provider == 'anthropic':
		from langchain_anthropic import ChatAnthropic

		api_key = os.getenv('ANTHROPIC_API_KEY')
		if not api_key:
			st.error('Error: ANTHROPIC_API_KEY is not set. Please provide a valid API key.')
			st.stop()

		return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None, temperature=0.0)
	elif provider == 'openai':
		from langchain_openai import ChatOpenAI

		api_key = os.getenv('OPENAI_API_KEY')
		if not api_key:
			st.error('Error: OPENAI_API_KEY is not set. Please provide a valid API key.')
			st.stop()

		return ChatOpenAI(model='gpt-4o', temperature=0.0)
	else:
		st.error(f'Unsupported provider: {provider}')
		st.stop()


# Function to initialize the agent
def initialize_agent(query: str, provider: str):
	llm = get_llm(provider)
	controller = Controller()
	browser = Browser(config=BrowserConfig())

	return Agent(
		task=query,
		llm=llm,
		controller=controller,
		browser=browser,
		use_vision=True,
		max_actions_per_step=1,
	), browser


# Streamlit UI
st.title('Automated Browser Agent with LLMs 🤖')

query = st.text_input('Enter your query:', 'go to reddit and search for posts about browser-use')
provider = st.radio('Select LLM Provider:', ['openai', 'anthropic'], index=0)

if st.button('Run Agent'):
	st.write('Initializing agent...')
	agent, browser = initialize_agent(query, provider)

	async def run_agent():
		with st.spinner('Running automation...'):
			await agent.run(max_steps=25)
		st.success('Task completed! 🎉')

	asyncio.run(run_agent())

	st.button('Close Browser', on_click=lambda: asyncio.run(browser.close()))

```

---

## `browser-use-main\examples\use-cases\captcha.py`

```py
"""
Goal: Automates CAPTCHA solving on a demo website.


Simple try of the agent.
@dev You need to add OPENAI_API_KEY to your environment variables.
NOTE: captchas are hard. For this example it works. But e.g. for iframes it does not.
for this example it helps to zoom in.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')


async def main():
	llm = ChatOpenAI(model='gpt-4o')
	agent = Agent(
		task='go to https://captcha.com/demos/features/captcha-demo.aspx and solve the captcha',
		llm=llm,
	)
	await agent.run()
	input('Press Enter to exit')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\check_appointment.py`

```py
# Goal: Checks for available visa appointment slots on the Greece MFA website.

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use.agent.service import Agent
from browser_use.controller.service import Controller

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')

controller = Controller()


class WebpageInfo(BaseModel):
	"""Model for webpage link."""

	link: str = 'https://appointment.mfa.gr/en/reservations/aero/ireland-grcon-dub/'


@controller.action('Go to the webpage', param_model=WebpageInfo)
def go_to_webpage(webpage_info: WebpageInfo):
	"""Returns the webpage link."""
	return webpage_info.link


async def main():
	"""Main function to execute the agent task."""
	task = (
		'Go to the Greece MFA webpage via the link I provided you.'
		'Check the visa appointment dates. If there is no available date in this month, check the next month.'
		'If there is no available date in both months, tell me there is no available date.'
	)

	model = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(os.getenv('OPENAI_API_KEY', '')))
	agent = Agent(task, model, controller=controller, use_vision=True)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\find_and_apply_to_jobs.py`

```py
"""
Goal: Searches for job listings, evaluates relevance based on a CV, and applies

@dev You need to add OPENAI_API_KEY to your environment variables.
Also you have to install PyPDF2 to read pdf files: pip install PyPDF2
"""

import asyncio
import csv
import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, SecretStr
from PyPDF2 import PdfReader

from browser_use import ActionResult, Agent, Controller
from browser_use.browser import BrowserProfile, BrowserSession

required_env_vars = ['AZURE_OPENAI_KEY', 'AZURE_OPENAI_ENDPOINT']
for var in required_env_vars:
	if not os.getenv(var):
		raise ValueError(f'{var} is not set. Please add it to your environment variables.')

logger = logging.getLogger(__name__)
# full screen mode
controller = Controller()

# NOTE: This is the path to your cv file
CV = Path.cwd() / 'cv_04_24.pdf'

if not CV.exists():
	raise FileNotFoundError(f'You need to set the path to your cv file in the CV variable. CV file not found at {CV}')


class Job(BaseModel):
	title: str
	link: str
	company: str
	fit_score: float
	location: str | None = None
	salary: str | None = None


@controller.action('Save jobs to file - with a score how well it fits to my profile', param_model=Job)
def save_jobs(job: Job):
	with open('jobs.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		writer.writerow([job.title, job.company, job.link, job.salary, job.location])

	return 'Saved job to file'


@controller.action('Read jobs from file')
def read_jobs():
	with open('jobs.csv') as f:
		return f.read()


@controller.action('Read my cv for context to fill forms')
def read_cv():
	pdf = PdfReader(CV)
	text = ''
	for page in pdf.pages:
		text += page.extract_text() or ''
	logger.info(f'Read cv with {len(text)} characters')
	return ActionResult(extracted_content=text, include_in_memory=True)


@controller.action(
	'Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element',
)
async def upload_cv(index: int, browser_session: BrowserSession):
	path = str(CV.absolute())
	dom_el = await browser_session.get_dom_element_by_index(index)

	if dom_el is None:
		return ActionResult(error=f'No element found at index {index}')

	file_upload_dom_el = dom_el.get_file_upload_element()

	if file_upload_dom_el is None:
		logger.info(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')

	file_upload_el = await browser_session.get_locate_element(file_upload_dom_el)

	if file_upload_el is None:
		logger.info(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')

	try:
		await file_upload_el.set_input_files(path)
		msg = f'Successfully uploaded file "{path}" to index {index}'
		logger.info(msg)
		return ActionResult(extracted_content=msg)
	except Exception as e:
		logger.debug(f'Error in set_input_files: {str(e)}')
		return ActionResult(error=f'Failed to upload file to index {index}')


browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		disable_security=True,
		user_data_dir='~/.config/browseruse/profiles/default',
	)
)


async def main():
	# ground_task = (
	# 	'You are a professional job finder. '
	# 	'1. Read my cv with read_cv'
	# 	'2. Read the saved jobs file '
	# 	'3. start applying to the first link of Amazon '
	# 	'You can navigate through pages e.g. by scrolling '
	# 	'Make sure to be on the english version of the page'
	# )
	ground_task = (
		'You are a professional job finder. '
		'1. Read my cv with read_cv'
		'find ml internships in and save them to a file'
		'search at company:'
	)
	tasks = [
		ground_task + '\n' + 'Google',
		# ground_task + '\n' + 'Amazon',
		# ground_task + '\n' + 'Apple',
		# ground_task + '\n' + 'Microsoft',
		# ground_task
		# + '\n'
		# + 'go to https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Taiwan%2C-Remote/Fulfillment-Analyst---New-College-Graduate-2025_JR1988949/apply/autofillWithResume?workerSubType=0c40f6bd1d8f10adf6dae42e46d44a17&workerSubType=ab40a98049581037a3ada55b087049b7 NVIDIA',
		# ground_task + '\n' + 'Meta',
	]
	model = AzureChatOpenAI(
		model='gpt-4o',
		api_version='2024-10-21',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)

	agents = []
	for task in tasks:
		agent = Agent(task=task, llm=model, controller=controller, browser_session=browser_session)
		agents.append(agent)

	await asyncio.gather(*[agent.run() for agent in agents])


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\find_influencer_profiles.py`

```py
"""
Show how to use custom outputs.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import Agent, Controller
from browser_use.agent.views import ActionResult


class Profile(BaseModel):
	platform: str
	profile_url: str


class Profiles(BaseModel):
	profiles: list[Profile]


controller = Controller(exclude_actions=['search_google'], output_model=Profiles)
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

if not BEARER_TOKEN:
	# use the api key for ask tessa
	# you can also use other apis like exa, xAI, perplexity, etc.
	raise ValueError('BEARER_TOKEN is not set - go to https://www.heytessa.ai/ and create an api key')


@controller.registry.action('Search the web for a specific query')
async def search_web(query: str):
	keys_to_use = ['url', 'title', 'content', 'author', 'score']
	headers = {'Authorization': f'Bearer {BEARER_TOKEN}'}
	async with httpx.AsyncClient() as client:
		response = await client.post(
			'https://asktessa.ai/api/search',
			headers=headers,
			json={'query': query},
		)

	final_results = [
		{key: source[key] for key in keys_to_use if key in source}
		for source in await response.json()['sources']
		if source['score'] >= 0.2
	]
	# print(json.dumps(final_results, indent=4))
	result_text = json.dumps(final_results, indent=4)
	print(result_text)
	return ActionResult(extracted_content=result_text, include_in_memory=True)


async def main():
	task = (
		'Go to this tiktok video url, open it and extract the @username from the resulting url. Then do a websearch for this username to find all his social media profiles. Return me the links to the social media profiles with the platform name.'
		' https://www.tiktokv.com/share/video/7470981717659110678/  '
	)
	model = ChatOpenAI(model='gpt-4o')
	agent = Agent(task=task, llm=model, controller=controller)

	history = await agent.run()

	result = history.final_result()
	if result:
		parsed: Profiles = Profiles.model_validate_json(result)

		for profile in parsed.profiles:
			print('\n--------------------------------')
			print(f'Platform:         {profile.platform}')
			print(f'Profile URL:      {profile.profile_url}')

	else:
		print('No result')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\google_sheets.py`

```py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

import pyperclip
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import ActionResult, Agent, Controller
from browser_use.browser import BrowserProfile, BrowserSession

# Load environment variables
load_dotenv()
if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')


controller = Controller()


def is_google_sheet(page) -> bool:
	return page.url.startswith('https://docs.google.com/spreadsheets/')


@controller.registry.action('Google Sheets: Open a specific Google Sheet')
async def open_google_sheet(browser_session: BrowserSession, google_sheet_url: str):
	page = await browser_session.get_current_page()
	if page.url != google_sheet_url:
		await page.goto(google_sheet_url)
		await page.wait_for_load_state()
	if not is_google_sheet(page):
		return ActionResult(error='Failed to open Google Sheet, are you sure you have permissions to access this sheet?')
	return ActionResult(extracted_content=f'Opened Google Sheet {google_sheet_url}', include_in_memory=False)


@controller.registry.action('Google Sheets: Get the contents of the entire sheet', page_filter=is_google_sheet)
async def get_sheet_contents(browser_session: BrowserSession):
	page = await browser_session.get_current_page()

	# select all cells
	await page.keyboard.press('Enter')
	await page.keyboard.press('Escape')
	await page.keyboard.press('ControlOrMeta+A')
	await page.keyboard.press('ControlOrMeta+C')

	extracted_tsv = pyperclip.paste()
	return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)


@controller.registry.action('Google Sheets: Select a specific cell or range of cells', page_filter=is_google_sheet)
async def select_cell_or_range(browser_session: BrowserSession, cell_or_range: str):
	page = await browser_session.get_current_page()

	await page.keyboard.press('Enter')  # make sure we dont delete current cell contents if we were last editing
	await page.keyboard.press('Escape')  # to clear current focus (otherwise select range popup is additive)
	await asyncio.sleep(0.1)
	await page.keyboard.press('Home')  # move cursor to the top left of the sheet first
	await page.keyboard.press('ArrowUp')
	await asyncio.sleep(0.1)
	await page.keyboard.press('Control+G')  # open the goto range popup
	await asyncio.sleep(0.2)
	await page.keyboard.type(cell_or_range, delay=0.05)
	await asyncio.sleep(0.2)
	await page.keyboard.press('Enter')
	await asyncio.sleep(0.2)
	await page.keyboard.press('Escape')  # to make sure the popup still closes in the case where the jump failed
	return ActionResult(extracted_content=f'Selected cell {cell_or_range}', include_in_memory=False)


@controller.registry.action('Google Sheets: Get the contents of a specific cell or range of cells', page_filter=is_google_sheet)
async def get_range_contents(browser_session: BrowserSession, cell_or_range: str):
	page = await browser_session.get_current_page()

	await select_cell_or_range(browser_session, cell_or_range)

	await page.keyboard.press('ControlOrMeta+C')
	await asyncio.sleep(0.1)
	extracted_tsv = pyperclip.paste()
	return ActionResult(extracted_content=extracted_tsv, include_in_memory=True)


@controller.registry.action('Google Sheets: Clear the currently selected cells', page_filter=is_google_sheet)
async def clear_selected_range(browser_session: BrowserSession):
	page = await browser_session.get_current_page()

	await page.keyboard.press('Backspace')
	return ActionResult(extracted_content='Cleared selected range', include_in_memory=False)


@controller.registry.action('Google Sheets: Input text into the currently selected cell', page_filter=is_google_sheet)
async def input_selected_cell_text(browser_session: BrowserSession, text: str):
	page = await browser_session.get_current_page()

	await page.keyboard.type(text, delay=0.1)
	await page.keyboard.press('Enter')  # make sure to commit the input so it doesn't get overwritten by the next action
	await page.keyboard.press('ArrowUp')
	return ActionResult(extracted_content=f'Inputted text {text}', include_in_memory=False)


@controller.registry.action('Google Sheets: Batch update a range of cells', page_filter=is_google_sheet)
async def update_range_contents(browser_session: BrowserSession, range: str, new_contents_tsv: str):
	page = await browser_session.get_current_page()

	await select_cell_or_range(browser_session, range)

	# simulate paste event from clipboard with TSV content
	await page.evaluate(f"""
        const clipboardData = new DataTransfer();
        clipboardData.setData('text/plain', `{new_contents_tsv}`);
        document.activeElement.dispatchEvent(new ClipboardEvent('paste', {{clipboardData}}));
    """)

	return ActionResult(extracted_content=f'Updated cell {range} with {new_contents_tsv}', include_in_memory=False)


# many more snippets for keyboard-shortcut based Google Sheets automation can be found here, see:
# - https://github.com/philc/sheetkeys/blob/master/content_scripts/sheet_actions.js
# - https://github.com/philc/sheetkeys/blob/master/content_scripts/commands.js
# - https://support.google.com/docs/answer/181110?hl=en&co=GENIE.Platform%3DDesktop#zippy=%2Cmac-shortcuts

# Tip: LLM is bad at spatial reasoning, don't make it navigate with arrow keys relative to current cell
# if given arrow keys, it will try to jump from G1 to A2 by pressing Down, without realizing needs to go Down+LeftLeftLeftLeft


async def main():
	browser_session = BrowserSession(
		browser_profile=BrowserProfile(
			executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
			user_data_dir='~/.config/browseruse/profiles/default',
		)
	)

	async with browser_session:
		model = ChatOpenAI(model='gpt-4o')

		eraser = Agent(
			task="""
                Clear all the existing values in columns A through F in this Google Sheet:
                https://docs.google.com/spreadsheets/d/1INaIcfpYXlMRWO__de61SHFCaqt1lfHlcvtXZPItlpI/edit
            """,
			llm=model,
			browser_session=browser_session,
			controller=controller,
		)
		await eraser.run()

		researcher = Agent(
			task="""
                Google to find the full name, nationality, and date of birth of the CEO of the top 10 Fortune 100 companies.
                For each company, append a row to this existing Google Sheet: https://docs.google.com/spreadsheets/d/1INaIcfpYXlMRWO__de61SHFCaqt1lfHlcvtXZPItlpI/edit
                Make sure column headers are present and all existing values in the sheet are formatted correctly.
                Columns:
                    A: Company Name
                    B: CEO Full Name
                    C: CEO Country of Birth
                    D: CEO Date of Birth (YYYY-MM-DD)
                    E: Source URL where the information was found
            """,
			llm=model,
			browser_session=browser_session,
			controller=controller,
		)
		await researcher.run()

		improvised_continuer = Agent(
			task="""
                Read the Google Sheet https://docs.google.com/spreadsheets/d/1INaIcfpYXlMRWO__de61SHFCaqt1lfHlcvtXZPItlpI/edit
                Add 3 more rows to the bottom continuing the existing pattern, make sure any data you add is sourced correctly.
            """,
			llm=model,
			browser_session=browser_session,
			controller=controller,
		)
		await improvised_continuer.run()

		final_fact_checker = Agent(
			task="""
                Read the Google Sheet https://docs.google.com/spreadsheets/d/1INaIcfpYXlMRWO__de61SHFCaqt1lfHlcvtXZPItlpI/edit
                Fact-check every entry, add a new column F with your findings for each row.
                Make sure to check the source URL for each row, and make sure the information is correct.
            """,
			llm=model,
			browser_session=browser_session,
			controller=controller,
		)
		await final_fact_checker.run()


if __name__ == '__main__':
	asyncio.run(main
… (truncated)

```

---

## `browser-use-main\examples\use-cases\online_coding_agent.py`

```py
# Goal: Implements a multi-agent system for online code editors, with separate agents for coding and execution.

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')


async def main():
	browser = Browser()
	async with await browser.new_context() as context:
		model = ChatOpenAI(model='gpt-4o')

		# Initialize browser agent
		agent1 = Agent(
			task='Open an online code editor programiz.',
			llm=model,
			browser_context=context,
		)
		executor = Agent(
			task='Executor. Execute the code written by the coder and suggest some updates if there are errors.',
			llm=model,
			browser_context=context,
		)

		coder = Agent(
			task='Coder. Your job is to write and complete code. You are an expert coder. Code a simple calculator. Write the code on the coding interface after agent1 has opened the link.',
			llm=model,
			browser_context=context,
		)
		await agent1.run()
		await executor.run()
		await coder.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\post-twitter.py`

```py
"""
Goal: Provides a template for automated posting on X (Twitter), including new tweets, tagging, and replies.

X Posting Template using browser-use
----------------------------------------

This template allows you to automate posting on X using browser-use.
It supports:
- Posting new tweets
- Tagging users
- Replying to tweets

Add your target user and message in the config section.

target_user="XXXXX"
message="XXXXX"
reply_url="XXXXX"

Any issues, contact me on X @defichemist95
"""

import asyncio
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set. Please add it to your environment variables.')


# ============ Configuration Section ============
@dataclass
class TwitterConfig:
	"""Configuration for Twitter posting"""

	openai_api_key: str
	chrome_path: str
	target_user: str  # Twitter handle without @
	message: str
	reply_url: str
	headless: bool = False
	model: str = 'gpt-4o-mini'
	base_url: str = 'https://x.com/home'


# Customize these settings
config = TwitterConfig(
	openai_api_key=os.getenv('OPENAI_API_KEY'),
	chrome_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # This is for MacOS (Chrome)
	target_user='XXXXX',
	message='XXXXX',
	reply_url='XXXXX',
	headless=False,
)


def create_twitter_agent(config: TwitterConfig) -> Agent:
	llm = ChatOpenAI(model=config.model, api_key=config.openai_api_key)

	browser = Browser(
		config=BrowserConfig(
			headless=config.headless,
			browser_binary_path=config.chrome_path,
		)
	)

	controller = Controller()

	# Construct the full message with tag
	full_message = f'@{config.target_user} {config.message}'

	# Create the agent with detailed instructions
	return Agent(
		task=f"""Navigate to Twitter and create a post and reply to a tweet.

        Here are the specific steps:

        1. Go to {config.base_url}. See the text input field at the top of the page that says "What's happening?"
        2. Look for the text input field at the top of the page that says "What's happening?"
        3. Click the input field and type exactly this message:
        "{full_message}"
        4. Find and click the "Post" button (look for attributes: 'button' and 'data-testid="tweetButton"')
        5. Do not click on the '+' button which will add another tweet.

        6. Navigate to {config.reply_url}
        7. Before replying, understand the context of the tweet by scrolling down and reading the comments.
        8. Reply to the tweet under 50 characters.

        Important:
        - Wait for each element to load before interacting
        - Make sure the message is typed exactly as shown
        - Verify the post button is clickable before clicking
        - Do not click on the '+' button which will add another tweet
        """,
		llm=llm,
		controller=controller,
		browser=browser,
	)


async def post_tweet(agent: Agent):
	try:
		await agent.run(max_steps=100)
		agent.create_history_gif()
		print('Tweet posted successfully!')
	except Exception as e:
		print(f'Error posting tweet: {str(e)}')


async def main():
	agent = create_twitter_agent(config)
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\README.md`

```md
# Use Cases of Browser-Use

| File Name | Description |
|-----------|------------|
| `captcha.py` | Automates CAPTCHA solving on a demo website. |
| `check_appointment.py` | Checks for available visa appointment slots on the Greece MFA website. |
| `find_and_apply_to_jobs.py` | Searches for job listings, evaluates relevance based on a CV, and applies automatically. |
| `online_coding_agent.py` | Implements a multi-agent system for online code editors, with separate agents for coding and execution. |
| `post-twitter.py` | Provides a template for automated posting on X (Twitter), including new tweets, tagging, and replies. |
| `scrolling_page.py` | Automates webpage scrolling with various scrolling actions and text search functionality. |
| `twitter_post_using_cookies.py` | Automates posting on X (Twitter) using stored authentication cookies. |
| `web_voyager_agent.py` | A general-purpose web navigation agent for tasks like flight booking and course searching. |

```

---

## `browser-use-main\examples\use-cases\scrolling_page.py`

```py
# Goal: Automates webpage scrolling with various scrolling actions and text search functionality.

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig

if not os.getenv('OPENAI_API_KEY'):
	raise ValueError('OPENAI_API_KEY is not set')

"""
Example: Using the 'Scroll down' action.

This script demonstrates how the agent can navigate to a webpage and scroll down the content.
If no amount is specified, the agent will scroll down by one page height.
"""

llm = ChatOpenAI(model='gpt-4o')

agent = Agent(
	# task="Navigate to 'https://en.wikipedia.org/wiki/Internet' and scroll down by one page - then scroll up by 100 pixels - then scroll down by 100 pixels - then scroll down by 10000 pixels.",
	task="Navigate to 'https://en.wikipedia.org/wiki/Internet' and scroll to the string 'The vast majority of computer'",
	llm=llm,
	browser=Browser(config=BrowserConfig(headless=False)),
)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\shopping.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent, Browser

task = """
   ### Prompt for Shopping Agent – Migros Online Grocery Order

**Objective:**
Visit [Migros Online](https://www.migros.ch/en), search for the required grocery items, add them to the cart, select an appropriate delivery window, and complete the checkout process using TWINT.

**Important:**
- Make sure that you don't buy more than it's needed for each article.
- After your search, if you click  the "+" button, it adds the item to the basket.
- if you open the basket sidewindow menu, you can close it by clicking the X button on the top right. This will help you navigate easier.
---

### Step 1: Navigate to the Website
- Open [Migros Online](https://www.migros.ch/en).
- You should be logged in as Nikolaos Kaliorakis

---

### Step 2: Add Items to the Basket

#### Shopping List:

**Meat & Dairy:**
- Beef Minced meat (1 kg)
- Gruyère cheese (grated preferably)
- 2 liters full-fat milk
- Butter (cheapest available)

**Vegetables:**
- Carrots (1kg pack)
- Celery
- Leeks (1 piece)
- 1 kg potatoes

At this stage, check the basket on the top right (indicates the price) and check if you bought the right items.

**Fruits:**
- 2 lemons
- Oranges (for snacking)

**Pantry Items:**
- Lasagna sheets
- Tahini
- Tomato paste (below CHF2)
- Black pepper refill (not with the mill)
- 2x 1L Oatly Barista(oat milk)
- 1 pack of eggs (10 egg package)

#### Ingredients I already have (DO NOT purchase):
- Olive oil, garlic, canned tomatoes, dried oregano, bay leaves, salt, chili flakes, flour, nutmeg, cumin.

---

### Step 3: Handling Unavailable Items
- If an item is **out of stock**, find the best alternative.
- Use the following recipe contexts to choose substitutions:
  - **Pasta Bolognese & Lasagna:** Minced meat, tomato paste, lasagna sheets, milk (for béchamel), Gruyère cheese.
  - **Hummus:** Tahini, chickpeas, lemon juice, olive oil.
  - **Chickpea Curry Soup:** Chickpeas, leeks, curry, lemons.
  - **Crispy Slow-Cooked Pork Belly with Vegetables:** Potatoes, butter.
- Example substitutions:
  - If Gruyère cheese is unavailable, select another semi-hard cheese.
  - If Tahini is unavailable, a sesame-based alternative may work.

---

### Step 4: Adjusting for Minimum Order Requirement
- If the total order **is below CHF 99**, add **a liquid soap refill** to reach the minimum. If it;s still you can buy some bread, dark chockolate.
- At this step, check if you have bought MORE items than needed. If the price is more then CHF200, you MUST remove items.
- If an item is not available, choose an alternative.
- if an age verification is needed, remove alcoholic products, we haven't verified yet.

---

### Step 5: Select Delivery Window
- Choose a **delivery window within the current week**. It's ok to pay up to CHF2 for the window selection.
- Preferably select a slot within the workweek.

---

### Step 6: Checkout
- Proceed to checkout.
- Select **TWINT** as the payment method.
- Check out.
- 
- if it's needed the username is: nikoskalio.dev@gmail.com 
- and the password is : TheCircuit.Migros.dev!
---

### Step 7: Confirm Order & Output Summary
- Once the order is placed, output a summary including:
  - **Final list of items purchased** (including any substitutions).
  - **Total cost**.
  - **Chosen delivery time**.

**Important:** Ensure efficiency and accuracy throughout the process."""

browser = Browser()

agent = Agent(
	task=task,
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

---

## `browser-use-main\examples\use-cases\test_cv.txt`

```txt
123

```

---

## `browser-use-main\examples\use-cases\twitter_cookies.txt`

```txt
[{
    "name": "auth_token",
    "value": "auth_token_cookie_value",
    "domain": ".x.com",
    "path": "/"
  },
{
    "name": "ct0",
    "value": "ct0_cookie_value",
    "domain": ".x.com",
    "path": "/"
}]

```

---

## `browser-use-main\examples\use-cases\twitter_post_using_cookies.py`

```py
# Goal: Automates posting on X (Twitter) using stored authentication cookies.

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
	raise ValueError('GOOGLE_API_KEY is not set')

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))


browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		user_data_dir='~/.config/browseruse/profiles/default',
		# headless=False,  # Uncomment to see the browser
		# executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
	)
)


async def main():
	agent = Agent(
		browser_session=browser_session,
		task=('go to https://x.com. write a new post with the text "browser-use ftw", and submit it'),
		llm=llm,
		max_actions_per_step=4,
	)
	await agent.run(max_steps=25)
	input('Press Enter to close the browser...')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\web_voyager_agent.py`

```py
# Goal: A general-purpose web navigation agent for tasks like flight booking and course searching.

import asyncio
import os
import sys

# Adjust Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from browser_use.agent.service import Agent
from browser_use.browser import BrowserProfile, BrowserSession

# Set LLM based on defined environment variables
if os.getenv('OPENAI_API_KEY'):
	llm = ChatOpenAI(
		model='gpt-4o',
	)
elif os.getenv('AZURE_OPENAI_KEY') and os.getenv('AZURE_OPENAI_ENDPOINT'):
	llm = AzureChatOpenAI(
		model='gpt-4o',
		api_version='2024-10-21',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)
else:
	raise ValueError('No LLM found. Please set OPENAI_API_KEY or AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT.')


browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		headless=False,  # This is True in production
		disable_security=True,
		minimum_wait_page_load_time=1,  # 3 on prod
		maximum_wait_page_load_time=10,  # 20 on prod
		# Set no_viewport=False to constrain the viewport to the specified dimensions
		# This is useful for specific cases where you need a fixed viewport size
		no_viewport=False,
		window_width=1280,
		window_height=1100,
		user_data_dir='~/.config/browseruse/profiles/default',
		# trace_path='./tmp/web_voyager_agent',
	)
)

# TASK = """
# Find the lowest-priced one-way flight from Cairo to Montreal on February 21, 2025, including the total travel time and number of stops. on https://www.google.com/travel/flights/
# """
# TASK = """
# Browse Coursera, which universities offer Master of Advanced Study in Engineering degrees? Tell me what is the latest application deadline for this degree? on https://www.coursera.org/"""
TASK = """
Find and book a hotel in Paris with suitable accommodations for a family of four (two adults and two children) offering free cancellation for the dates of February 14-21, 2025. on https://www.booking.com/
"""


async def main():
	agent = Agent(
		task=TASK,
		llm=llm,
		browser_session=browser_session,
		validate_output=True,
		enable_memory=False,
	)
	history = await agent.run(max_steps=50)
	history.save_to_file('./tmp/history.json')


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\examples\use-cases\wikipedia_banana_to_quantum.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from browser_use import Agent
from browser_use.browser import BrowserProfile, BrowserSession

# video https://preview.screen.studio/share/vuq91Ej8
llm = ChatOpenAI(
	model='gpt-4o',
	temperature=0.0,
)
task = 'go to https://en.wikipedia.org/wiki/Banana and click on buttons on the wikipedia page to go as fast as possible from banna to Quantum mechanics'

browser_session = BrowserSession(
	browser_profile=BrowserProfile(
		viewport_expansion=-1,
		highlight_elements=False,
		user_data_dir='~/.config/browseruse/profiles/default',
	),
)
agent = Agent(task=task, llm=llm, browser_session=browser_session, use_vision=False)


async def main():
	await agent.run()


if __name__ == '__main__':
	asyncio.run(main())

```

---

## `browser-use-main\pyproject.toml`

```toml
[project]
name = "browser-use"
description = "Make websites accessible for AI agents"
authors = [{ name = "Gregor Zunic" }]
version = "0.2.1"
readme = "README.md"
requires-python = ">=3.11,<4.0"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "anyio>=4.9.0",
    "httpx>=0.27.2",
    "pydantic>=2.10.4,<2.11.0",
    "python-dotenv>=1.0.1",
    "requests>=2.32.3",
    "posthog>=3.7.0",
    "playwright>=1.52.0",
    "markdownify==1.1.0",
    "langchain-core==0.3.49",
    "langchain-openai==0.3.11",
    "langchain-anthropic==0.3.3",
    "langchain-ollama==0.3.0",
    "langchain-google-genai==2.1.2",
    "langchain-deepseek>=0.1.3",
    "langchain>=0.3.21",
    "langchain-aws>=0.2.11",
    "google-api-core>=2.24.0",
    "pyperclip>=1.9.0",
    "pyobjc>=11.0; platform_system == 'darwin'",
    "screeninfo>=0.8.1; platform_system != 'darwin'",
    "typing-extensions>=4.12.2",
    "psutil>=7.0.0",
    "faiss-cpu>=1.9.0",
    "mem0ai==0.1.93",
    "uuid7>=0.1.0",
    "patchright>=1.52.4",
]
# pydantic: >2.11 introduces many pydantic deprecation warnings until langchain-core upgrades their pydantic support lets keep it on 2.10
# google-api-core: only used for Google LLM APIs
# pyperclip: only used for examples that use copy/paste
# pyobjc: only used to get screen resolution on macOS
# screeninfo: only used to get screen resolution on Linux/Windows
# markdownify: used for page text content extraction for passing to LLM
# openai: datalib,voice-helpers are actually NOT NEEDED but openai produces noisy errors on exit without them TODO: fix
# rich: used for terminal formatting and styling in CLI
# click: used for command-line argument parsing
# textual: used for terminal UI

[project.optional-dependencies]
memory = [
    # sentence-transformers: depends on pytorch, which does not support python 3.13 yet
    "sentence-transformers>=4.0.2",
]
cli = [
    "rich>=14.0.0",
    "click>=8.1.8",
    "textual>=3.2.0",
]
examples = [
    # botocore: only needed for Bedrock Claude boto3 examples/models/bedrock_claude.py
    "botocore>=1.37.23",
    "imgcat>=0.6.0",
]
all = [
    "browser-use[memory,cli,examples]",
]

[project.urls]
Repository = "https://github.com/browser-use/browser-use"

[project.scripts]
browseruse = "browser_use.cli:main"
browser-use = "browser_use.cli:main"

[tool.codespell]
ignore-words-list = "bu"
skip = "*.json"

[tool.ruff]
line-length = 130
fix = true

[tool.ruff.lint]
select = ["ASYNC", "E", "F", "FAST", "I", "PLE"]
ignore = ["ASYNC109", "E101", "E402", "E501", "F841", "E731", "W291"]  # TODO: determine if adding timeouts to all the unbounded async functions is needed / worth-it so we can un-ignore ASYNC109
unfixable = ["E101", "E402", "E501", "F841", "E731"]

[tool.ruff.format]
quote-style = "single"
indent-style = "tab"
line-ending = "lf"
docstring-code-format = true
docstring-code-line-length = 140
skip-magic-trailing-comma = false

[tool.pyright]
typeCheckingMode = "off"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build]
include = [
    "browser_use/**/*.py",
    "!browser_use/**/tests/*.py",
    "!browser_use/**/tests.py",
    "browser_use/agent/system_prompt.md",
    "browser_use/dom/buildDomTree.js",
]

[tool.uv]
dev-dependencies = [
    "ruff>=0.11.2",
    "tokencost>=0.1.16",
    "build>=1.2.2",
    "pytest>=8.3.5",
    "pytest-asyncio>=0.24.0",
    "pytest-httpserver>=1.0.8",
    "fastapi>=0.115.8",
    "inngest>=0.4.19",
    "uvicorn>=0.34.0",
    "langchain-fireworks>=0.2.6",
    "ipdb>=0.13.13",
    "pre-commit>=4.2.0",
    "codespell>=2.4.1",
    "pyright>=1.1.399",
    "ty>=0.0.1a1",
]

```

---

## `browser-use-main\README.md`

```md
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./static/browser-use-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./static/browser-use.png">
  <img alt="Shows a black Browser Use Logo in light color mode and a white one in dark color mode." src="./static/browser-use.png"  width="full">
</picture>

<h1 align="center">Enable AI to control your browser 🤖</h1>

[![GitHub stars](https://img.shields.io/github/stars/gregpr07/browser-use?style=social)](https://github.com/gregpr07/browser-use/stargazers)
[![Discord](https://img.shields.io/discord/1303749220842340412?color=7289DA&label=Discord&logo=discord&logoColor=white)](https://link.browser-use.com/discord)
[![Cloud](https://img.shields.io/badge/Cloud-☁️-blue)](https://cloud.browser-use.com)
[![Documentation](https://img.shields.io/badge/Documentation-📕-blue)](https://docs.browser-use.com)
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/mamagnus00)
[![Weave Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fapp.workweave.ai%2Fapi%2Frepository%2Fbadge%2Forg_T5Pvn3UBswTHIsN1dWS3voPg%2F881458615&labelColor=#EC6341)](https://app.workweave.ai/reports/repository/org_T5Pvn3UBswTHIsN1dWS3voPg/881458615)

🌐 Browser-use is the easiest way to connect your AI agents with the browser.

💡 See what others are building and share your projects in our [Discord](https://link.browser-use.com/discord)! Want Swag? Check out our [Merch store](https://browsermerch.com).

🌤️ Skip the setup - try our <b>hosted version</b> for instant browser automation! <b>[Try the cloud ☁︎](https://cloud.browser-use.com)</b>.

# Quick start

With pip (Python>=3.11):

```bash
pip install browser-use
```

For memory functionality (requires Python<3.13 due to PyTorch compatibility):  

```bash
pip install "browser-use[memory]"
```

Install the browser:
```bash
playwright install chromium --with-deps --no-shell
```

Spin up your agent:

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()
from browser_use import Agent
from langchain_openai import ChatOpenAI

async def main():
    agent = Agent(
        task="Compare the price of gpt-4o and DeepSeek-V3",
        llm=ChatOpenAI(model="gpt-4o"),
    )
    await agent.run()

asyncio.run(main())
```

Add your API keys for the provider you want to use to your `.env` file.

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_KEY=
GOOGLE_API_KEY=
DEEPSEEK_API_KEY=
GROK_API_KEY=
NOVITA_API_KEY=
```

For other settings, models, and more, check out the [documentation 📕](https://docs.browser-use.com).

### Test with UI

You can test browser-use using its [Web UI](https://github.com/browser-use/web-ui) or [Desktop App](https://github.com/browser-use/desktop).

### Test with an interactive CLI

You can also use our `browser-use` interactive CLI (similar to `claude` code):

```bash
pip install browser-use[cli]
browser-use
```

# Demos

<br/><br/>

[Task](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/shopping.py): Add grocery items to cart, and checkout.

[![AI Did My Groceries](https://github.com/user-attachments/assets/a0ffd23d-9a11-4368-8893-b092703abc14)](https://www.youtube.com/watch?v=L2Ya9PYNns8)

<br/><br/>

Prompt: Add my latest LinkedIn follower to my leads in Salesforce.

![LinkedIn to Salesforce](https://github.com/user-attachments/assets/50d6e691-b66b-4077-a46c-49e9d4707e07)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/use-cases/find_and_apply_to_jobs.py): Read my CV & find ML jobs, save them to a file, and then start applying for them in new tabs, if you need help, ask me.'

https://github.com/user-attachments/assets/171fb4d6-0355-46f2-863e-edb04a828d04

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/browser/real_browser.py): Write a letter in Google Docs to my Papa, thanking him for everything, and save the document as a PDF.

![Letter to Papa](https://github.com/user-attachments/assets/242ade3e-15bc-41c2-988f-cbc5415a66aa)

<br/><br/>

[Prompt](https://github.com/browser-use/browser-use/blob/main/examples/custom-functions/save_to_file_hugging_face.py): Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.

https://github.com/user-attachments/assets/de73ee39-432c-4b97-b4e8-939fd7f323b3

<br/><br/>

## More examples

For more examples see the [examples](examples) folder or join the [Discord](https://link.browser-use.com/discord) and show off your project. You can also see our [`awesome-prompts`](https://github.com/browser-use/awesome-prompts) repo for prompting inspiration.

# Vision

Tell your computer what to do, and it gets it done.

## Roadmap

### Agent

- [ ] Improve agent memory to handle +100 steps
- [ ] Enhance planning capabilities (load website specific context)
- [ ] Reduce token consumption (system prompt, DOM state)

### DOM Extraction

- [ ] Enable detection for all possible UI elements
- [ ] Improve state representation for UI elements so that all LLMs can understand what's on the page

### Workflows

- [ ] Let user record a workflow - which we can rerun with browser-use as a fallback
- [ ] Make rerunning of workflows work, even if pages change

### User Experience

- [ ] Create various templates for tutorial execution, job application, QA testing, social media, etc. which users can just copy & paste.
- [ ] Improve docs
- [ ] Make it faster

### Parallelization

- [ ] Human work is sequential. The real power of a browser agent comes into reality if we can parallelize similar tasks. For example, if you want to find contact information for 100 companies, this can all be done in parallel and reported back to a main agent, which processes the results and kicks off parallel subtasks again.


## Contributing

We love contributions! Feel free to open issues for bugs or feature requests. To contribute to the docs, check out the `/docs` folder.

## Local Setup

To learn more about the library, check out the [local setup 📕](https://docs.browser-use.com/development/local-setup).


`main` is the primary development branch with frequent changes. For production use, install a stable [versioned release](https://github.com/browser-use/browser-use/releases) instead.

---

## Swag

Want to show off your Browser-use swag? Check out our [Merch store](https://browsermerch.com). Good contributors will receive swag for free 👀.

## Citation

If you use Browser Use in your research or project, please cite:

```bibtex
@software{browser_use2024,
  author = {Müller, Magnus and Žunič, Gregor},
  title = {Browser Use: Enable AI to control your browser},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/browser-use/browser-use}
}
```

 <div align="center"> <img src="https://github.com/user-attachments/assets/06fa3078-8461-4560-b434-445510c1766f" width="400"/> 
 
[![Twitter Follow](https://img.shields.io/twitter/follow/Gregor?style=social)](https://x.com/gregpr07)
[![Twitter Follow](https://img.shields.io/twitter/follow/Magnus?style=social)](https://x.com/mamagnus00)
 
 </div>

<div align="center">
Made with ❤️ in Zurich and San Francisco
 </div>

```

---

## `browser-use-main\SECURITY.md`

```md
## Reporting Security Issues

If you believe you have found a security vulnerability in browser-use, please report it through coordinated disclosure.

**Please do not report security vulnerabilities through the repository issues, discussions, or pull requests.**

Instead, please open a new [Github security advisory](https://github.com/browser-use/browser-use/security/advisories/new).

Please include as much of the information listed below as you can to help me better understand and resolve the issue:

* The type of issue (e.g., buffer overflow, SQL injection, or cross-site scripting)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit the issue

This information will help me triage your report more quickly.

```

---

## `browser-use-main\tests\debug_page_structure.py`

```py
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext


async def analyze_page_structure(url: str):
	"""Analyze and print the structure of a webpage with enhanced debugging"""
	browser = Browser(
		config=BrowserConfig(
			headless=False,  # Set to True if you don't need to see the browser
		),
		user_data_dir=None,
	)

	context = BrowserContext(browser=browser)

	try:
		async with context as ctx:
			# Navigate to the URL
			page = await ctx.get_current_page()
			await page.goto(url)
			await page.wait_for_load_state('networkidle')

			# Get viewport dimensions
			viewport_info = await page.evaluate("""() => {
				return {
					viewport: {
						width: window.innerWidth,
						height: window.innerHeight,
						scrollX: window.scrollX,
						scrollY: window.scrollY
					}
				}
			}""")

			print('\nViewport Information:')
			print(f'Width: {viewport_info["viewport"]["width"]}')
			print(f'Height: {viewport_info["viewport"]["height"]}')
			print(f'ScrollX: {viewport_info["viewport"]["scrollX"]}')
			print(f'ScrollY: {viewport_info["viewport"]["scrollY"]}')

			# Enhanced debug information for cookie consent and fixed position elements
			debug_info = await page.evaluate("""() => {
				function getElementInfo(element) {
					const rect = element.getBoundingClientRect();
					const style = window.getComputedStyle(element);
					return {
						tag: element.tagName.toLowerCase(),
						id: element.id,
						className: element.className,
						position: style.position,
						rect: {
							top: rect.top,
							right: rect.right,
							bottom: rect.bottom,
							left: rect.left,
							width: rect.width,
							height: rect.height
						},
						isFixed: style.position === 'fixed',
						isSticky: style.position === 'sticky',
						zIndex: style.zIndex,
						visibility: style.visibility,
						display: style.display,
						opacity: style.opacity
					};
				}

				// Find cookie-related elements
				const cookieElements = Array.from(document.querySelectorAll('[id*="cookie"], [id*="consent"], [class*="cookie"], [class*="consent"]'));
				const fixedElements = Array.from(document.querySelectorAll('*')).filter(el => {
					const style = window.getComputedStyle(el);
					return style.position === 'fixed' || style.position === 'sticky';
				});

				return {
					cookieElements: cookieElements.map(getElementInfo),
					fixedElements: fixedElements.map(getElementInfo)
				};
			}""")

			print('\nCookie-related Elements:')
			for elem in debug_info['cookieElements']:
				print(f'\nElement: {elem["tag"]}#{elem["id"]} .{elem["className"]}')
				print(f'Position: {elem["position"]}')
				print(f'Rect: {elem["rect"]}')
				print(f'Z-Index: {elem["zIndex"]}')
				print(f'Visibility: {elem["visibility"]}')
				print(f'Display: {elem["display"]}')
				print(f'Opacity: {elem["opacity"]}')

			print('\nFixed/Sticky Position Elements:')
			for elem in debug_info['fixedElements']:
				print(f'\nElement: {elem["tag"]}#{elem["id"]} .{elem["className"]}')
				print(f'Position: {elem["position"]}')
				print(f'Rect: {elem["rect"]}')
				print(f'Z-Index: {elem["zIndex"]}')

			print(f'\nPage Structure for {url}:\n')
			structure = await ctx.get_page_structure()
			print(structure)

			input('Press Enter to close the browser...')
	finally:
		await browser.close()


if __name__ == '__main__':
	# You can modify this URL to analyze different pages

	urls = [
		'https://www.mlb.com/yankees/stats/',
		'https://immobilienscout24.de',
		'https://www.zeiss.com/career/en/job-search.html?page=1',
		'https://www.zeiss.com/career/en/job-search.html?page=1',
		'https://reddit.com',
	]
	for url in urls:
		asyncio.run(analyze_page_structure(url))

```

---

## `browser-use-main\tests\extraction_test.py`

```py
import asyncio
import os

import anyio
from langchain_openai import ChatOpenAI

from browser_use.agent.prompts import AgentMessagePrompt
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService


def count_string_tokens(string: str, model: str) -> tuple[int, float]:
	"""Count the number of tokens in a string using a specified model."""

	def get_price_per_token(model: str) -> float:
		"""Get the price per token for a specified model.

		@todo: move to utils, use a package or sth
		"""
		prices = {
			'gpt-4o': 2.5 / 1e6,
			'gpt-4o-mini': 0.15 / 1e6,
		}
		return prices[model]

	llm = ChatOpenAI(model=model)
	token_count = llm.get_num_tokens(string)
	price = token_count * get_price_per_token(model)
	return token_count, price


TIMEOUT = 60

DEFAULT_INCLUDE_ATTRIBUTES = [
	'id',
	'title',
	'type',
	'name',
	'role',
	'aria-label',
	'placeholder',
	'value',
	'alt',
	'aria-expanded',
	'data-date-format',
]


async def test_focus_vs_all_elements():
	config = BrowserContextConfig(
		# cookies_file='cookies3.json',
		disable_security=True,
		wait_for_network_idle_page_load_time=1,
	)

	browser = Browser(
		config=BrowserConfig(
			# browser_binary_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
		)
	)
	context = BrowserContext(browser=browser, config=config)

	websites = [
		'https://demos.telerik.com/kendo-react-ui/treeview/overview/basic/func?theme=default-ocean-blue-a11y',
		'https://www.ycombinator.com/companies',
		'https://kayak.com/flights',
		# 'https://en.wikipedia.org/wiki/Humanist_Party_of_Ontario',
		# 'https://www.google.com/travel/flights?tfs=CBwQARoJagcIARIDTEpVGglyBwgBEgNMSlVAAUgBcAGCAQsI____________AZgBAQ&tfu=KgIIAw&hl=en-US&gl=US',
		# # 'https://www.concur.com/?&cookie_preferences=cpra',
		# 'https://immobilienscout24.de',
		'https://docs.google.com/spreadsheets/d/1INaIcfpYXlMRWO__de61SHFCaqt1lfHlcvtXZPItlpI/edit',
		'https://www.zeiss.com/career/en/job-search.html?page=1',
		'https://www.mlb.com/yankees/stats/',
		'https://www.amazon.com/s?k=laptop&s=review-rank&crid=1RZCEJ289EUSI&qid=1740202453&sprefix=laptop%2Caps%2C166&ref=sr_st_review-rank&ds=v1%3A4EnYKXVQA7DIE41qCvRZoNB4qN92Jlztd3BPsTFXmxU',
		'https://reddit.com',
		'https://codepen.io/geheimschriftstift/pen/mPLvQz',
		'https://www.google.com/search?q=google+hi&oq=google+hi&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIMjI2NmowajSoAgCwAgE&sourceid=chrome&ie=UTF-8',
		'https://google.com',
		'https://amazon.com',
		'https://github.com',
	]

	async with context as context:
		page = await context.get_current_page()
		dom_service = DomService(page)

		for website in websites:
			# sleep 2
			await page.goto(website)
			asyncio.sleep(1)

			last_clicked_index = None  # Track the index for text input
			while True:
				try:
					print(f'\n{"=" * 50}\nTesting {website}\n{"=" * 50}')

					# Get/refresh the state (includes removing old highlights)
					print('\nGetting page state...')
					all_elements_state = await context.get_state_summary(True)

					selector_map = all_elements_state.selector_map
					total_elements = len(selector_map.keys())
					print(f'Total number of elements: {total_elements}')

					# print(all_elements_state.element_tree.clickable_elements_to_string())
					prompt = AgentMessagePrompt(
						browser_state_summary=all_elements_state,
						result=None,
						include_attributes=DEFAULT_INCLUDE_ATTRIBUTES,
						step_info=None,
					)
					# print(prompt.get_user_message(use_vision=False).content)
					# Write the user message to a file for analysis
					user_message = prompt.get_user_message(use_vision=False).content
					os.makedirs('./tmp', exist_ok=True)
					async with await anyio.open_file('./tmp/user_message.txt', 'w', encoding='utf-8') as f:
						await f.write(user_message)

					token_count, price = count_string_tokens(user_message, model='gpt-4o')
					print(f'Prompt token count: {token_count}, price: {round(price, 4)} USD')
					print('User message written to ./tmp/user_message.txt')

					# also save all_elements_state.element_tree.clickable_elements_to_string() to a file
					# with open('./tmp/clickable_elements.json', 'w', encoding='utf-8') as f:
					# 	f.write(json.dumps(all_elements_state.element_tree.__json__(), indent=2))
					# print('Clickable elements written to ./tmp/clickable_elements.json')

					answer = input("Enter element index to click, 'index,text' to input, or 'q' to quit: ")

					if answer.lower() == 'q':
						break

					try:
						if ',' in answer:
							# Input text format: index,text
							parts = answer.split(',', 1)
							if len(parts) == 2:
								try:
									target_index = int(parts[0].strip())
									text_to_input = parts[1]
									if target_index in selector_map:
										element_node = selector_map[target_index]
										print(
											f"Inputting text '{text_to_input}' into element {target_index}: {element_node.tag_name}"
										)
										await context._input_text_element_node(element_node, text_to_input)
										print('Input successful.')
									else:
										print(f'Invalid index: {target_index}')
								except ValueError:
									print(f'Invalid index format: {parts[0]}')
							else:
								print("Invalid input format. Use 'index,text'.")
						else:
							# Click element format: index
							try:
								clicked_index = int(answer)
								if clicked_index in selector_map:
									element_node = selector_map[clicked_index]
									print(f'Clicking element {clicked_index}: {element_node.tag_name}')
									await context._click_element_node(element_node)
									print('Click successful.')
								else:
									print(f'Invalid index: {clicked_index}')
							except ValueError:
								print(f"Invalid input: '{answer}'. Enter an index, 'index,text', or 'q'.")

					except Exception as action_e:
						print(f'Action failed: {action_e}')

				# No explicit highlight removal here, get_state handles it at the start of the loop

				except Exception as e:
					print(f'Error in loop: {e}')
					# Optionally add a small delay before retrying
					await asyncio.sleep(1)


if __name__ == '__main__':
	asyncio.run(test_focus_vs_all_elements())
	# asyncio.run(test_process_html_file()) # Commented out the other test

```

---

## `browser-use-main\tests\httpx_client_test.py`

```py
import httpx
import pytest

from browser_use.browser.browser import Browser, BrowserConfig


@pytest.mark.asyncio
async def test_browser_close_doesnt_affect_external_httpx_clients():
	"""
	Test that Browser.close() doesn't close HTTPX clients created outside the Browser instance.
	This test demonstrates the issue where Browser.close() is closing all HTTPX clients.
	"""
	# Create an external HTTPX client that should remain open
	external_client = httpx.AsyncClient()

	# Create a Browser instance
	browser = Browser(config=BrowserConfig(headless=True))

	# Close the browser (which should trigger cleanup_httpx_clients)
	await browser.close()

	# Check if the external client is still usable
	try:
		# If the client is closed, this will raise RuntimeError
		# Using a simple HEAD request to a reliable URL
		await external_client.head('https://www.example.com', timeout=2.0)
		client_is_closed = False
	except RuntimeError as e:
		# If we get "Cannot send a request, as the client has been closed"
		client_is_closed = 'client has been closed' in str(e)
	except Exception:
		# Any other exception means the client is not closed but request failed
		client_is_closed = False
	finally:
		# Always clean up our test client properly
		await external_client.aclose()

	# Our external client should not be closed by browser.close()
	assert not client_is_closed, 'External HTTPX client was incorrectly closed by Browser.close()'

```

---

## `browser-use-main\tests\mind2web_data\processed.json`

```json
[
    {
        "website": "exploretock",
        "id": "7bda9645-0b5f-470a-8dd7-6af0bff4da68",
        "domain": "Travel",
        "subdomain": "Restaurant",
        "confirmed_task": "Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest",
        "action_reprs": [
            "[combobox]  Reservation type -> SELECT: Pickup",
            "[svg]   -> CLICK",
            "[searchbox]  Find a location -> TYPE: Boston",
            "[span]  Boston -> CLICK",
            "[svg]   -> CLICK",
            "[button]  18 -> CLICK",
            "[combobox]  Time -> SELECT: 5:00 PM",
            "[svg]   -> CLICK",
            "[span]  2 guests -> CLICK",
            "[combobox]  Size -> SELECT: 1 guest",
            "[button]  Update search -> CLICK"
        ]
    },
    {
        "website": "exploretock",
        "id": "a6372f23-f462-4706-8455-5b350c46d83c",
        "domain": "Travel",
        "subdomain": "Restaurant",
        "confirmed_task": "Book a winery tour in Napa Valley in a winery which serves Mediterranean cuisine with wine testing for 4 guests on April 15, 10 am in a outdoor setup.",
        "action_reprs": [
            "[svg]   -> CLICK",
            "[svg]   -> CLICK",
            "[searchbox]  Find a location -> TYPE: NAPA VALLEY",
            "[span]  Napa Valley -> CLICK",
            "[combobox]  Reservation type -> SELECT: Wineries",
            "[svg]   -> CLICK",
            "[svg]   -> CLICK",
            "[button]  15 -> CLICK",
            "[combobox]  Time -> SELECT: 10:00 AM",
            "[combobox]  Party size -> SELECT: 4 guests",
            "[svg]   -> CLICK",
            "[button]  Edit cuisine type filter -> CLICK",
            "[checkbox]  Mediterranean -> CLICK",
            "[button]  Submit -> CLICK",
            "[button]  Open additional search filters -> CLICK",
            "[checkbox]  Outdoors -> CLICK",
            "[checkbox]  Wine tasting -> CLICK",
            "[button]  Update search -> CLICK",
            "[span]  10:00 AM -> CLICK"
        ]
    },
    {
        "website": "enterprise",
        "id": "c0eeead1-f8ea-4819-a6da-ef0108b40c89",
        "domain": "Travel",
        "subdomain": "Car rental",
        "confirmed_task": "Sign Allan Smith for email offers with the email allan.smith@gmail.com and zip code 10001",
        "action_reprs": [
            "[a]   -> CLICK",
            "[textbox]  First Name (required) -> TYPE: Allan",
            "[textbox]  Last Name (required) -> TYPE: Smith",
            "[textbox]  Email Address (required) -> TYPE: allan.smith@gmail.com",
            "[textbox]  Confirm Email Address (required) -> TYPE: allan.smith@gmail.com",
            "[textbox]  ZIP Code (required) -> TYPE: 10001",
            "[button]  Submit -> CLICK"
        ]
    },
    {
        "website": "enterprise",
        "id": "ce34bc61-e3d2-40c8-b02b-b149efc4b115",
        "domain": "Travel",
        "subdomain": "Car rental",
        "confirmed_task": "Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.",
        "action_reprs": [
            "[searchbox]  Pick-up & Return Location (ZIP, City or Airport) (... -> TYPE: Brooklyn",
            "[option]  Brooklyn, NY, US Select -> CLICK",
            "[button]  Selected Pick-Up Date 03/19/2023 -> CLICK",
            "[button]  04/05/2023 -> CLICK",
            "[svg]   -> CLICK",
            "[button]  04/08/2023 -> CLICK",
            "[combobox]  Renter Age -> SELECT: 22",
            "[button]  Vehicle Class -> CLICK",
            "[radio]  Minivans -> CLICK",
            "[button]  Apply Filter -> CLICK",
            "[button]  Browse Vehicles -> CLICK"
        ]
    },
    {
        "website": "kohls",
        "id": "bf469f30-6628-4017-b963-672645d7feab",
        "domain": "Shopping",
        "subdomain": "Department",
        "confirmed_task": "Find the highest rated dog collar under 10 dollar.",
        "action_reprs": [
            "[link]  Shop by Category -> CLICK",
            "[link]  Pet -> CLICK",
            "[span]  Pet Type -> CLICK",
            "[link]  Dog (1,338) -> CLICK",
            "[span]  Category -> CLICK",
            "[link]  Collars & Leashes (485) -> CLICK",
            "[link]  Sort by: Featured -> CLICK",
            "[link]  Highest Rated -> CLICK",
            "[div]  Price -> CLICK",
            "[link]  Under $10 (1) -> CLICK"
        ]
    },
    {
        "website": "united",
        "id": "9e035a36-1c77-4014-98ec-4d48ee41d904",
        "domain": "Travel",
        "subdomain": "Airlines",
        "confirmed_task": "Compare the fare types to book a 1-adult ticket from Springfiels, IL to Austin, TX for April 29th 2023",
        "action_reprs": [
            "[combobox]  Enter your departing city, airport name, or airpor... -> TYPE: SPRINGFIELD",
            "[button]  Springfield, IL, US (SPI) -> CLICK",
            "[combobox]  Enter your destination city, airport name, or airp... -> TYPE: AUSTIN",
            "[button]  Austin, TX, US (AUS) -> CLICK",
            "[span]   -> CLICK",
            "[button]  Find flights -> CLICK",
            "[textbox]  Date -> CLICK",
            "[button]  Move backward to switch to the previous month. -> CLICK",
            "[button]  Saturday, April 29, 2023 -> CLICK",
            "[button]  Update -> CLICK",
            "[link]  Details -> CLICK",
            "[link]  Seats -> CLICK"
        ]
    },
    {
        "website": "budget",
        "id": "cf361c84-6414-4b05-a7a1-77383997150a",
        "domain": "Travel",
        "subdomain": "Car rental",
        "confirmed_task": "Get an SUV with an additional driver and wifi for pick up in  any rental location near Washington regional airport on June 1, 11 am, and drop off at Washington international airport on June 2, 11 am, and pay for the booking instantly.",
        "action_reprs": [
            "[button]  Locations -> HOVER",
            "[link]  Find a Location -> CLICK",
            "[textbox]  Search by Airport, City, Zip, Address or Attractio... -> TYPE: washington",
            "[span]  Washington County Regional Apo -> CLICK",
            "[link]  Make a Reservation -> CLICK",
            "[textbox]  mm/dd/yyyy -> CLICK",
            "[link]  Next -> CLICK",
            "[link]  1 -> CLICK",
            "[link]  2 -> CLICK",
            "[combobox]  Pick Up Time -> SELECT: 11:00 AM",
            "[combobox]  Return Time -> SELECT: 11:00 AM",
            "[textbox]  Return to same location -> TYPE: washington",
            "[div]  Washington Dulles Intl Airport -> CLICK",
            "[generic]  Vehicle Type * -> CLICK",
            "[p]  SUVs & Wagons -> CLICK",
            "[button]  Select My Car -> CLICK",
            "[link]  Pay Now -> CLICK",
            "[checkbox]  $21.99/Day -> CLICK",
            "[checkbox]  $13.00/Day -> CLICK",
            "[button]  Continue -> CLICK"
        ]
    },
    {
        "website": "underarmour",
        "id": "46a3683f-fbe0-40d0-8729-6c7964d994e6",
        "domain": "Shopping",
        "subdomain": "Fashion",
        "confirmed_task": "Find a men's UA outlet T-shirt of XL size and add to cart.",
        "action_reprs": [
            "[menuitem]  Outlet -> CLICK",
            "[link]  Mens -> CLICK",
            "[div]  Product Category -> CLICK",
            "[link]  Clothing -> CLICK",
            "[div]  Product Type -> CLICK",
            "[link]  Short Sleeves -> CLICK",
            "[div]  Size -> CLICK",
            "[link]  XL -> CLICK",
            "[img]  Men's UA Tech\u2122 2.0 Short Sleeve -> CLICK",
            "[button]  XL -> CLICK",
            "[button]  Add to Bag -> CLICK"
        ]
    },
    {
        "website": "kohls",
        "id": "4b2030ff-b83c-445f-bf87-9c8fbc68498b",
        "domain": "Shopping",
        "subdomain": "Department",
        "confirmed_task": "Browse for wall art with a price range of $25 to $50.",
        "action_reprs": [
            "[textbox]  Search by keyword o
… (truncated)

```

---

## `browser-use-main\tests\process_dom_test.py`

```py
import asyncio
import json
import os
import time

import anyio

from browser_use.browser.browser import Browser, BrowserConfig


async def test_process_dom():
	browser = Browser(config=BrowserConfig(headless=False))

	async with await browser.new_context() as context:
		page = await context.get_current_page()
		await page.goto('https://kayak.com/flights')
		# await page.goto('https://google.com/flights')
		# await page.goto('https://immobilienscout24.de')
		# await page.goto('https://seleniumbase.io/w3schools/iframes')

		await asyncio.sleep(3)

		async with await anyio.open_file('browser_use/dom/buildDomTree.js', 'r') as f:
			js_code = await f.read()

		start = time.time()
		dom_tree = await page.evaluate(js_code)
		end = time.time()

		# print(dom_tree)
		print(f'Time: {end - start:.2f}s')

		os.makedirs('./tmp', exist_ok=True)
		async with await anyio.open_file('./tmp/dom.json', 'w') as f:
			await f.write(json.dumps(dom_tree, indent=1))

		# both of these work for immobilienscout24.de
		# await page.click('.sc-dcJsrY.ezjNCe')
		# await page.click(
		# 	'div > div:nth-of-type(2) > div > div:nth-of-type(2) > div > div:nth-of-type(2) > div > div > div > button:nth-of-type(2)'
		# )

		input('Press Enter to continue...')

```

---

## `browser-use-main\tests\screenshot_test.py`

```py
import asyncio
import base64

import pytest

from browser_use.browser.browser import Browser, BrowserConfig


async def test_take_full_page_screenshot():
	browser = Browser(config=BrowserConfig(headless=False, disable_security=True))
	try:
		async with await browser.new_context() as context:
			page = await context.get_current_page()
			# Go to a test page
			await page.goto('https://example.com')

			await asyncio.sleep(3)
			# Take full page screenshot
			screenshot_b64 = await context.take_screenshot(full_page=True)
			await asyncio.sleep(3)
			# Verify screenshot is not empty and is valid base64
			assert screenshot_b64 is not None
			assert isinstance(screenshot_b64, str)
			assert len(screenshot_b64) > 0

			# Test we can decode the base64 string
			try:
				base64.b64decode(screenshot_b64)
			except Exception as e:
				pytest.fail(f'Failed to decode base64 screenshot: {str(e)}')
	finally:
		await browser.close()


if __name__ == '__main__':
	asyncio.run(test_take_full_page_screenshot())

```

---

## `browser-use-main\tests\test_action_filters.py`

```py
from unittest.mock import MagicMock

import pytest
from playwright.async_api import Page
from pydantic import BaseModel

from browser_use.controller.registry.service import Registry
from browser_use.controller.registry.views import ActionRegistry, RegisteredAction


class EmptyParamModel(BaseModel):
	pass


class TestActionFilters:
	def test_get_prompt_description_no_filters(self):
		"""Test that system prompt only includes actions with no filters"""
		registry = ActionRegistry()

		# Add actions with and without filters
		no_filter_action = RegisteredAction(
			name='no_filter_action',
			description='Action with no filters',
			function=lambda: None,
			param_model=EmptyParamModel,
			domains=None,
			page_filter=None,
		)

		page_filter_action = RegisteredAction(
			name='page_filter_action',
			description='Action with page filter',
			function=lambda: None,
			param_model=EmptyParamModel,
			domains=None,
			page_filter=lambda page: True,
		)

		domain_filter_action = RegisteredAction(
			name='domain_filter_action',
			description='Action with domain filter',
			function=lambda: None,
			param_model=EmptyParamModel,
			domains=['example.com'],
			page_filter=None,
		)

		registry.actions = {
			'no_filter_action': no_filter_action,
			'page_filter_action': page_filter_action,
			'domain_filter_action': domain_filter_action,
		}

		# System prompt (no page) should only include actions with no filters
		system_description = registry.get_prompt_description()
		assert 'no_filter_action' in system_description
		assert 'page_filter_action' not in system_description
		assert 'domain_filter_action' not in system_description

	def test_page_filter_matching(self):
		"""Test that page filters work correctly"""
		registry = ActionRegistry()

		# Create a mock page
		mock_page = MagicMock(spec=Page)
		mock_page.url = 'https://example.com/page'

		# Create actions with different page filters
		matching_action = RegisteredAction(
			name='matching_action',
			description='Action with matching page filter',
			function=lambda: None,
			param_model=EmptyParamModel,
			domains=None,
			page_filter=lambda page: 'example.com' in page.url,
		)

		non_matching_action = RegisteredAction(
			name='non_matching_action',
			description='Action with non-matching page filter',
			function=lambda: None,
			param_model=EmptyParamModel,
			domains=None,
			page_filter=lambda page: 'other.com' in page.url,
		)

		registry.actions = {'matching_action': matching_action, 'non_matching_action': non_matching_action}

		# Page-specific description should only include matching actions
		page_description = registry.get_prompt_description(mock_page)
		assert 'matching_action' in page_description
		assert 'non_matching_action' not in page_description

	def test_domain_filter_matching(self):
		"""Test that domain filters work correctly with glob patterns"""
		registry = ActionRegistry()

		# Create actions with different domain patterns
		actions = {
			'exact_match': RegisteredAction(
				name='exact_match',
				description='Exact domain match',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['example.com'],
				page_filter=None,
			),
			'subdomain_match': RegisteredAction(
				name='subdomain_match',
				description='Subdomain wildcard match',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['*.example.com'],
				page_filter=None,
			),
			'prefix_match': RegisteredAction(
				name='prefix_match',
				description='Prefix wildcard match',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['example*'],
				page_filter=None,
			),
			'non_matching': RegisteredAction(
				name='non_matching',
				description='Non-matching domain',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['other.com'],
				page_filter=None,
			),
		}

		registry.actions = actions

		# Test exact domain match
		mock_page = MagicMock(spec=Page)
		mock_page.url = 'https://example.com/page'

		exact_match_description = registry.get_prompt_description(mock_page)
		assert 'exact_match' in exact_match_description
		assert 'non_matching' not in exact_match_description

		# Test subdomain match
		mock_page.url = 'https://sub.example.com/page'
		subdomain_match_description = registry.get_prompt_description(mock_page)
		assert 'subdomain_match' in subdomain_match_description
		assert 'exact_match' not in subdomain_match_description

		# Test prefix match
		mock_page.url = 'https://example123.org/page'
		prefix_match_description = registry.get_prompt_description(mock_page)
		assert 'prefix_match' in prefix_match_description

	def test_domain_and_page_filter_together(self):
		"""Test that actions can be filtered by both domain and page filter"""
		registry = ActionRegistry()

		# Create a mock page
		mock_page = MagicMock(spec=Page)
		mock_page.url = 'https://example.com/admin'

		# Actions with different combinations of filters
		actions = {
			'domain_only': RegisteredAction(
				name='domain_only',
				description='Domain filter only',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['example.com'],
				page_filter=None,
			),
			'page_only': RegisteredAction(
				name='page_only',
				description='Page filter only',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=None,
				page_filter=lambda page: 'admin' in page.url,
			),
			'both_matching': RegisteredAction(
				name='both_matching',
				description='Both filters matching',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['example.com'],
				page_filter=lambda page: 'admin' in page.url,
			),
			'both_one_fail': RegisteredAction(
				name='both_one_fail',
				description='One filter fails',
				function=lambda: None,
				param_model=EmptyParamModel,
				domains=['other.com'],
				page_filter=lambda page: 'admin' in page.url,
			),
		}

		registry.actions = actions

… (truncated)

```

---

## `browser-use-main\tests\test_agent_actions.py`

```py
import asyncio
import os

import pytest
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig


@pytest.fixture
def llm():
	"""Initialize language model for testing"""

	# return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None)
	return AzureChatOpenAI(
		model='gpt-4o',
		api_version='2024-10-21',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)
	# return ChatOpenAI(model='gpt-4o-mini')


@pytest.fixture(scope='session')
def event_loop():
	"""Create an instance of the default event loop for each test case."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='session')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as context:
		yield context
		# Clean up automatically happens with __aexit__


# pytest tests/test_agent_actions.py -v -k "test_ecommerce_interaction" --capture=no
# @pytest.mark.asyncio
@pytest.mark.skip(reason='Kinda expensive to run')
async def test_ecommerce_interaction(llm, context):
	"""Test complex ecommerce interaction sequence"""
	agent = Agent(
		task="Go to amazon.com, search for 'laptop', filter by 4+ stars, and find the price of the first result",
		llm=llm,
		browser_context=context,
		save_conversation_path='tmp/test_ecommerce_interaction/conversation',
	)

	history: AgentHistoryList = await agent.run(max_steps=20)

	# Verify sequence of actions
	action_sequence = []
	for action in history.model_actions():
		action_name = list(action.keys())[0]
		if action_name in ['go_to_url', 'open_tab']:
			action_sequence.append('navigate')
		elif action_name == 'input_text':
			action_sequence.append('input')
			# Check that the input is 'laptop'
			inp = action['input_text']['text'].lower()  # type: ignore
			if inp == 'laptop':
				action_sequence.append('input_exact_correct')
			elif 'laptop' in inp:
				action_sequence.append('correct_in_input')
			else:
				action_sequence.append('incorrect_input')
		elif action_name == 'click_element':
			action_sequence.append('click')

	# Verify essential steps were performed
	assert 'navigate' in action_sequence  # Navigated to Amazon
	assert 'input' in action_sequence  # Entered search term
	assert 'click' in action_sequence  # Clicked search/filter
	assert 'input_exact_correct' in action_sequence or 'correct_in_input' in action_sequence


# @pytest.mark.asyncio
async def test_error_recovery(llm, context):
	"""Test agent's ability to recover from errors"""
	agent = Agent(
		task='Navigate to nonexistent-site.com and then recover by going to google.com ',
		llm=llm,
		browser_context=context,
	)

	history: AgentHistoryList = await agent.run(max_steps=10)

	actions_names = history.action_names()
	actions = history.model_actions()
	assert 'go_to_url' in actions_names or 'open_tab' in actions_names, f'{actions_names} does not contain go_to_url or open_tab'
	for action in actions:
		if 'go_to_url' in action:
			assert 'url' in action['go_to_url'], 'url is not in go_to_url'
			assert action['go_to_url']['url'].endswith('google.com'), 'url does not end with google.com'
			break


# @pytest.mark.asyncio
async def test_find_contact_email(llm, context):
	"""Test agent's ability to find contact email on a website"""
	agent = Agent(
		task='Go to https://browser-use.com/ and find out the contact email',
		llm=llm,
		browser_context=context,
	)

	history: AgentHistoryList = await agent.run(max_steps=10)

	# Verify the agent found the contact email
	extracted_content = history.extracted_content()
	email = 'info@browser-use.com'
	for content in extracted_content:
		if email in content:
			break
	else:
		pytest.fail(f'{extracted_content} does not contain {email}')


# @pytest.mark.asyncio
async def test_agent_finds_installation_command(llm, context):
	"""Test agent's ability to find the pip installation command for browser-use on the web"""
	agent = Agent(
		task='Find the pip installation command for the browser-use repo',
		llm=llm,
		browser_context=context,
	)

	history: AgentHistoryList = await agent.run(max_steps=10)

	# Verify the agent found the correct installation command
	extracted_content = history.extracted_content()
	install_command = 'pip install browser-use'
	for content in extracted_content:
		if install_command in content:
			break
	else:
		pytest.fail(f'{extracted_content} does not contain {install_command}')


class CaptchaTest(BaseModel):
	name: str
	url: str
	success_text: str
	additional_text: str | None = None


# run 3 test: python -m pytest tests/test_agent_actions.py -v -k "test_captcha_solver" --capture=no --log-cli-level=INFO
# pytest tests/test_agent_actions.py -v -k "test_captcha_solver" --capture=no --log-cli-level=INFO
@pytest.mark.asyncio
@pytest.mark.parametrize(
	'captcha',
	[
		CaptchaTest(
			name='Text Captcha',
			url='https://2captcha.com/demo/text',
			success_text='Captcha is passed successfully!',
		),
		CaptchaTest(
			name='Basic Captcha',
			url='https://captcha.com/demos/features/captcha-demo.aspx',
			success_text='Correct!',
		),
		CaptchaTest(
			name='Rotate Captcha',
			url='https://2captcha.com/demo/rotatecaptcha',
			success_text='Captcha is passed successfully',
			additional_text='Use multiple clicks at once. click done when image is exact correct position.',
		),
		CaptchaTest(
			name='MT Captcha',
			url='https://2captcha.com/demo/mtcaptcha',
			success_text='Verified Successfully',
			additional_text='Stop when you solved it successfully.',
		),
	],
)
async def test_captcha_solver(llm, context, captcha: CaptchaTest):
	"""Test agent's ability to solve different types of captchas"""
	agent = Agent(
		task=f'Go to {captcha.url} and solve the captcha. {captcha.additional_text}',
		llm=llm,
		browser_context=context,
	)
	from browser_use.agent.views import AgentHistoryList
… (truncated)

```

---

## `browser-use-main\tests\test_attach_chrome.py`

```py
import asyncio

from playwright.async_api import async_playwright


async def test_full_screen(start_fullscreen: bool, maximize: bool):
	async with async_playwright() as p:
		try:
			print('Attempting to connect to Chrome...')
			# run in terminal: /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --no-first-run
			browser = await p.chromium.connect_over_cdp(
				'http://localhost:9222',
				timeout=20000,  # 20 second timeout for connection
			)
			print('Connected to Chrome successfully')

			# Get the first context and page, or create new ones if needed
			if len(browser.contexts) == 0:
				context = await browser.new_context(ignore_https_errors=True)
			else:
				context = browser.contexts[0]

			if len(context.pages) == 0:
				page = await context.new_page()
			else:
				page = context.pages[0]

			print('Attempting to navigate to Gmail...')
			try:
				# First try with a shorter timeout
				await page.goto(
					'https://mail.google.com',
					wait_until='load',  # Changed from domcontentloaded
					timeout=10000,
				)
			except Exception as e:
				print(f'First navigation attempt failed: {e}')
				print('Trying again with different settings...')
				# If that fails, try again with different settings
				await page.goto(
					'https://mail.google.com',
					wait_until='commit',  # Less strict wait condition
					timeout=30000,
				)

			# Wait for the page to stabilize
			await asyncio.sleep(2)

			print(f'Current page title: {await page.title()}')

			# Optional: wait for specific Gmail elements
			try:
				await page.wait_for_selector('div[role="main"]', timeout=5000)
				print('Gmail interface detected')
			except Exception as e:
				print(f'Note: Gmail interface not detected: {e}')

			await asyncio.sleep(30)
		except Exception as e:
			print(f'An error occurred: {e}')
			import traceback

			traceback.print_exc()
		finally:
			await browser.close()


if __name__ == '__main__':
	asyncio.run(test_full_screen(False, False))

```

---

## `browser-use-main\tests\test_browser.py`

```py
import pytest
from playwright.async_api import async_playwright

from browser_use.browser import BrowserSession


@pytest.mark.asyncio
async def test_connection_via_cdp(monkeypatch):
	browser_session = BrowserSession(
		cdp_url='http://localhost:9898',
	)
	with pytest.raises(Exception) as e:
		await browser_session.start()

	# Assert on the exception value outside the context manager
	assert 'ECONNREFUSED' in str(e.value)

	playwright = await async_playwright().start()
	browser = await playwright.chromium.launch(args=['--remote-debugging-port=9898'])

	await browser_session.start()
	await browser_session.create_new_tab()

	assert (await browser_session.get_current_page()).url == 'about:blank'

	await browser_session.close()
	await browser.close()

```

---

## `browser-use-main\tests\test_browser_config_models.py`

```py
import os

import pytest

from browser_use.browser.profile import BrowserProfile, ProxySettings
from browser_use.browser.session import BrowserSession


@pytest.mark.asyncio
async def test_proxy_settings_pydantic_model():
	"""
	Test that ProxySettings as a Pydantic model is correctly converted to a dictionary when used.
	"""
	# Create ProxySettings with Pydantic model
	proxy_settings = ProxySettings(
		server='http://example.proxy:8080', bypass='localhost', username='testuser', password='testpass'
	)

	# Verify the model has correct dict-like access
	assert proxy_settings['server'] == 'http://example.proxy:8080'
	assert proxy_settings.get('bypass') == 'localhost'
	assert proxy_settings.get('nonexistent', 'default') == 'default'

	# Verify model_dump works correctly
	proxy_dict = proxy_settings.model_dump()
	assert isinstance(proxy_dict, dict)
	assert proxy_dict['server'] == 'http://example.proxy:8080'
	assert proxy_dict['bypass'] == 'localhost'
	assert proxy_dict['username'] == 'testuser'
	assert proxy_dict['password'] == 'testpass'

	# We don't launch the actual browser - we just verify the model itself works as expected


@pytest.mark.asyncio
async def test_window_size_config():
	"""
	Test that BrowserProfile correctly handles window_size property.
	"""
	# Create profile with specific window dimensions
	profile = BrowserProfile(window_size={'width': 1280, 'height': 1100})

	# Verify the properties are set correctly
	assert profile.window_size['width'] == 1280
	assert profile.window_size['height'] == 1100

	# Verify model_dump works correctly
	profile_dict = profile.model_dump()
	assert isinstance(profile_dict, dict)
	assert profile_dict['window_size']['width'] == 1280
	assert profile_dict['window_size']['height'] == 1100

	# Create with different values
	profile2 = BrowserProfile(window_size={'width': 1920, 'height': 1080})
	assert profile2.window_size['width'] == 1920
	assert profile2.window_size['height'] == 1080


@pytest.mark.asyncio
@pytest.mark.skipif(os.environ.get('CI') == 'true', reason='Skip browser test in CI')
async def test_window_size_with_real_browser():
	"""
	Integration test that verifies our window size Pydantic model is correctly
	passed to Playwright and the actual browser window is configured with these settings.
	This test is skipped in CI environments.
	"""
	# Create browser profile with headless mode and specific dimensions
	browser_profile = BrowserProfile(
		headless=True,  # Use headless for faster test
		window_size={'width': 1024, 'height': 768},
		maximum_wait_page_load_time=2.0,  # Faster timeouts for test
		minimum_wait_page_load_time=0.2,
		no_viewport=True,  # Use actual window size instead of viewport
	)

	# Create browser session
	browser_session = BrowserSession(browser_profile=browser_profile)
	try:
		await browser_session.start()
		# Get the current page
		page = await browser_session.get_current_page()
		assert page is not None, 'Failed to get current page'

		# Get the context configuration used for browser window size
		video_size = await page.evaluate("""
                () => {
                    // This returns information about the context recording settings
                    // which should match our configured video size (browser_window_size)
                    try {
                        const settings = window.getPlaywrightContextSettings ? 
                            window.getPlaywrightContextSettings() : null;
                        if (settings && settings.recordVideo) {
                            return settings.recordVideo.size;
                        }
                    } catch (e) {}
                    
                    // Fallback to window dimensions
                    return {
                        width: window.innerWidth,
                        height: window.innerHeight
                    };
                }
            """)

		# Let's also check the viewport size
		viewport_size = await page.evaluate("""
                () => {
                    return {
                        width: window.innerWidth,
                        height: window.innerHeight
                    }
                }
            """)

		print(f'Window size config: width={browser_profile.window_size["width"]}, height={browser_profile.window_size["height"]}')
		print(f'Browser viewport size: {viewport_size}')

		# This is a lightweight test to verify that the page has a size (details may vary by browser)
		assert viewport_size['width'] > 0, 'Expected viewport width to be positive'
		assert viewport_size['height'] > 0, 'Expected viewport height to be positive'

		# For browser context creation in record_video_size, this is what truly matters
		# Verify that our window size was properly serialized to a dictionary
		print(f'Content of context session: {browser_session.browser_context}')
		print('✅ Browser window size used in the test')
	finally:
		await browser_session.stop()


@pytest.mark.asyncio
async def test_proxy_with_real_browser():
	"""
	Integration test that verifies our proxy Pydantic model is correctly
	passed to Playwright without requiring a working proxy server.

	This test:
	1. Creates a ProxySettings Pydantic model
	2. Passes it to BrowserProfile
	3. Verifies browser initialization works (proving the model was correctly serialized)
	4. We don't actually verify proxy functionality (would require a working proxy)
	"""
	# Create proxy settings with a fake proxy server
	proxy_settings = ProxySettings(
		server='http://non.existent.proxy:9999', bypass='localhost', username='testuser', password='testpass'
	)

	# Test model serialization
	proxy_dict = proxy_settings.model_dump()
	assert isinstance(proxy_dict, dict)
	assert proxy_dict['server'] == 'http://non.existent.proxy:9999'

	# Create browser profile with proxy
	browser_profile = BrowserProfile(
		headless=True,
		proxy=proxy_settings,
	)

	# Create browser session
	browser_session = BrowserSession(browser_profile=browser_profile)
	try:
		await browser_session.start()
		# Success - the browser was initialized with our proxy settings
		# We won't try to make requests (which would fail with non-existent proxy)
		print('✅ Browser initialized with proxy settings successfully')
	finally:
		await browser_session.stop()

```

---

## `browser-use-main\tests\test_browser_session.py`

```py
import asyncio
import base64

import pytest
from pytest_httpserver import HTTPServer

from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.dom.views import DOMElementNode


class TestBrowserContext:
	"""Tests for browser context functionality using real browser instances."""

	@pytest.fixture(scope='module')
	def event_loop(self):
		"""Create and provide an event loop for async tests."""
		loop = asyncio.get_event_loop_policy().new_event_loop()
		yield loop
		loop.close()

	@pytest.fixture(scope='module')
	def http_server(self):
		"""Create and provide a test HTTP server that serves static content."""
		server = HTTPServer()
		server.start()

		# Add routes for test pages
		server.expect_request('/').respond_with_data(
			'<html><head><title>Test Home Page</title></head><body><h1>Test Home Page</h1><p>Welcome to the test site</p></body></html>',
			content_type='text/html',
		)

		server.expect_request('/scroll_test').respond_with_data(
			"""
            <html>
            <head>
                <title>Scroll Test</title>
                <style>
                    body { height: 3000px; }
                    .marker { position: absolute; }
                    #top { top: 0; }
                    #middle { top: 1000px; }
                    #bottom { top: 2000px; }
                </style>
            </head>
            <body>
                <div id="top" class="marker">Top of the page</div>
                <div id="middle" class="marker">Middle of the page</div>
                <div id="bottom" class="marker">Bottom of the page</div>
            </body>
            </html>
            """,
			content_type='text/html',
		)

		yield server
		server.stop()

	@pytest.fixture
	def base_url(self, http_server):
		"""Return the base URL for the test HTTP server."""
		return f'http://{http_server.host}:{http_server.port}'

	@pytest.fixture(scope='module')
	async def browser_session(self, event_loop):
		"""Create and provide a BrowserSession instance with security disabled."""
		browser_session = BrowserSession(
			# browser_profile=BrowserProfile(...),
			headless=True,
			user_data_dir=None,
		)
		await browser_session.start()
		yield browser_session
		await browser_session.stop()

	def test_is_url_allowed(self):
		"""
		Test the _is_url_allowed method to verify that it correctly checks URLs against
		the allowed domains configuration.
		"""
		# Scenario 1: allowed_domains is None, any URL should be allowed.
		config1 = BrowserProfile(allowed_domains=None)
		context1 = BrowserSession(browser_profile=config1)
		assert context1._is_url_allowed('http://anydomain.com') is True
		assert context1._is_url_allowed('https://anotherdomain.org/path') is True

		# Scenario 2: allowed_domains is provided.
		allowed = ['example.com', '*.mysite.org']
		config2 = BrowserProfile(allowed_domains=allowed)
		context2 = BrowserSession(browser_profile=config2)

		# URL exactly matching
		assert context2._is_url_allowed('http://example.com') is True
		# URL with subdomain (should not be allowed)
		assert context2._is_url_allowed('http://sub.example.com/path') is False
		# URL with different domain (should not be allowed)
		assert context2._is_url_allowed('http://sub.mysite.org') is True
		# URL that matches second allowed domain
		assert context2._is_url_allowed('https://mysite.org/page') is True
		# URL with port number, still allowed (port is stripped)
		assert context2._is_url_allowed('http://example.com:8080') is True
		assert context2._is_url_allowed('https://example.com:443') is True

		# Scenario 3: Malformed URL or empty domain
		# urlparse will return an empty netloc for some malformed URLs.
		assert context2._is_url_allowed('notaurl') is False

	def test_convert_simple_xpath_to_css_selector(self):
		"""
		Test the _convert_simple_xpath_to_css_selector method of BrowserSession.
		This verifies that simple XPath expressions are correctly converted to CSS selectors.
		"""
		# Test empty xpath returns empty string
		assert BrowserSession._convert_simple_xpath_to_css_selector('') == ''

		# Test a simple xpath without indices
		xpath = '/html/body/div/span'
		expected = 'html > body > div > span'
		result = BrowserSession._convert_simple_xpath_to_css_selector(xpath)
		assert result == expected

		# Test xpath with an index on one element: [2] should translate to :nth-of-type(2)
		xpath = '/html/body/div[2]/span'
		expected = 'html > body > div:nth-of-type(2) > span'
		result = BrowserSession._convert_simple_xpath_to_css_selector(xpath)
		assert result == expected

		# Test xpath with indices on multiple elements
		xpath = '/ul/li[3]/a[1]'
		expected = 'ul > li:nth-of-type(3) > a:nth-of-type(1)'
		result = BrowserSession._convert_simple_xpath_to_css_selector(xpath)
		assert result == expected

	def test_enhanced_css_selector_for_element(self):
		"""
		Test the _enhanced_css_selector_for_element method to verify that
		it returns the correct CSS selector string for a DOMElementNode.
		"""
		# Create a DOMElementNode instance with a complex set of attributes
		dummy_element = DOMElementNode(
			tag_name='div',
			is_visible=True,
			parent=None,
			xpath='/html/body/div[2]',
			attributes={'class': 'foo bar', 'id': 'my-id', 'placeholder': 'some "quoted" text', 'data-testid': '123'},
			children=[],
		)

		# Call the method with include_dynamic_attributes=True
		actual_selector = BrowserSession._enhanced_css_selector_for_element(dummy_element, include_dynamic_attributes=True)

		# Expected conversion includes the xpath conversion, class attributes, and other attributes
		expected_selector = (
			'html > body > div:nth-of-type(2).foo.bar[id="my-id"][placeholder*="some \\"quoted\\" text"][data-testid="123"]'
		)
		assert actual_selector == expected_selector, f'Expected {expected_selector}, but got {actual_selector}'

	@pytest.mark.asyncio
	async def test_navigate_and_get_current_page(self, browser_session, base_url):
		"""Test that navigate method changes the URL and get_current_page returns the proper page."""
		# Navigate to the test page
		await browser_session.navigate(f'{base_url}/')

		# Get the current page
		page = await browser_session.get_current_page()

		# Verify the page URL matches what we navigated to
		assert f'{base_url}/' in page.url

		# Verify the page title
		title = await page.title()
		assert title == 'Test Home Page'

	@pytest.mark.asyncio
	async def test_refresh_page(self, browser_session, base_url):
		"""Test that refresh_page correctly reloads the current page."""
		# Navigate to the test page
		await browser_session.navigate(f'{base_url}/')

		# Get the current page before refresh
		page_before = await browser_session.get_current_page()

		# Refresh the page
		await browser_session.refresh()

		# Get the current page after refresh
		page_after = await browser_session.get_current_page()

		# Verify it's still on the same URL
		assert page_after.url == page_before.url

		# Verify the page title is still correct
		title = await page_after.title()
		assert title == 'Test Home Page'

	@pytest.mark.asyncio
	async def test_execute_javascript(self, browser_session, base_url):
		"""Test that execute_javascript correctly executes JavaScript in the current page."""
		# Navigate to a test page
		await browser_session.navigate(f'{base_url}/')
… (truncated)

```

---

## `browser-use-main\tests\test_browser_window_size_height.py`

```py
"""
Example script demonstrating the browser_window_size feature.
This script shows how to set a custom window size for the browser.
"""

import asyncio
import sys
from typing import Any

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig


async def main():
	"""Demonstrate setting a custom browser window size"""
	# Create a browser with a specific window size
	config = BrowserContextConfig(window_width=800, window_height=400)  # Small size to clearly demonstrate the fix

	browser = None
	browser_context = None

	try:
		# Initialize the browser with error handling
		try:
			browser = Browser(
				config=BrowserConfig(
					headless=False,  # Use non-headless mode to see the window
				)
			)
		except Exception as e:
			print(f'Failed to initialize browser: {e}')
			return 1

		# Create a browser context
		try:
			browser_context = await browser.new_context(config=config)
		except Exception as e:
			print(f'Failed to create browser context: {e}')
			return 1

		# Get the current page
		page = await browser_context.get_current_page()

		# Navigate to a test page with error handling
		try:
			await page.goto('https://example.com')
			await page.wait_for_load_state('domcontentloaded')
		except Exception as e:
			print(f'Failed to navigate to example.com: {e}')
			print('Continuing with test anyway...')

		# Wait a bit to see the window
		await asyncio.sleep(2)

		# Get the actual viewport size using JavaScript
		viewport_size = await page.evaluate("""
			() => {
				return {
					width: window.innerWidth,
					height: window.innerHeight
				}
			}
		""")

		print(f'Configured window size: width={config.window_width}, height={config.window_height}')
		print(f'Actual viewport size: {viewport_size}')

		# Validate the window size
		validate_window_size({'width': config.window_width, 'height': config.window_height}, viewport_size)

		# Wait a bit more to see the window
		await asyncio.sleep(3)

		return 0

	except Exception as e:
		print(f'Unexpected error: {e}')
		return 1

	finally:
		# Close resources
		if browser_context:
			await browser_context.close()
		if browser:
			await browser.close()


def validate_window_size(configured: dict[str, Any], actual: dict[str, Any]) -> None:
	"""Compare configured window size with actual size and report differences"""
	# Allow for small differences due to browser chrome, scrollbars, etc.
	width_diff = abs(configured['width'] - actual['width'])
	height_diff = abs(configured['height'] - actual['height'])

	# Tolerance of 5% or 20px, whichever is greater
	width_tolerance = max(configured['width'] * 0.05, 20)
	height_tolerance = max(configured['height'] * 0.05, 20)

	if width_diff > width_tolerance or height_diff > height_tolerance:
		print('WARNING: Significant difference between configured and actual window size!')
		print(f'Width difference: {width_diff}px, Height difference: {height_diff}px')
	else:
		print('Window size validation passed: actual size matches configured size within tolerance')


if __name__ == '__main__':
	result = asyncio.run(main())
	sys.exit(result)

```

---

## `browser-use-main\tests\test_browser_window_size_height_no_viewport.py`

```py
import asyncio

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig


async def test():
	print('Testing browser window sizing with no_viewport=False...')
	browser = Browser(BrowserConfig(headless=False))
	context_config = BrowserContextConfig(window_width=1440, window_height=900, no_viewport=False)
	browser_context = await browser.new_context(config=context_config)
	page = await browser_context.get_current_page()
	await page.goto('https://example.com')
	await asyncio.sleep(2)
	viewport = await page.evaluate('() => ({width: window.innerWidth, height: window.innerHeight})')
	print('Configured size: width=1440, height=900')
	print(f'Actual viewport size: {viewport}')

	# Get the actual window size
	window_size = await page.evaluate("""
        () => ({
            width: window.outerWidth,
            height: window.outerHeight
        })
    """)
	print(f'Actual window size: {window_size}')

	await browser_context.close()
	await browser.close()


if __name__ == '__main__':
	asyncio.run(test())

```

---

## `browser-use-main\tests\test_clicks.py`

```py
import asyncio
import json

import anyio
import pytest

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.dom.views import DOMBaseNode, DOMElementNode, DOMTextNode
from browser_use.utils import time_execution_sync


class ElementTreeSerializer:
	@staticmethod
	def dom_element_node_to_json(element_tree: DOMElementNode) -> dict:
		def node_to_dict(node: DOMBaseNode) -> dict:
			if isinstance(node, DOMTextNode):
				return {'type': 'text', 'text': node.text}
			elif isinstance(node, DOMElementNode):
				return {
					'type': 'element',
					'tag_name': node.tag_name,
					'attributes': node.attributes,
					'highlight_index': node.highlight_index,
					'children': [node_to_dict(child) for child in node.children],
				}
			return {}

		return node_to_dict(element_tree)


# run with: pytest browser_use/browser/tests/test_clicks.py
@pytest.mark.asyncio
async def test_highlight_elements():
	browser = Browser(config=BrowserConfig(headless=False, disable_security=True, user_data_dir=None))

	async with await browser.new_context() as context:
		page = await context.get_current_page()
		# await page.goto('https://immobilienscout24.de')
		# await page.goto('https://help.sap.com/docs/sap-ai-core/sap-ai-core-service-guide/service-plans')
		# await page.goto('https://google.com/search?q=elon+musk')
		# await page.goto('https://kayak.com')
		# await page.goto('https://www.w3schools.com/tags/tryit.asp?filename=tryhtml_iframe')
		# await page.goto('https://dictionary.cambridge.org')
		# await page.goto('https://github.com')
		await page.goto('https://huggingface.co/')

		await asyncio.sleep(1)

		while True:
			try:
				# await asyncio.sleep(10)
				state = await context.get_state_summary(True)

				async with await anyio.open_file('./tmp/page.json', 'w') as f:
					await f.write(
						json.dumps(
							ElementTreeSerializer.dom_element_node_to_json(state.element_tree),
							indent=1,
						)
					)

				# await time_execution_sync('highlight_selector_map_elements')(
				# 	browser.highlight_selector_map_elements
				# )(state.selector_map)

				# Find and print duplicate XPaths
				xpath_counts = {}
				if not state.selector_map:
					continue
				for selector in state.selector_map.values():
					xpath = selector.xpath
					if xpath in xpath_counts:
						xpath_counts[xpath] += 1
					else:
						xpath_counts[xpath] = 1

				print('\nDuplicate XPaths found:')
				for xpath, count in xpath_counts.items():
					if count > 1:
						print(f'XPath: {xpath}')
						print(f'Count: {count}\n')

				print(list(state.selector_map.keys()), 'Selector map keys')
				print(state.element_tree.clickable_elements_to_string())
				action = input('Select next action: ')

				await time_execution_sync('remove_highlight_elements')(context.remove_highlights)()

				node_element = state.selector_map[int(action)]

				# check if index of selector map are the same as index of items in dom_items

				await context._click_element_node(node_element)

			except Exception as e:
				print(e)

```

---

## `browser-use-main\tests\test_controller.py`

```py
import asyncio
import time

import pytest
from pydantic import BaseModel
from pytest_httpserver import HTTPServer

from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser import BrowserSession
from browser_use.controller.service import Controller
from browser_use.controller.views import (
	ClickElementAction,
	CloseTabAction,
	DoneAction,
	DragDropAction,
	GoToUrlAction,
	InputTextAction,
	NoParamsAction,
	OpenTabAction,
	ScrollAction,
	SearchGoogleAction,
	SendKeysAction,
	SwitchTabAction,
)


class TestControllerIntegration:
	"""Integration tests for Controller using actual browser instances."""

	@pytest.fixture(scope='module')
	def event_loop(self):
		"""Create and provide an event loop for async tests."""
		loop = asyncio.get_event_loop_policy().new_event_loop()
		yield loop
		loop.close()

	@pytest.fixture(scope='module')
	def http_server(self):
		"""Create and provide a test HTTP server that serves static content."""
		server = HTTPServer()
		server.start()

		# Add routes for common test pages
		server.expect_request('/').respond_with_data(
			'<html><head><title>Test Home Page</title></head><body><h1>Test Home Page</h1><p>Welcome to the test site</p></body></html>',
			content_type='text/html',
		)

		server.expect_request('/page1').respond_with_data(
			'<html><head><title>Test Page 1</title></head><body><h1>Test Page 1</h1><p>This is test page 1</p></body></html>',
			content_type='text/html',
		)

		server.expect_request('/page2').respond_with_data(
			'<html><head><title>Test Page 2</title></head><body><h1>Test Page 2</h1><p>This is test page 2</p></body></html>',
			content_type='text/html',
		)

		server.expect_request('/search').respond_with_data(
			"""
			<html>
			<head><title>Search Results</title></head>
			<body>
				<h1>Search Results</h1>
				<div class="results">
					<div class="result">Result 1</div>
					<div class="result">Result 2</div>
					<div class="result">Result 3</div>
				</div>
			</body>
			</html>
			""",
			content_type='text/html',
		)

		yield server
		server.stop()

	@pytest.fixture
	def base_url(self, http_server):
		"""Return the base URL for the test HTTP server."""
		return f'http://{http_server.host}:{http_server.port}'

	@pytest.fixture(scope='module')
	async def browser_session(self, event_loop):
		"""Create and provide a Browser instance with security disabled."""
		browser_session = BrowserSession(
			# browser_profile=BrowserProfile(),
			headless=True,
			user_data_dir=None,
		)
		await browser_session.start()
		yield browser_session
		await browser_session.stop()

	@pytest.fixture
	def controller(self):
		"""Create and provide a Controller instance."""
		return Controller()

	@pytest.mark.asyncio
	async def test_go_to_url_action(self, controller, browser_session, base_url):
		"""Test that GoToUrlAction navigates to the specified URL."""
		# Create action model for go_to_url
		action_data = {'go_to_url': GoToUrlAction(url=f'{base_url}/page1')}

		# Create the ActionModel instance
		class GoToUrlActionModel(ActionModel):
			go_to_url: GoToUrlAction | None = None

		action_model = GoToUrlActionModel(**action_data)

		# Execute the action
		result = await controller.act(action_model, browser_session)

		# Verify the result
		assert isinstance(result, ActionResult)
		assert f'Navigated to {base_url}/page1' in result.extracted_content

		# Verify the current page URL
		page = await browser_session.get_current_page()
		assert f'{base_url}/page1' in page.url

	@pytest.mark.asyncio
	async def test_scroll_actions(self, controller, browser_session, base_url):
		"""Test that scroll actions correctly scroll the page."""
		# First navigate to a page
		goto_action = {'go_to_url': GoToUrlAction(url=f'{base_url}/page1')}

		class GoToUrlActionModel(ActionModel):
			go_to_url: GoToUrlAction | None = None

		await controller.act(GoToUrlActionModel(**goto_action), browser_session)

		# Create scroll down action
		scroll_action = {'scroll_down': ScrollAction(amount=200)}

		class ScrollActionModel(ActionModel):
			scroll_down: ScrollAction | None = None

		# Execute scroll down
		result = await controller.act(ScrollActionModel(**scroll_action), browser_session)

		# Verify the result
		assert isinstance(result, ActionResult)
		assert 'Scrolled down' in result.extracted_content

		# Create scroll up action
		scroll_up_action = {'scroll_up': ScrollAction(amount=100)}

		class ScrollUpActionModel(ActionModel):
			scroll_up: ScrollAction | None = None

		# Execute scroll up
		result = await controller.act(ScrollUpActionModel(**scroll_up_action), browser_session)

		# Verify the result
		assert isinstance(result, ActionResult)
		assert 'Scrolled up' in result.extracted_content

	@pytest.mark.asyncio
	async def test_registry_actions(self, controller, browser_session):
		"""Test that the registry contains the expected default actions."""
		# Check that common actions are registered
		common_actions = [
			'go_to_url',
			'search_google',
			'click_element_by_index',
			'input_text',
			'scroll_down',
			'scroll_up',
			'go_back',
			'switch_tab',
			'open_tab',
			'close_tab',
			'wait',
		]

		for action in common_actions:
			assert action in controller.registry.registry.actions
			assert controller.registry.registry.actions[action].function is not None
			assert controller.registry.registry.actions[action].description is not None

	@pytest.mark.asyncio
	async def test_custom_action_registration(self, controller, browser_session, base_url):
		"""Test registering a custom action and executing it."""

		# Define a custom action
		class CustomParams(BaseModel):
			text: str

		@controller.action('Test custom action', param_model=CustomParams)
		async def custom_action(params: CustomParams, browser_session):
			page = await browser_session.get_current_page()
			return ActionResult(extracted_content=f'Custom action executed with: {params.text} on {page.url}')

		# Navigate to a page first
		goto_action = {'go_to_url': GoToUrlAction(url=f'{base_url}/page1')}

		class GoToUrlActionModel(ActionModel):
… (truncated)

```

---

## `browser-use-main\tests\test_core_functionality.py`

```py
import asyncio

import pytest
from langchain_openai import ChatOpenAI
from pytest_httpserver import HTTPServer

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser import BrowserProfile, BrowserSession


class TestCoreFunctionality:
	"""Tests for core functionality of the Agent using real browser instances."""

	@pytest.fixture(scope='module')
	def event_loop(self):
		"""Create and provide an event loop for async tests."""
		loop = asyncio.get_event_loop_policy().new_event_loop()
		yield loop
		loop.close()

	@pytest.fixture(scope='module')
	def http_server(self):
		"""Create and provide a test HTTP server that serves static content."""
		server = HTTPServer()
		server.start()

		# Add routes for common test pages
		server.expect_request('/').respond_with_data(
			'<html><head><title>Test Home Page</title></head><body><h1>Test Home Page</h1><p>Welcome to the test site</p></body></html>',
			content_type='text/html',
		)

		server.expect_request('/page1').respond_with_data(
			'<html><head><title>Test Page 1</title></head><body><h1>Test Page 1</h1><p>This is test page 1</p><a href="/page2">Link to Page 2</a></body></html>',
			content_type='text/html',
		)

		server.expect_request('/page2').respond_with_data(
			'<html><head><title>Test Page 2</title></head><body><h1>Test Page 2</h1><p>This is test page 2</p><a href="/page1">Back to Page 1</a></body></html>',
			content_type='text/html',
		)

		server.expect_request('/search').respond_with_data(
			"""
            <html>
            <head><title>Search Results</title></head>
            <body>
                <h1>Search Results</h1>
                <form>
                    <input type="text" id="search-box" placeholder="Search...">
                    <button type="submit">Search</button>
                </form>
                <div class="results">
                    <div class="result">Result 1</div>
                    <div class="result">Result 2</div>
                    <div class="result">Result 3</div>
                </div>
            </body>
            </html>
            """,
			content_type='text/html',
		)

		yield server
		server.stop()

	@pytest.fixture
	def base_url(self, http_server):
		"""Return the base URL for the test HTTP server."""
		return f'http://{http_server.host}:{http_server.port}'

	@pytest.fixture(scope='module')
	async def browser_session(self, event_loop):
		"""Create and provide a BrowserSession instance with security disabled."""
		browser_session = BrowserSession(
			browser_profile=BrowserProfile(
				headless=True,
				disable_security=True,
			)
		)
		yield browser_session
		await browser_session.stop()

	@pytest.fixture
	def llm(self):
		"""Initialize language model for testing with minimal settings."""
		return ChatOpenAI(
			model='gpt-4o',
			temperature=0.0,
		)

	@pytest.mark.asyncio
	async def test_search_google(self, llm, browser_session, base_url):
		"""Test 'Search Google' action using a mock search page."""
		agent = Agent(
			task=f"Go to '{base_url}/search' and search for 'OpenAI'.",
			llm=llm,
			browser_session=browser_session,
		)
		history: AgentHistoryList = await agent.run(max_steps=3)
		action_names = history.action_names()
		assert 'go_to_url' in action_names
		assert any('input_text' in action or 'click_element_by_index' in action for action in action_names)

	@pytest.mark.asyncio
	async def test_go_to_url(self, llm, browser_session, base_url):
		"""Test 'Navigate to URL' action."""
		agent = Agent(
			task=f"Navigate to '{base_url}/page1'.",
			llm=llm,
			browser_session=browser_session,
		)
		history = await agent.run(max_steps=2)
		action_names = history.action_names()
		assert 'go_to_url' in action_names

		# Verify we're on the correct page
		page = await browser_session.get_current_page()
		assert f'{base_url}/page1' in page.url

	@pytest.mark.asyncio
	async def test_go_back(self, llm, browser_session, base_url):
		"""Test 'Go back' action."""
		# First navigate to page1, then to page2, then go back
		agent = Agent(
			task=f"Go to '{base_url}/page1', then go to '{base_url}/page2', then go back.",
			llm=llm,
			browser_session=browser_session,
		)
		history = await agent.run(max_steps=4)
		action_names = history.action_names()
		assert 'go_to_url' in action_names
		assert 'go_back' in action_names

		# Verify we're back on page1
		page = await browser_session.get_current_page()
		assert f'{base_url}/page1' in page.url

	@pytest.mark.asyncio
	async def test_click_element(self, llm, browser_session, base_url):
		"""Test 'Click element' action."""
		agent = Agent(
			task=f"Go to '{base_url}/page1' and click on the link to Page 2.",
			llm=llm,
			browser_session=browser_session,
		)
		history = await agent.run(max_steps=3)
		action_names = history.action_names()
		assert 'go_to_url' in action_names
		assert 'click_element_by_index' in action_names

		# Verify we're now on page2 after clicking the link
		page = await browser_session.get_current_page()
		assert f'{base_url}/page2' in page.url

	@pytest.mark.asyncio
	async def test_input_text(self, llm, browser_session, base_url):
		"""Test 'Input text' action."""
		agent = Agent(
			task=f"Go to '{base_url}/search' and input 'OpenAI' into the search box.",
			llm=llm,
			browser_session=browser_session,
		)
		history = await agent.run(max_steps=3)
		action_names = history.action_names()
		assert 'go_to_url' in action_names
		assert 'input_text' in action_names

		# Verify text was entered in the search box
		page = await browser_session.get_current_page()
		search_value = await page.evaluate("document.getElementById('search-box').value")
		assert 'OpenAI' in search_value

	@pytest.mark.asyncio
	async def test_switch_tab(self, llm, browser_session, base_url):
		"""Test 'Switch tab' action."""
		agent = Agent(
			task=f"Open '{base_url}/page1' in the current tab, then open a new tab with '{base_url}/page2', then switch back to the first tab.",
			llm=llm,
			browser_session=browser_session,
		)
		history = await agent.run(max_steps=4)
		action_names = history.action_names()
		assert 'go_to_url' in action_names
		assert 'open_tab' in action_names
		assert 'switch_tab' in action_names

		# Verify we're back on the first tab with page1
		page = await browser_session.get_current_page()
		assert f'{base_url}/page1' in page.url

	@pytest.mark.asyncio
	async def test_open_new_tab(self, llm, browser_session, base_url):
		"""Test 'Open new tab' action."""
		agent = Agent(
			task=f"Open a new tab and go to '{base_url}/page2'.",
			llm=llm,
			browser_session=browser_session,
		)
… (truncated)

```

---

## `browser-use-main\tests\test_dropdown.py`

```py
"""
Test dropdown interaction functionality.
"""

import pytest

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList


@pytest.mark.asyncio
async def test_dropdown(llm, browser_context):
	"""Test selecting an option from a dropdown menu."""
	agent = Agent(
		task=(
			'go to https://codepen.io/geheimschriftstift/pen/mPLvQz and first get all options for the dropdown and then select the 5th option'
		),
		llm=llm,
		browser_context=browser_context,
	)

	try:
		history: AgentHistoryList = await agent.run(20)
		result = history.final_result()

		# Verify dropdown interaction
		assert result is not None
		assert 'Duck' in result, "Expected 5th option 'Duck' to be selected"

		# Verify dropdown state
		element = await browser_context.get_element_by_selector('select')
		assert element is not None, 'Dropdown element should exist'

		value = await element.evaluate('el => el.value')
		assert value == '5', 'Dropdown should have 5th option selected'

	except Exception as e:
		pytest.fail(f'Dropdown test failed: {str(e)}')
	finally:
		await browser_context.close()

```

---

## `browser-use-main\tests\test_dropdown_complex.py`

```py
"""
Test complex dropdown interaction functionality.
"""

import pytest

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList


@pytest.mark.asyncio
async def test_dropdown_complex(llm, browser_context):
	"""Test selecting an option from a complex dropdown menu."""
	agent = Agent(
		task=(
			'go to https://codepen.io/shyam-king/pen/pvzpByJ and first get all options for the dropdown and then select the json option'
		),
		llm=llm,
		browser_context=browser_context,
	)

	try:
		history: AgentHistoryList = await agent.run(20)
		result = history.final_result()

		# Verify dropdown interaction
		assert result is not None
		assert 'json' in result.lower(), "Expected 'json' option to be selected"

		# Verify dropdown state
		element = await browser_context.get_element_by_selector('.select-selected')
		assert element is not None, 'Custom dropdown element should exist'

		text = await element.text_content()
		assert 'json' in text.lower(), 'Dropdown should display json option'

		# Verify the selected option's effect
		code_element = await browser_context.get_element_by_selector('pre code')
		assert code_element is not None, 'Code element should be visible when JSON is selected'

	except Exception as e:
		pytest.fail(f'Complex dropdown test failed: {str(e)}')
	finally:
		await browser_context.close()

```

---

## `browser-use-main\tests\test_dropdown_error.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList

llm = ChatOpenAI(model='gpt-4o')
# browser = Browser(config=BrowserConfig(headless=False))

agent = Agent(
	task=('go to https://codepen.io/shyam-king/pen/emOyjKm and select number "4" and return the output of "selected value"'),
	llm=llm,
	browser_context=BrowserContext(
		browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
	),
)


async def test_dropdown():
	history: AgentHistoryList = await agent.run(20)
	# await controller.browser.close(force=True)

	result = history.final_result()
	assert result is not None
	assert '4' in result
	print(result)

	# await browser.close()

```

---

## `browser-use-main\tests\test_excluded_actions.py`

```py
import asyncio
import os

import pytest
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller

# run with:
# python -m pytest tests/test_excluded_actions.py -v -k "test_only_open_tab_allowed" --capture=no


@pytest.fixture(scope='session')
def event_loop():
	"""Create an instance of the default event loop for each test case."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='session')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as context:
		yield context


@pytest.fixture
def llm():
	"""Initialize language model for testing"""
	return AzureChatOpenAI(
		model='gpt-4o',
		api_version='2024-10-21',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)


# pytest tests/test_excluded_actions.py -v -k "test_only_open_tab_allowed" --capture=no
@pytest.mark.asyncio
async def test_only_open_tab_allowed(llm, context):
	"""Test that only open_tab action is available while others are excluded"""

	# Create list of all default actions except open_tab
	excluded_actions = [
		'search_google',
		'go_to_url',
		'go_back',
		'click_element',
		'input_text',
		'switch_tab',
		'extract_content',
		'done',
		'scroll_down',
		'scroll_up',
		'send_keys',
		'scroll_to_text',
		'get_dropdown_options',
		'select_dropdown_option',
	]

	# Initialize controller with excluded actions
	controller = Controller(exclude_actions=excluded_actions)

	# Create agent with a task that would normally use other actions
	agent = Agent(
		task="Go to google.com and search for 'python programming'",
		llm=llm,
		browser_context=context,
		controller=controller,
	)

	history: AgentHistoryList = await agent.run(max_steps=2)

	# Verify that only open_tab was used
	action_names = history.action_names()

	# Only open_tab should be in the actions
	assert all(action == 'open_tab' for action in action_names), (
		f'Found unexpected actions: {[a for a in action_names if a != "open_tab"]}'
	)

	# open_tab should be used at least once
	assert 'open_tab' in action_names, 'open_tab action was not used'

```

---

## `browser-use-main\tests\test_full_screen.py`

```py
import asyncio

from playwright.async_api import async_playwright


async def test_full_screen(start_fullscreen: bool, maximize: bool):
	async with async_playwright() as p:
		browser = await p.chromium.launch(
			headless=False,
			args=['--start-maximized'],
		)
		context = await browser.new_context(no_viewport=True, viewport=None)
		page = await context.new_page()
		await page.goto('https://google.com')

		await asyncio.sleep(10)
		await browser.close()


if __name__ == '__main__':
	asyncio.run(test_full_screen(False, False))

```

---

## `browser-use-main\tests\test_gif_path.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList

llm = ChatOpenAI(model='gpt-4o')

agent = Agent(
	task=('go to google.com and search for text "hi there"'),
	llm=llm,
	browser_context=BrowserContext(
		browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
	),
	generate_gif='./google.gif',
)


async def test_gif_path():
	if os.path.exists('./google.gif'):
		os.unlink('./google.gif')

	history: AgentHistoryList = await agent.run(20)

	result = history.final_result()
	assert result is not None

	assert os.path.exists('./google.gif'), 'google.gif was not created'

```

---

## `browser-use-main\tests\test_mind2web.py`

```py
"""
Test browser automation using Mind2Web dataset tasks with pytest framework.
"""

import asyncio
import json
import os
from typing import Any

import pytest
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.utils import logger

# Constants
MAX_STEPS = 50
TEST_SUBSET_SIZE = 10


@pytest.fixture(scope='session')
def event_loop():
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='session')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as new_context:
		yield new_context


@pytest.fixture(scope='session')
def test_cases() -> list[dict[str, Any]]:
	"""Load test cases from Mind2Web dataset"""
	file_path = os.path.join(os.path.dirname(__file__), 'mind2web_data/processed.json')
	logger.info(f'Loading test cases from {file_path}')

	with open(file_path) as f:
		data = json.load(f)

	subset = data[:TEST_SUBSET_SIZE]
	logger.info(f'Loaded {len(subset)}/{len(data)} test cases')
	return subset


@pytest.fixture
def llm():
	"""Initialize language model for testing"""

	# return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None)
	return AzureChatOpenAI(
		model='gpt-4o',
		api_version='2024-10-21',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)


# run with: pytest -s -v tests/test_mind2web.py:test_random_samples
@pytest.mark.asyncio
async def test_random_samples(test_cases: list[dict[str, Any]], llm, context, validator):
	"""Test a random sampling of tasks across different websites"""
	import random

	logger.info('=== Testing Random Samples ===')

	# Take random samples
	samples = random.sample(test_cases, 1)

	for i, case in enumerate(samples, 1):
		task = f'Go to {case["website"]}.com and {case["confirmed_task"]}'
		logger.info(f'--- Random Sample {i}/{len(samples)} ---')
		logger.info(f'Task: {task}\n')

		agent = Agent(task, llm, browser_context=context)

		await agent.run()

		logger.info('Validating random sample task...')

		# TODO: Validate the task


def test_dataset_integrity(test_cases):
	"""Test the integrity of the test dataset"""
	logger.info('\n=== Testing Dataset Integrity ===')

	required_fields = ['website', 'confirmed_task', 'action_reprs']
	missing_fields = []

	logger.info(f'Checking {len(test_cases)} test cases for required fields')

	for i, case in enumerate(test_cases, 1):
		logger.debug(f'Checking case {i}/{len(test_cases)}')

		for field in required_fields:
			if field not in case:
				missing_fields.append(f'Case {i}: {field}')
				logger.warning(f"Missing field '{field}' in case {i}")

		# Type checks
		if not isinstance(case.get('confirmed_task'), str):
			logger.error(f"Case {i}: 'confirmed_task' must be string")
			assert False, 'Task must be string'

		if not isinstance(case.get('action_reprs'), list):
			logger.error(f"Case {i}: 'action_reprs' must be list")
			assert False, 'Actions must be list'

		if len(case.get('action_reprs', [])) == 0:
			logger.error(f"Case {i}: 'action_reprs' must not be empty")
			assert False, 'Must have at least one action'

	if missing_fields:
		logger.error('Dataset integrity check failed')
		assert False, f'Missing fields: {missing_fields}'
	else:
		logger.info('✅ Dataset integrity check passed')


if __name__ == '__main__':
	pytest.main([__file__, '-v'])

```

---

## `browser-use-main\tests\test_models.py`

```py
import asyncio
import os

import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import SecretStr

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig


@pytest.fixture(scope='function')
def event_loop():
	"""Create an instance of the default event loop for each test case."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='function')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as context:
		yield context


api_key_gemini = SecretStr(os.getenv('GOOGLE_API_KEY') or '')
api_key_deepseek = SecretStr(os.getenv('DEEPSEEK_API_KEY') or '')
api_key_anthropic = SecretStr(os.getenv('ANTHROPIC_API_KEY') or '')


# pytest -s -v tests/test_models.py
@pytest.fixture(
	params=[
		ChatOpenAI(model='gpt-4o'),
		ChatOpenAI(model='gpt-4o-mini'),
		AzureChatOpenAI(
			model='gpt-4o',
			api_version='2024-10-21',
			azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
			api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
		),
		# ChatOpenAI(
		# base_url='https://api.deepseek.com/v1',
		# model='deepseek-reasoner',
		# api_key=api_key_deepseek,
		# ),
		# run: ollama start
		ChatOllama(
			model='qwen2.5:latest',
			num_ctx=128000,
		),
		AzureChatOpenAI(
			model='gpt-4o-mini',
			api_version='2024-10-21',
			azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
			api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
		),
		ChatAnthropic(
			model_name='claude-3-5-sonnet-20240620',
			timeout=100,
			temperature=0.0,
			stop=None,
			api_key=api_key_anthropic,
		),
		ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=api_key_gemini),
		ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=api_key_gemini),
		ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', api_key=api_key_gemini),
		ChatOpenAI(
			base_url='https://api.deepseek.com/v1',
			model='deepseek-chat',
			api_key=api_key_deepseek,
		),
	],
	ids=[
		'gpt-4o',
		'gpt-4o-mini',
		'azure-gpt-4o',
		#'deepseek-reasoner',
		'qwen2.5:latest',
		'azure-gpt-4o-mini',
		'claude-3-5-sonnet',
		'gemini-2.0-flash-exp',
		'gemini-1.5-pro',
		'gemini-1.5-flash-latest',
		'deepseek-chat',
	],
)
async def llm(request):
	return request.param


@pytest.mark.asyncio
async def test_model_search(llm, context):
	"""Test 'Search Google' action"""
	model_name = llm.model if hasattr(llm, 'model') else llm.model_name
	print(f'\nTesting model: {model_name}')

	use_vision = True
	models_without_vision = ['deepseek-chat', 'deepseek-reasoner']
	if hasattr(llm, 'model') and llm.model in models_without_vision:
		use_vision = False
	elif hasattr(llm, 'model_name') and llm.model_name in models_without_vision:
		use_vision = False

	# require ollama run
	local_models = ['qwen2.5:latest']
	if model_name in local_models:
		# check if ollama is running
		# ping ollama http://127.0.0.1
		try:
			async with httpx.AsyncClient() as client:
				response = await client.get('http://127.0.0.1:11434/')
				if response.status_code != 200:
					raise Exception('Ollama is not running - start with `ollama start`')
		except Exception:
			raise Exception('Ollama is not running - start with `ollama start`')

	agent = Agent(
		task="Search Google for 'elon musk' then click on the first result and scroll down.",
		llm=llm,
		browser_context=context,
		max_failures=2,
		use_vision=use_vision,
	)
	history: AgentHistoryList = await agent.run(max_steps=2)
	done = history.is_done()
	successful = history.is_successful()
	action_names = history.action_names()
	print(f'Actions performed: {action_names}')
	errors = [e for e in history.errors() if e is not None]
	errors = '\n'.join(errors)
	passed = False
	if 'search_google' in action_names:
		passed = True
	elif 'go_to_url' in action_names:
		passed = True
	elif 'open_tab' in action_names:
		passed = True

	else:
		passed = False
	print(f'Model {model_name}: {"✅ PASSED - " if passed else "❌ FAILED - "} Done: {done} Successful: {successful}')

	assert passed, f'Model {model_name} not working\nActions performed: {action_names}\nErrors: {errors}'

```

---

## `browser-use-main\tests\test_qwen.py`

```py
import asyncio

import pytest
from langchain_ollama import ChatOllama

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig


@pytest.fixture
def llm():
	"""Initialize language model for testing"""

	# return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None)
	# NOTE: Make sure to run ollama server with `ollama start'
	return ChatOllama(
		model='qwen2.5:latest',
		num_ctx=128000,
	)


@pytest.fixture(scope='session')
def event_loop():
	"""Create an instance of the default event loop for each test case."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='session')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as context:
		yield context


# pytest tests/test_qwen.py -v -k "test_qwen_url" --capture=no
# @pytest.mark.asyncio
async def test_qwen_url(llm, context):
	"""Test complex ecommerce interaction sequence"""
	agent = Agent(
		task='go_to_url amazon.com',
		llm=llm,
	)

	history: AgentHistoryList = await agent.run(max_steps=3)

	# Verify sequence of actions
	action_sequence = []
	for action in history.model_actions():
		action_name = list(action.keys())[0]
		if action_name in ['go_to_url', 'open_tab']:
			action_sequence.append('navigate')

	assert 'navigate' in action_sequence  # Navigated to Amazon

```

---

## `browser-use-main\tests\test_react_dropdown.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList

llm = ChatOpenAI(model='gpt-4o')
# browser = Browser(config=BrowserConfig(headless=False))

agent = Agent(
	task=(
		'go to https://codepen.io/shyam-king/pen/ByBJoOv and select "Tiger" dropdown and read the text given in "Selected Animal" box (it can be empty as well)'
	),
	llm=llm,
	browser_context=BrowserContext(
		browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
	),
)


async def test_dropdown():
	history: AgentHistoryList = await agent.run(10)
	# await controller.browser.close(force=True)

	result = history.final_result()
	assert result is not None
	print('result: ', result)
	# await browser.close()


if __name__ == '__main__':
	asyncio.run(test_dropdown())

```

---

## `browser-use-main\tests\test_save_conversation.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import shutil
import sys

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList

llm = ChatOpenAI(model='gpt-4o')


async def test_save_conversation_contains_slash():
	if os.path.exists('./logs'):
		shutil.rmtree('./logs')

	agent = Agent(
		task=('go to google.com and search for text "hi there"'),
		llm=llm,
		browser_context=BrowserContext(
			browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
		),
		save_conversation_path='logs/conversation',
	)
	history: AgentHistoryList = await agent.run(20)

	result = history.final_result()
	assert result is not None

	assert os.path.exists('./logs'), 'logs directory was not created'
	assert os.path.exists('./logs/conversation_2.txt'), 'logs file was not created'


async def test_save_conversation_not_contains_slash():
	if os.path.exists('./logs'):
		shutil.rmtree('./logs')

	agent = Agent(
		task=('go to google.com and search for text "hi there"'),
		llm=llm,
		browser_context=BrowserContext(
			browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
		),
		save_conversation_path='logs',
	)
	history: AgentHistoryList = await agent.run(20)

	result = history.final_result()
	assert result is not None

	assert os.path.exists('./logs'), 'logs directory was not created'
	assert os.path.exists('./logs/_2.txt'), 'logs file was not created'


async def test_save_conversation_deep_directory():
	if os.path.exists('./logs'):
		shutil.rmtree('./logs')

	agent = Agent(
		task=('go to google.com and search for text "hi there"'),
		llm=llm,
		browser_context=BrowserContext(
			browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
		),
		save_conversation_path='logs/deep/directory/conversation',
	)
	history: AgentHistoryList = await agent.run(20)

	result = history.final_result()
	assert result is not None

	assert os.path.exists('./logs/deep/directory'), 'logs directory was not created'
	assert os.path.exists('./logs/deep/directory/conversation_2.txt'), 'logs file was not created'

```

---

## `browser-use-main\tests\test_self_registered_actions.py`

```py
import asyncio
import os

import pytest
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use.agent.service import Agent
from browser_use.agent.views import AgentHistoryList
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller


@pytest.fixture(scope='session')
def event_loop():
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='session')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as context:
		yield context


@pytest.fixture
async def controller():
	"""Initialize the controller with self-registered actions"""
	controller = Controller()

	# Define custom actions without Pydantic models
	@controller.action('Print a message')
	def print_message(message: str):
		print(f'Message: {message}')
		return f'Printed message: {message}'

	@controller.action('Add two numbers')
	def add_numbers(a: int, b: int):
		result = a + b
		return f'The sum is {result}'

	@controller.action('Concatenate strings')
	def concatenate_strings(str1: str, str2: str):
		result = str1 + str2
		return f'Concatenated string: {result}'

	# Define Pydantic models
	class SimpleModel(BaseModel):
		name: str
		age: int

	class Address(BaseModel):
		street: str
		city: str

	class NestedModel(BaseModel):
		user: SimpleModel
		address: Address

	# Add actions with Pydantic model arguments
	@controller.action('Process simple model', param_model=SimpleModel)
	def process_simple_model(model: SimpleModel):
		return f'Processed {model.name}, age {model.age}'

	@controller.action('Process nested model', param_model=NestedModel)
	def process_nested_model(model: NestedModel):
		user_info = f'{model.user.name}, age {model.user.age}'
		address_info = f'{model.address.street}, {model.address.city}'
		return f'Processed user {user_info} at address {address_info}'

	@controller.action('Process multiple models')
	def process_multiple_models(model1: SimpleModel, model2: Address):
		return f'Processed {model1.name} living at {model2.street}, {model2.city}'

	yield controller


@pytest.fixture
def llm():
	"""Initialize language model for testing"""

	# return ChatAnthropic(model_name='claude-3-5-sonnet-20240620', timeout=25, stop=None)
	return AzureChatOpenAI(
		model='gpt-4o',
		api_version='2024-10-21',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)


# @pytest.mark.skip(reason="Skipping test for now")
@pytest.mark.asyncio
async def test_self_registered_actions_no_pydantic(llm, controller):
	"""Test self-registered actions with individual arguments"""
	agent = Agent(
		task="First, print the message 'Hello, World!'. Then, add 10 and 20. Next, concatenate 'foo' and 'bar'.",
		llm=llm,
		controller=controller,
	)
	history: AgentHistoryList = await agent.run(max_steps=10)
	# Check that custom actions were executed
	action_names = history.action_names()

	assert 'print_message' in action_names
	assert 'add_numbers' in action_names
	assert 'concatenate_strings' in action_names


# @pytest.mark.skip(reason="Skipping test for now")
@pytest.mark.asyncio
async def test_mixed_arguments_actions(llm, controller):
	"""Test actions with mixed argument types"""

	# Define another action during the test
	# Test for async actions
	@controller.action('Calculate the area of a rectangle')
	async def calculate_area(length: float, width: float):
		area = length * width
		return f'The area is {area}'

	agent = Agent(
		task='Calculate the area of a rectangle with length 5.5 and width 3.2.',
		llm=llm,
		controller=controller,
	)
	history = await agent.run(max_steps=5)

	# Check that the action was executed
	action_names = history.action_names()

	assert 'calculate_area' in action_names
	# check result
	correct = 'The area is 17.6'
	for content in history.extracted_content():
		if correct in content:
			break
	else:
		pytest.fail(f'{correct} not found in extracted content')


@pytest.mark.asyncio
async def test_pydantic_simple_model(llm, controller):
	"""Test action with a simple Pydantic model argument"""
	agent = Agent(
		task="Process a simple model with name 'Alice' and age 30.",
		llm=llm,
		controller=controller,
	)
	history = await agent.run(max_steps=5)

	# Check that the action was executed
	action_names = history.action_names()

	assert 'process_simple_model' in action_names
	correct = 'Processed Alice, age 30'
	for content in history.extracted_content():
		if correct in content:
			break
	else:
		pytest.fail(f'{correct} not found in extracted content')


@pytest.mark.asyncio
async def test_pydantic_nested_model(llm, controller):
	"""Test action with a nested Pydantic model argument"""
	agent = Agent(
		task="Process a nested model with user name 'Bob', age 25, living at '123 Maple St', 'Springfield'.",
		llm=llm,
		controller=controller,
	)
	history = await agent.run(max_steps=5)

	# Check that the action was executed
	action_names = history.action_names()

	assert 'process_nested_model' in action_names
	correct = 'Processed user Bob, age 25 at address 123 Maple St, Springfield'
	for content in history.extracted_content():
		if correct in content:
			break
	else:
		pytest.fail(f'{correct} not found in extracted content')


# run this file with:
# pytest tests/test_self_registered_actions.py --capture=no

```

---

## `browser-use-main\tests\test_sensitive_data.py`

```py
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from browser_use.agent.message_manager.service import MessageManager, MessageManagerSettings
from browser_use.agent.views import MessageManagerState
from browser_use.controller.registry.service import Registry


class SensitiveParams(BaseModel):
	"""Test parameter model for sensitive data testing."""

	text: str = Field(description='Text with sensitive data placeholders')


@pytest.fixture
def registry():
	return Registry()


@pytest.fixture
def message_manager():
	return MessageManager(
		task='Test task',
		system_message=SystemMessage(content='System message'),
		settings=MessageManagerSettings(),
		state=MessageManagerState(),
	)


def test_replace_sensitive_data_with_missing_keys(registry):
	"""Test that _replace_sensitive_data handles missing keys gracefully"""
	# Create a simple Pydantic model with sensitive data placeholders
	params = SensitiveParams(text='Please enter <secret>username</secret> and <secret>password</secret>')

	# Case 1: All keys present
	sensitive_data = {'username': 'user123', 'password': 'pass456'}
	result = registry._replace_sensitive_data(params, sensitive_data)
	assert 'user123' in result.text
	assert 'pass456' in result.text
	# Both keys should be replaced

	# Case 2: One key missing
	sensitive_data = {'username': 'user123'}  # password is missing
	result = registry._replace_sensitive_data(params, sensitive_data)
	assert 'user123' in result.text
	assert '<secret>password</secret>' in result.text
	# Verify the behavior - username replaced, password kept as tag

	# Case 3: Multiple keys missing
	sensitive_data = {}  # both keys missing
	result = registry._replace_sensitive_data(params, sensitive_data)
	assert '<secret>username</secret>' in result.text
	assert '<secret>password</secret>' in result.text
	# Verify both tags are preserved when keys are missing

	# Case 4: One key empty
	sensitive_data = {'username': 'user123', 'password': ''}
	result = registry._replace_sensitive_data(params, sensitive_data)
	assert 'user123' in result.text
	assert '<secret>password</secret>' in result.text
	# Empty value should be treated the same as missing key


def test_filter_sensitive_data(message_manager):
	"""Test that _filter_sensitive_data handles all sensitive data scenarios correctly"""
	# Set up a message with sensitive information
	message = HumanMessage(content='My username is admin and password is secret123')

	# Case 1: No sensitive data provided
	message_manager.settings.sensitive_data = None
	result = message_manager._filter_sensitive_data(message)
	assert result.content == 'My username is admin and password is secret123'

	# Case 2: All sensitive data is properly replaced
	message_manager.settings.sensitive_data = {'username': 'admin', 'password': 'secret123'}
	result = message_manager._filter_sensitive_data(message)
	assert '<secret>username</secret>' in result.content
	assert '<secret>password</secret>' in result.content

	# Case 3: Make sure it works with nested content
	nested_message = HumanMessage(content=[{'type': 'text', 'text': 'My username is admin and password is secret123'}])
	result = message_manager._filter_sensitive_data(nested_message)
	assert '<secret>username</secret>' in result.content[0]['text']
	assert '<secret>password</secret>' in result.content[0]['text']

	# Case 4: Test with empty values
	message_manager.settings.sensitive_data = {'username': 'admin', 'password': ''}
	result = message_manager._filter_sensitive_data(message)
	assert '<secret>username</secret>' in result.content
	# Only username should be replaced since password is empty

```

---

## `browser-use-main\tests\test_service.py`

```py
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserStateSummary
from browser_use.controller.registry.service import Registry
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller

# run with python -m pytest tests/test_service.py


# run test with:
# python -m pytest tests/test_service.py
class TestAgent:
	@pytest.fixture
	def mock_controller(self):
		controller = Mock(spec=Controller)
		registry = Mock(spec=Registry)
		registry.registry = MagicMock()
		registry.registry.actions = {'test_action': MagicMock(param_model=MagicMock())}  # type: ignore
		controller.registry = registry
		return controller

	@pytest.fixture
	def mock_llm(self):
		return Mock(spec=BaseChatModel)

	@pytest.fixture
	def mock_browser(self):
		return Mock(spec=Browser)

	@pytest.fixture
	def mock_browser_context(self):
		return Mock(spec=BrowserContext)

	def test_convert_initial_actions(self, mock_controller, mock_llm, mock_browser, mock_browser_context):  # type: ignore
		"""
		Test that the _convert_initial_actions method correctly converts
		dictionary-based actions to ActionModel instances.

		This test ensures that:
		1. The method processes the initial actions correctly.
		2. The correct param_model is called with the right parameters.
		3. The ActionModel is created with the validated parameters.
		4. The method returns a list of ActionModel instances.
		"""
		# Arrange
		agent = Agent(
			task='Test task', llm=mock_llm, controller=mock_controller, browser=mock_browser, browser_context=mock_browser_context
		)
		initial_actions = [{'test_action': {'param1': 'value1', 'param2': 'value2'}}]

		# Mock the ActionModel
		mock_action_model = MagicMock(spec=ActionModel)
		mock_action_model_instance = MagicMock()
		mock_action_model.return_value = mock_action_model_instance
		agent.ActionModel = mock_action_model  # type: ignore

		# Act
		result = agent._convert_initial_actions(initial_actions)

		# Assert
		assert len(result) == 1
		mock_controller.registry.registry.actions['test_action'].param_model.assert_called_once_with(  # type: ignore
			param1='value1', param2='value2'
		)
		mock_action_model.assert_called_once()
		assert isinstance(result[0], MagicMock)
		assert result[0] == mock_action_model_instance

		# Check that the ActionModel was called with the correct parameters
		call_args = mock_action_model.call_args[1]
		assert 'test_action' in call_args
		assert call_args['test_action'] == mock_controller.registry.registry.actions['test_action'].param_model.return_value  # type: ignore

	@pytest.mark.asyncio
	async def test_step_error_handling(self):
		"""
		Test the error handling in the step method of the Agent class.
		This test simulates a failure in the get_next_action method and
		checks if the error is properly handled and recorded.
		"""
		# Mock the LLM
		mock_llm = MagicMock(spec=BaseChatModel)

		# Mock the MessageManager
		with patch('browser_use.agent.service.MessageManager') as mock_message_manager:
			# Create an Agent instance with mocked dependencies
			agent = Agent(task='Test task', llm=mock_llm)

			# Mock the get_next_action method to raise an exception
			agent.get_next_action = AsyncMock(side_effect=ValueError('Test error'))

			# Mock the browser_context
			agent.browser_context = AsyncMock()
			agent.browser_context.get_state_summary = AsyncMock(
				return_value=BrowserStateSummary(
					url='https://example.com',
					title='Example',
					element_tree=MagicMock(),  # Mocked element tree
					tabs=[],
					selector_map={},
					screenshot='',
				)
			)

			# Mock the controller
			agent.controller = AsyncMock()

			# Call the step method
			await agent.step()

			# Assert that the error was handled and recorded
			assert agent.consecutive_failures == 1
			assert len(agent._last_result) == 1
			assert isinstance(agent._last_result[0], ActionResult)
			assert 'Test error' in agent._last_result[0].error
			assert agent._last_result[0].include_in_memory is True


class TestRegistry:
	@pytest.fixture
	def registry_with_excludes(self):
		return Registry(exclude_actions=['excluded_action'])

	def test_action_decorator_with_excluded_action(self, registry_with_excludes):
		"""
		Test that the action decorator does not register an action
		if it's in the exclude_actions list.
		"""

		# Define a function to be decorated
		def excluded_action():
			pass

		# Apply the action decorator
		decorated_func = registry_with_excludes.action(description='This should be excluded')(excluded_action)

		# Assert that the decorated function is the same as the original
		assert decorated_func == excluded_action

		# Assert that the action was not added to the registry
		assert 'excluded_action' not in registry_with_excludes.registry.actions

		# Define another function that should be included
		def included_action():
			pass

		# Apply the action decorator to an included action
		registry_with_excludes.action(description='This should be included')(included_action)

		# Assert that the included action was added to the registry
		assert 'included_action' in registry_with_excludes.registry.actions

	@pytest.mark.asyncio
	async def test_execute_action_with_and_without_browser_context(self):
		"""
		Test that the execute_action method correctly handles actions with and without a browser context.
		This test ensures that:
		1. An action requiring a browser context is executed correctly.
		2. An action not requiring a browser context is executed correctly.
		3. The browser context is passed to the action function when required.
		4. The action function receives the correct parameters.
		5. The method raises an error when a browser context is required but not provided.
		"""
		registry = Registry()

		# Define a mock action model
		class TestActionModel(BaseModel):
			param1: str

		# Define mock action functions
		async def test_action_with_browser(param1: str, browser):
			return f'Action executed with {param1} and browser'

		async def test_action_without_browser(param1: str):
			return f'Action executed with {param1}'

		# Register the actions
		registry.registry.actions['test_action_with_browser'] = MagicMock(
			function=AsyncMock(side_effect=test_action_with_browser),
			param_model=TestActionModel,
			description='Test action with browser',
		)

		registry.registry.actions['test_action_without_browser'] = MagicMock(
			function=AsyncMock(side_effect=test_action_without_browser),
			param_model=TestActionModel,
			description='Test action without browser',
		)

		# Mock BrowserContext
… (truncated)

```

---

## `browser-use-main\tests\test_stress.py`

```py
import asyncio
import os
import random
import string
import time

import pytest
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.controller.service import Controller


@pytest.fixture(scope='session')
def event_loop():
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope='session')
async def browser(event_loop):
	browser_instance = Browser(
		config=BrowserConfig(
			headless=True,
		)
	)
	yield browser_instance
	await browser_instance.close()


@pytest.fixture
async def context(browser):
	async with await browser.new_context() as context:
		yield context


@pytest.fixture
def llm():
	"""Initialize the language model"""
	model = AzureChatOpenAI(
		api_version='2024-10-21',
		model='gpt-4o',
		azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT', ''),
		api_key=SecretStr(os.getenv('AZURE_OPENAI_KEY', '')),
	)
	return model


def generate_random_text(length: int) -> str:
	"""Generate random text of specified length"""
	return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))


@pytest.fixture
async def controller():
	"""Initialize the controller"""
	controller = Controller()
	large_text = generate_random_text(10000)

	@controller.action('call this magical function to get very special text')
	def get_very_special_text():
		return large_text

	yield controller


@pytest.mark.asyncio
async def test_token_limit_with_multiple_extractions(llm, controller, context):
	"""Test handling of multiple smaller extractions accumulating tokens"""
	agent = Agent(
		task='Call the magical function to get very special text 5 times',
		llm=llm,
		controller=controller,
		browser_context=context,
		max_input_tokens=2000,
		save_conversation_path='tmp/stress_test/test_token_limit_with_multiple_extractions.json',
	)

	history = await agent.run(max_steps=5)

	# check if 5 times called get_special_text
	calls = [a for a in history.action_names() if a == 'get_very_special_text']
	assert len(calls) == 5
	# check the message history should be max 3 messages
	assert len(agent.message_manager.history.messages) > 3


@pytest.mark.slow
@pytest.mark.parametrize('max_tokens', [4000])  # 8000 20000
@pytest.mark.asyncio
async def test_open_3_tabs_and_extract_content(llm, controller, context, max_tokens):
	"""Stress test: Open 3 tabs with urls and extract content"""
	agent = Agent(
		task='Open 3 tabs with https://en.wikipedia.org/wiki/Internet and extract the content from each.',
		llm=llm,
		controller=controller,
		browser_context=context,
		max_input_tokens=max_tokens,
		save_conversation_path='tmp/stress_test/test_open_3_tabs_and_extract_content.json',
	)
	start_time = time.time()
	history = await agent.run(max_steps=7)
	end_time = time.time()

	total_time = end_time - start_time

	print(f'Total time: {total_time:.2f} seconds')
	# Check for errors
	errors = history.errors()
	assert len(errors) == 0, 'Errors occurred during the test'
	# check if 3 tabs were opened
	assert len(context.current_state.tabs) >= 3, '3 tabs were not opened'

```

---

## `browser-use-main\tests\test_tab_management.py`

```py
import asyncio
import logging

import pytest
from dotenv import load_dotenv
from pytest_httpserver import HTTPServer

load_dotenv()

from browser_use.agent.views import ActionModel
from browser_use.browser.profile import BrowserProfile
from browser_use.browser.session import BrowserSession
from browser_use.controller.service import Controller

# Set up test logging
logger = logging.getLogger('tab_tests')
logger.setLevel(logging.DEBUG)


class TestTabManagement:
	"""Tests for the tab management system with separate agent_current_page and human_current_page references."""

	@pytest.fixture(scope='module')
	def event_loop(self):
		"""Create and provide an event loop for async tests."""
		loop = asyncio.get_event_loop_policy().new_event_loop()
		yield loop
		loop.close()

	@pytest.fixture(scope='module')
	def http_server(self):
		"""Create and provide a test HTTP server that serves static content."""
		server = HTTPServer()
		server.start()

		# Add routes for test pages
		server.expect_request('/page1').respond_with_data(
			'<html><head><title>Test Page 1</title></head><body><h1>Test Page 1</h1></body></html>', content_type='text/html'
		)
		server.expect_request('/page2').respond_with_data(
			'<html><head><title>Test Page 2</title></head><body><h1>Test Page 2</h1></body></html>', content_type='text/html'
		)
		server.expect_request('/page3').respond_with_data(
			'<html><head><title>Test Page 3</title></head><body><h1>Test Page 3</h1></body></html>', content_type='text/html'
		)
		server.expect_request('/page4').respond_with_data(
			'<html><head><title>Test Page 4</title></head><body><h1>Test Page 4</h1></body></html>', content_type='text/html'
		)

		yield server
		server.stop()

	@pytest.fixture(scope='module')
	async def browser_profile(self, event_loop):
		"""Create and provide a BrowserProfile with security disabled."""
		profile = BrowserProfile(headless=True)
		yield profile

	@pytest.fixture(scope='module')
	async def browser_session(self, browser_profile, http_server):
		"""Create and provide a BrowserSession instance with a properly initialized tab."""
		browser_session = BrowserSession(
			browser_profile=browser_profile,
			user_data_dir=None,
		)
		await browser_session.start()

		# Create an initial tab and wait for it to load completely
		base_url = f'http://{http_server.host}:{http_server.port}'
		await browser_session.new_tab(f'{base_url}/page1')
		await asyncio.sleep(1)  # Wait for the tab to fully initialize

		# Verify that agent_current_page and human_current_page are properly set
		assert browser_session.agent_current_page is not None
		assert browser_session.human_current_page is not None
		assert f'{http_server.host}:{http_server.port}' in browser_session.agent_current_page.url

		yield browser_session
		await browser_session.stop()

	@pytest.fixture
	def controller(self):
		"""Create and provide a Controller instance."""
		return Controller()

	@pytest.fixture
	def base_url(self, http_server):
		"""Return the base URL for the test HTTP server."""
		return f'http://{http_server.host}:{http_server.port}'

	# Helper methods

	async def _execute_action(self, controller, browser_session: BrowserSession, action_data):
		"""Generic helper to execute any action via the controller."""
		# Dynamically create an appropriate ActionModel class
		action_type = list(action_data.keys())[0]
		action_value = action_data[action_type]

		# Create the ActionModel with the single action field
		class DynamicActionModel(ActionModel):
			pass

		# Dynamically add the field with the right type annotation
		setattr(DynamicActionModel, action_type, type(action_value) | None)

		# Execute the action
		result = await controller.act(DynamicActionModel(**action_data), browser_session)

		# Give the browser a moment to process the action
		await asyncio.sleep(0.5)

		return result

	async def _reset_tab_state(self, browser_session: BrowserSession, base_url: str):
		browser_session.human_current_page = None
		browser_session.agent_current_page = None

		# close all existing tabs
		for page in browser_session.browser_context.pages:
			await page.close()

		await asyncio.sleep(0.5)

		# open one new tab and set it as the human_current_page & agent_current_page
		initial_tab = await browser_session.get_current_page()

		assert initial_tab is not None
		assert browser_session.human_current_page is not None
		assert browser_session.agent_current_page is not None
		assert browser_session.human_current_page.url == initial_tab.url
		assert browser_session.agent_current_page.url == initial_tab.url
		return initial_tab

	async def _simulate_human_tab_change(self, page, browser_session: BrowserSession):
		"""Simulate a user changing tabs by properly triggering events with Playwright."""

		logger.debug(
			f'BEFORE: agent_tab={browser_session.agent_current_page.url if browser_session.agent_current_page else "None"}, '
			f'human_current_page={browser_session.human_current_page.url if browser_session.human_current_page else "None"}'
		)
		logger.debug(f'Simulating user changing to -> {page.url}')

		# First bring the page to front - this is the physical action a user would take
		await page.bring_to_front()

		# To simulate a user switching tabs, we need to trigger the right events
		# Use Playwright's dispatch_event method to properly trigger events from outside

		await page.dispatch_event('body', 'focus')
		# await page.evaluate("""() => window.dispatchEvent(new Event('focus'))""")
		# await page.evaluate(
		# 	"""() => document.dispatchEvent(new Event('pointermove', { bubbles: true, cancelable: false, clientX: 0, clientY: 0 }))"""
		# )
		# await page.evaluate(
		# 	"() => document.dispatchEvent(new Event('deviceorientation', { bubbles: true, cancelable: false, alpha: 0, beta: 0, gamma: 0 }))"
		# )
		# await page.evaluate(
		# 	"""() => document.dispatchEvent(new Event('visibilitychange', { bubbles: true, cancelable: false }))"""
		# )
		# logger.debug('Dispatched window.focus event')

		# cheat for now, because playwright really messes with foreground tab detection
		# TODO: fix this properly by triggering the right events and detecting them in playwright
		if page.url == 'about:blank':
			raise Exception(
				'Cannot simulate tab change on about:blank because cannot execute JS to fire focus event on about:blank'
			)
		await page.evaluate("""async () => {
			return await window._BrowserUseonTabVisibilityChange({ bubbles: true, cancelable: false });
		}""")

		# Give the event handlers time to process
		await asyncio.sleep(0.5)

		logger.debug(
			f'AFTER: agent_tab URL={browser_session.agent_current_page.url if browser_session.agent_current_page else "None"}, '
			f'human_current_page URL={browser_session.human_current_page.url if browser_session.human_current_page else "None"}'
		)

	# Tab management tests

	@pytest.mark.asyncio
	async def test_initial_values(self, browser_session, base_url):
		"""Test that open_tab correctly updates both tab references."""

		await self._reset_tab_state(browser_session, base_url)

		initial_tab = await browser_session.get_current_page()
		assert initial_tab.url == 'about:blank'
		assert browser_session.human_current_page == initial_tab
		assert browser_session.agent_current_page == initial_tab

		for page in browser_session.browser_context.pages:
			await page.close()

		# should never be none even after all pages are closed
		current_tab = await browser_session.get_current_page()
		assert current_tab is not None
		assert current_tab.url == 'about:blank'

… (truncated)

```

---

## `browser-use-main\tests\test_url_allowlist_security.py`

```py
from browser_use.browser import BrowserProfile, BrowserSession


class TestUrlAllowlistSecurity:
	"""Tests for URL allowlist security bypass prevention and URL allowlist glob pattern matching."""

	def test_authentication_bypass_prevention(self):
		"""Test that the URL allowlist cannot be bypassed using authentication credentials."""
		# Create a context config with a sample allowed domain
		browser_profile = BrowserProfile(allowed_domains=['example.com'])
		browser_session = BrowserSession(browser_profile=browser_profile)

		# Security vulnerability test cases
		# These should all be detected as malicious despite containing "example.com"
		assert browser_session._is_url_allowed('https://example.com:password@malicious.com') is False
		assert browser_session._is_url_allowed('https://example.com@malicious.com') is False
		assert browser_session._is_url_allowed('https://example.com%20@malicious.com') is False
		assert browser_session._is_url_allowed('https://example.com%3A@malicious.com') is False

		# Make sure legitimate auth credentials still work
		assert browser_session._is_url_allowed('https://user:password@example.com') is True

	def test_glob_pattern_matching(self):
		"""Test that glob patterns in allowed_domains work correctly."""
		# Test *.example.com pattern (should match subdomains and main domain)
		browser_profile = BrowserProfile(allowed_domains=['*.example.com'])
		browser_session = BrowserSession(browser_profile=browser_profile)

		# Should match subdomains
		assert browser_session._is_url_allowed('https://sub.example.com') is True
		assert browser_session._is_url_allowed('https://deep.sub.example.com') is True

		# Should also match main domain
		assert browser_session._is_url_allowed('https://example.com') is True

		# Should not match other domains
		assert browser_session._is_url_allowed('https://notexample.com') is False
		assert browser_session._is_url_allowed('https://example.org') is False

		# Test more complex glob patterns
		browser_profile = BrowserProfile(allowed_domains=['*google.com', 'wiki*'])
		browser_session = BrowserSession(browser_profile=browser_profile)

		# Should match domains ending with google.com
		assert browser_session._is_url_allowed('https://google.com') is True
		assert browser_session._is_url_allowed('https://www.google.com') is True
		assert browser_session._is_url_allowed('https://anygoogle.com') is True

		# Should match domains starting with wiki
		assert browser_session._is_url_allowed('https://wiki.org') is True
		assert browser_session._is_url_allowed('https://wikipedia.org') is True

		# Should not match other domains
		assert browser_session._is_url_allowed('https://example.com') is False

		# Test browser internal URLs
		assert browser_session._is_url_allowed('chrome://settings') is True
		assert browser_session._is_url_allowed('about:blank') is True

		# Test security for glob patterns (authentication credentials bypass attempts)
		# These should all be detected as malicious despite containing allowed domain patterns
		assert browser_session._is_url_allowed('https://allowed.example.com:password@notallowed.com') is False
		assert browser_session._is_url_allowed('https://subdomain.example.com@evil.com') is False
		assert browser_session._is_url_allowed('https://sub.example.com%20@malicious.org') is False
		assert browser_session._is_url_allowed('https://anygoogle.com@evil.org') is False

	def test_glob_pattern_edge_cases(self):
		"""Test edge cases for glob pattern matching to ensure proper behavior."""
		# Test with domains containing glob pattern in the middle
		browser_profile = BrowserProfile(allowed_domains=['*google.com', 'wiki*'])
		browser_session = BrowserSession(browser_profile=browser_profile)

		# Verify that 'wiki*' pattern doesn't match domains that merely contain 'wiki' in the middle
		assert browser_session._is_url_allowed('https://notawiki.com') is False
		assert browser_session._is_url_allowed('https://havewikipages.org') is False
		assert browser_session._is_url_allowed('https://my-wiki-site.com') is False

		# Verify that '*google.com' doesn't match domains that have 'google' in the middle
		assert browser_session._is_url_allowed('https://mygoogle.company.com') is False

		# Create context with potentially risky glob pattern that demonstrates security concerns
		browser_profile = BrowserProfile(allowed_domains=['*.google.*'])
		browser_session = BrowserSession(browser_profile=browser_profile)

		# Should match legitimate Google domains
		assert browser_session._is_url_allowed('https://www.google.com') is True
		assert browser_session._is_url_allowed('https://mail.google.co.uk') is True

		# But could also match potentially malicious domains with a subdomain structure
		# This demonstrates why such wildcard patterns can be risky
		assert browser_session._is_url_allowed('https://www.google.evil.com') is True

```

---

## `browser-use-main\tests\test_vision.py`

```py
"""
Simple try of the agent.

@dev You need to add OPENAI_API_KEY to your environment variables.
"""

import os
import sys
from pprint import pprint

import pytest

from browser_use.browser.browser import Browser, BrowserConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent, AgentHistoryList, Controller

llm = ChatOpenAI(model='gpt-4o')
controller = Controller()

# use this test to ask the model questions about the page like
# which color do you see for bbox labels, list all with their label
# what's the smallest bboxes with labels and


@controller.registry.action(description='explain what you see on the screen and ask user for input')
async def explain_screen(text: str) -> str:
	pprint(text)
	answer = input('\nuser input next question: \n')
	return answer


@controller.registry.action(description='done')
async def done(text: str) -> str:
	# pprint(text)
	return 'call explain_screen'


@pytest.fixture(scope='function')
def event_loop():
	"""Create an instance of the default event loop for each test case."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.mark.skip(reason='this is for local testing only')
async def test_vision():
	agent = Agent(
		task='call explain_screen all the time the user asks you questions e.g. about the page like bbox which you see are labels  - your task is to explain it and get the next question',
		llm=llm,
		controller=controller,
		browser=Browser(config=BrowserConfig(disable_security=True, headless=False)),
	)
	try:
		history: AgentHistoryList = await agent.run(20)
	finally:
		# Make sure to close the browser
		await agent.browser.close()

```

---

## `browser-use-main\tests\test_wait_for_element.py`

```py
import asyncio
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Third-party imports
from browser_use import Agent, Controller

# Local imports
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

# Load environment variables.
load_dotenv()

# Initialize language model and controller.
llm = ChatOpenAI(model='gpt-4o')
controller = Controller()


@pytest.mark.skip(reason='this is for local testing only')
async def test_wait_for_element():
	"""Test 'Wait for element' action."""

	initial_actions = [
		{'open_tab': {'url': 'https://pypi.org/'}},
		# Uncomment the line below to include the wait action in initial actions.
		# {'wait_for_element': {'selector': '#search', 'timeout': 30}},
	]

	# Set up the browser context.
	context = BrowserContext(
		browser=Browser(config=BrowserConfig(headless=False, disable_security=True)),
	)

	# Create the agent with the task.
	agent = Agent(
		task="Wait for element '#search' to be visible with a timeout of 30 seconds.",
		llm=llm,
		browser_context=context,
		initial_actions=initial_actions,
		controller=controller,
	)

	# Run the agent for a few steps to trigger navigation and then the wait action.
	history = await agent.run(max_steps=3)
	action_names = history.action_names()

	# Ensure that the wait_for_element action was executed.
	assert 'wait_for_element' in action_names, 'Expected wait_for_element action to be executed.'

	# Verify that the #search element is visible by querying the page.
	page = await context.get_current_page()
	header_handle = await page.query_selector('#search')
	assert header_handle is not None, 'Expected to find a #search element on the page.'
	is_visible = await header_handle.is_visible()
	assert is_visible, 'Expected the #search element to be visible.'


if __name__ == '__main__':
	asyncio.run(test_wait_for_element())

```

---

## `browser_use_docs.md`

```md
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
… (truncated)

```

---

## `contact_controller.py`

```py
from browser_use import Controller, ActionResult, Browser
from contact_info import ContactInfo, ContactSearchResults
from datetime import datetime
import re
from typing import List, Optional
from urllib.parse import urlparse
import json
from pathlib import Path
from playwright.async_api import Page

class ContactController(Controller):
    def __init__(self, output_file: str = "contacts.json"):
        super().__init__(output_model=ContactSearchResults)
        self.output_file = output_file
        self.contacts: List[ContactInfo] = []
        self.search_query: Optional[str] = None
        
    def save_contacts(self):
        """Save contacts to JSON file"""
        results = ContactSearchResults(
            contacts=self.contacts,
            search_query=self.search_query or "",
            total_found=len(self.contacts),
            search_completed_at=datetime.now()
        )
        
        with open(self.output_file, 'w') as f:
            json.dump(results.model_dump(), f, indent=2, default=str)
        
        return f"Saved {len(self.contacts)} contacts to {self.output_file}"

    @Controller.action("Extract contact information from current page")
    async def extract_contacts(self, browser: Browser) -> str:
        page = await browser.get_current_page()
        content = await page.content()
        
        # Extract emails
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content)
        
        # Extract phone numbers (basic pattern)
        phones = re.findall(r'\+?[\d\s-()]{10,}', content)
        
        # Extract social media links
        social_patterns = [
            r'https?://(?:www\.)?linkedin\.com/[^\s<>"]+',
            r'https?://(?:www\.)?twitter\.com/[^\s<>"]+',
            r'https?://(?:www\.)?facebook\.com/[^\s<>"]+',
            r'https?://(?:www\.)?instagram\.com/[^\s<>"]+'
        ]
        social_links = []
        for pattern in social_patterns:
            social_links.extend(re.findall(pattern, content))
        
        # Create contact info
        contact = ContactInfo(
            email=emails[0] if emails else None,
            phone=phones[0] if phones else None,
            social_links=social_links,
            source_url=page.url,
            found_at=datetime.now()
        )
        
        if contact.email or contact.phone or contact.social_links:
            self.contacts.append(contact)
            return f"Found contact info: {contact.model_dump_json()}"
        
        return "No contact information found on this page"

    @Controller.action("Search for contact information")
    async def search_contacts(self, query: str, browser: Browser) -> str:
        self.search_query = query
        page = await browser.get_current_page()
        
        # Search for contact-related terms
        search_terms = [
            "contact us",
            "about us",
            "team",
            "staff",
            "people",
            "contact information",
            "get in touch"
        ]
        
        for term in search_terms:
            try:
                # Use a more reliable selector
                elements = await page.query_selector_all(f'text="{term}"')
                if elements:
                    await elements[0].click()
                    await page.wait_for_load_state('networkidle')
                    await self.extract_contacts(browser)
            except Exception as e:
                print(f"Error clicking {term}: {str(e)}")
                continue
        
        return self.save_contacts()

    @Controller.action("Follow contact page links")
    async def follow_contact_links(self, browser: Browser) -> str:
        page = await browser.get_current_page()
        
        # Look for common contact page link patterns
        contact_links = await page.query_selector_all('a[href*="contact"], a[href*="about"], a[href*="team"]')
        
        for link in contact_links:
            try:
                href = await link.get_attribute('href')
                if href:
                    # Use the same page instead of creating new ones
                    await page.goto(href)
                    await page.wait_for_load_state('networkidle')
                    await self.extract_contacts(browser)
            except Exception as e:
                print(f"Error following link: {str(e)}")
                continue
        
        return self.save_contacts() 
```

---

## `contact_finder.py`

```py
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from contact_controller import ContactController
from dotenv import load_dotenv
import asyncio
import argparse
from playwright.async_api import BrowserContext
import streamlit as st

load_dotenv()

async def find_contacts(search_query: str, output_file: str = "contacts.json", max_steps: int = 20, headless: bool = False):
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o")
    
    # Create custom controller
    controller = ContactController(output_file=output_file)
    
    # Configure browser settings
    browser_config = BrowserConfig(
        headless=headless,
        new_context_config=BrowserContextConfig(
            window_width=1280,
            window_height=800,
            wait_for_network_idle_page_load_time=2.0
        )
    )
    
    # Create a single browser instance
    browser = Browser(config=browser_config)
    
    try:
        # Create a single browser context
        async with await browser.new_context() as context:
            # Create the agent with the shared context
            agent = Agent(
                task=f"Search for contact information about '{search_query}'. Look for emails, phone numbers, and social media links. Visit multiple pages and save all found contact information.",
                llm=llm,
                controller=controller,
                browser_context=context  # Use the shared context
            )
            
            # Run the agent with progress tracking
            history = await agent.run(max_steps=max_steps)
            
            # Print results
            if history.final_result():
                if 'st' in globals():
                    st.success(f"Contact search completed. Results saved to {output_file}")
                else:
                    print(f"\nContact search completed. Results saved to {output_file}")
            else:
                if 'st' in globals():
                    st.warning("No contact information found.")
                else:
                    print("\nNo contact information found.")
                    
            return history
    finally:
        # Ensure browser is closed
        await browser.close()

def main():
    parser = argparse.ArgumentParser(description="Find contact information for a given search query")
    parser.add_argument("query", help="The search query to find contact information for")
    parser.add_argument("--output", "-o", default="contacts.json", help="Output file for contact information")
    parser.add_argument("--max-steps", "-m", type=int, default=20, help="Maximum number of search steps")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    
    args = parser.parse_args()
    
    asyncio.run(find_contacts(
        args.query,
        args.output,
        max_steps=args.max_steps,
        headless=args.headless
    ))

if __name__ == "__main__":
    main() 
```

---

## `contact_finder_ui.py`

```py
import streamlit as st
import asyncio
import json
from datetime import datetime
from contact_finder import find_contacts
from contact_info import ContactInfo, ContactSearchResults
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Contact Finder",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .contact-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🔍 Contact Finder")
st.markdown("""
    Find contact information from websites automatically. Enter a search query and the tool will search for:
    - Email addresses
    - Phone numbers
    - Social media profiles
    - Contact pages
""")

# Create two columns for input and results
col1, col2 = st.columns([1, 2])

with col1:
    # Search input
    search_query = st.text_input("Search Query", placeholder="Enter company name or website...")
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        output_file = st.text_input("Output File", value="contacts.json")
        max_steps = st.slider("Maximum Search Steps", 5, 30, 20)
        headless = st.checkbox("Run in Background", value=False)
    
    # Search button
    if st.button("🔍 Start Search", type="primary"):
        if search_query:
            with st.spinner("Searching for contacts..."):
                # Create a unique output file for this search
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"contacts_{timestamp}.json"
                
                # Run the search
                asyncio.run(find_contacts(
                    search_query=search_query,
                    output_file=output_path
                ))
                
                # Store the output path in session state
                st.session_state.last_output = output_path
                st.session_state.search_completed = True
        else:
            st.error("Please enter a search query")

with col2:
    # Results section
    st.subheader("Search Results")
    
    if 'search_completed' in st.session_state and st.session_state.search_completed:
        try:
            # Load and display results
            with open(st.session_state.last_output, 'r') as f:
                results = json.load(f)
                
            # Display summary
            st.success(f"Found {results['total_found']} contacts!")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Contact List", "Data Table"])
            
            with tab1:
                # Display contacts in cards
                for contact in results['contacts']:
                    with st.container():
                        st.markdown(f"""
                            <div class="contact-card">
                                <h4>Contact Information</h4>
                                <p><strong>Source:</strong> {contact['source_url']}</p>
                                {f"<p><strong>Email:</strong> {contact['email']}</p>" if contact['email'] else ""}
                                {f"<p><strong>Phone:</strong> {contact['phone']}</p>" if contact['phone'] else ""}
                                {f"<p><strong>Social Links:</strong> {', '.join(contact['social_links'])}</p>" if contact['social_links'] else ""}
                                <p><strong>Found at:</strong> {contact['found_at']}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                # Convert to DataFrame for better display
                df = pd.DataFrame(results['contacts'])
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"contacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
    else:
        st.info("Enter a search query and click 'Start Search' to begin")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Contact Finder - Powered by Browser Use</p>
    </div>
""", unsafe_allow_html=True) 
```

---

## `contact_info.py`

```py
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[HttpUrl] = None
    social_links: List[HttpUrl] = []
    source_url: HttpUrl
    found_at: datetime
    additional_info: Optional[str] = None

class ContactSearchResults(BaseModel):
    contacts: List[ContactInfo]
    search_query: str
    total_found: int
    search_completed_at: datetime 
```

---

## `contacts.json`

```json
[
  {
    "role": "Unknown",
    "name": "",
    "email": "glassanimals@umgstores.com",
    "phone": "",
    "source": "https://www.google.com/search?q=Glass%20Animals%20contact%20information&udm=14&sei=mz0vaLLFGOHQ2roP0K6-kAk",
    "target": "As you search for contact information for glass animals contact, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:07:34.880941"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "communications@umusic.com",
    "phone": "",
    "source": "https://www.google.com/search?q=Polydor%20contact%20email%20for%20representatives&udm=14",
    "target": "As you search for contact information for glass animals contact, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:08:11.748115"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "info@polydor.co.uk",
    "phone": "",
    "source": "https://www.google.com/search?q=Polydor%20contact%20email%20for%20representatives&udm=14",
    "target": "As you search for contact information for glass animals contact, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:08:11.748128"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "",
    "phone": "+44 (0)20 3932 8400",
    "source": "https://www.google.com/search?q=Polydor%20contact%20email%20for%20representatives&udm=14",
    "target": "As you search for contact information for glass animals contact, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:08:11.748135"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "",
    "phone": "+44 (0)20 3932 8400\n4",
    "source": "https://polydor.co.uk/pages/contact",
    "target": "As you search for contact information for glass animals contact, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:08:46.007304"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "CorporateEvents@teamwass.com",
    "phone": "",
    "source": "https://www.teamwass.com/music/denzel-curry/",
    "target": "As you search for contact information for denzel curry management, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:45:27.860341"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "Casinos@teamwass.com",
    "phone": "",
    "source": "https://www.teamwass.com/music/denzel-curry/",
    "target": "As you search for contact information for denzel curry management, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:45:27.860351"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "FairsAndFestivals@teamwass.com",
    "phone": "",
    "source": "https://www.teamwass.com/music/denzel-curry/",
    "target": "As you search for contact information for denzel curry management, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:45:27.860353"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "Colleges@teamwass.com",
    "phone": "",
    "source": "https://www.teamwass.com/music/denzel-curry/",
    "target": "As you search for contact information for denzel curry management, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:45:27.860354"
  },
  {
    "role": "Unknown",
    "name": "",
    "email": "BrandPartnerships@teamwass.com",
    "phone": "",
    "source": "https://www.teamwass.com/music/denzel-curry/",
    "target": "As you search for contact information for denzel curry management, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted.",
    "timestamp": "2025-05-22T15:45:27.860356"
  },
  {
    "name": "Mike Malak",
    "email": "mike.malak@teamwass.com",
    "source": "https://www.google.com/search?q=Denzel%20Curry%20management%20contact%20information&udm=14&sei=lVQvaPqgAr2YnesPjpeM2Qo",
    "target": "As you search for contact information for denzel curry management, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). Continue searching and updating the list until no more new contacts are found or all reasonable sources are ex
… (truncated)

```

---

## `contacts_20250522_213236.json`

```json
[]
```

---

## `extract_project_snapshot.py`

```py
#!/usr/bin/env python3
"""
extract_project_snapshot.py
Create a concise Markdown digest of the key files in a project
so it can be dropped into ChatGPT for README generation.

• Scans recursively from a project root you pass on the CLI
• Keeps only “important” file types (.py, .md, .toml, .txt, .html, .js, .json)
• Skips bulky or irrelevant folders (.git, .venv, node_modules, etc.)
• Truncates each file to avoid oversize pastes (200 lines / 8 kB max)
• Writes everything into project_snapshot.md with code-fenced sections
"""

from pathlib import Path
import argparse

IMPORTANT_EXTENSIONS = {".py", ".md", ".toml", ".txt", ".html", ".js", ".json"}
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

MAX_LINES = 200       # hard cap per file
MAX_CHARS = 8_000     # safety cap in case of long single-line files


def collect_files(root: Path):
    """Yield files worth including, respecting extension and exclude lists."""
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix in IMPORTANT_EXTENSIONS
            and not any(part in EXCLUDE_DIRS for part in path.parts)
        ):
            yield path


def abbreviated_text(path: Path):
    """Return at most MAX_LINES and MAX_CHARS worth of the file’s contents."""
    text = path.read_text("utf-8", errors="ignore")
    if len(text) > MAX_CHARS:
        text = text[: MAX_CHARS] + "\n… (truncated)\n"
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        text = "\n".join(lines[:MAX_LINES]) + "\n… (truncated)\n"
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown digest of a project’s important files."
    )
    parser.add_argument("project_root", type=Path, help="Path to project root")
    parser.add_argument(
        "-o",
        "--output",
        default="project_snapshot.md",
        help="Markdown file to write (default: project_snapshot.md)",
    )
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"{root} is not a directory")

    with open(args.output, "w", encoding="utf-8") as md:
        md.write(f"# Snapshot of **{root.name}**\n")
        md.write(
            "Auto-generated digest of key files. Paste this into ChatGPT to help craft a README.\n\n"
        )

        for file_path in sorted(collect_files(root)):
            rel = file_path.relative_to(root)
            md.write(f"---\n\n## `{rel}`\n\n```{file_path.suffix.lstrip('.')}\n")
            md.write(abbreviated_text(file_path))
            md.write("\n```\n\n")

    print(f"✅  Snapshot written to {args.output}")


if __name__ == "__main__":
    main()

```

---

## `project_snapshot.md`

```md
# Snapshot of **browser-use-main**
Auto-generated digest of key files. Paste this into ChatGPT to help craft a README.

---

## `app.py`

```py
from flask import Flask, render_template, request, jsonify, Response
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
import re
import queue
import threading
from cryptography.fernet import Fernet
import base64
from pathlib import Path

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global queue to store search progress updates
search_progress = queue.Queue()

# Initialize encryption key
def get_or_create_key():
    key_file = Path('.api_key')
    if key_file.exists():
        return key_file.read_bytes()
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    return key

# Initialize Fernet cipher
cipher = Fernet(get_or_create_key())

def encrypt_api_key(api_key):
    return cipher.encrypt(api_key.encode()).decode()

def decrypt_api_key(encrypted_key):
    try:
        return cipher.decrypt(encrypted_key.encode()).decode()
    except Exception:
        return None

@app.route('/')
def home():
    logger.info('Rendering home page')
    return render_template('index.html')

def extract_emails_and_phones(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'(\+?\d[\d\s().-]{7,}\d)'
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    return emails, phones

def deduplicate_contacts(existing, new_contacts):
    # Use a set of (email, phone) tuples for deduplication
    seen = set((c.get('email',''), c.get('phone','')) for c in existing)
    unique = []
    for c in new_contacts:
        key = (c.get('email',''), c.get('phone',''))
        if key not in seen and (c.get('email') or c.get('phone')):
            unique.append(c)
            seen.add(key)
    return existing + unique

def save_contacts_stepwise(target, contacts):
    try:
        with open("contacts.json", "r") as f:
            all_contacts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_contacts = []
    for contact in contacts:
        contact['target'] = target
        contact['timestamp'] = datetime.utcnow().isoformat()
    all_contacts = deduplicate_contacts(all_contacts, contacts)
    with open("contacts.json", "w") as f:
        json.dump(all_contacts, f, indent=2)

@app.route('/contacts', methods=['GET'])
def get_contacts():
    try:
        with open("contacts.json", "r") as f:
            contacts = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        contacts = []
    return jsonify(contacts)

@app.route('/search-progress')
def search_progress_stream():
    def generate():
        while True:
            try:
                # Get the next progress update from the queue
                progress = search_progress.get(timeout=30)  # 30 second timeout
                yield f"data: {json.dumps(progress)}\n\n"
            except queue.Empty:
                # Send a heartbeat to keep the connection alive
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/search', methods=['POST'])
def search():
    logger.info('Received search request')
    data = request.get_json()
    api_key = data.get('api_key')
    query = data.get('query')
    logger.info(f'API key provided: {bool(api_key)}, Query: {query}')

    if not api_key or not query:
        logger.warning('Missing API key or query')
        return jsonify({'error': 'API key and query are required'}), 400

    os.environ['OPENAI_API_KEY'] = api_key
    logger.info('Set OpenAI API key')

    model = ChatOpenAI(model="gpt-4o")
    logger.info('Initialized ChatOpenAI model')

    robust_prompt = (
        f"As you search for contact information for {query}, collect every email address and phone number you find, including those for managers, assistants, agencies, or related contacts. "
        "After each relevant page or step, output a JSON array of all contacts found so far, with fields: 'role', 'name', 'email', 'phone', and 'source' (the URL or page name). "
        "Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted."
    )

    async def on_step_end(agent):
        history = agent.state.history
        extracted = history.extracted_content()
        source = history.urls()[-1] if history.urls() else ""
        found_contacts = []
        if extracted:
            last = extracted[-1]
            # Try to parse JSON array as before
            try:
                start = last.find('[')
                end = last.rfind(']')
                if start != -1 and end != -1:
                    json_str = last[start:end+1]
                    contacts = json.loads(json_str)
                    for c in contacts:
                        if 'source' not in c:
                            c['source'] = source
                    found_contacts.extend(contacts)
            except Exception as e:
                logger.error(f'Error parsing/saving contacts: {e}')
            # Always scan for emails/phones in the text
            emails, phones = extract_emails_and_phones(last)
            for email in emails:
                found_contacts.append({
                    'role': 'Unknown',
                    'name': '',
                    'email': email,
                    'phone': '',
                    'source': source
                })
            for phone in phones:
                found_contacts.append({
                    'role': 'Unknown',
                    'name': '',
                    'email': '',
                    'phone': phone,
                    'source': source
                })
        if found_contacts:
            save_contacts_stepwise(agent.task, found_contacts)
            # Send progress update
            search_progress.put({
                'type': 'contacts_found',
                'contacts': found_contacts,
                'source': source
            })

    async def run_agent():
        agent = Agent(task=robust_prompt, llm=model)
        logger.info('Initialized Agent')
        # Send initial progress update
        search_progress.put({
            'type': 'status',
            'message': 'Starting search...'
        })
        result = await agent.run(on_step_end=on_step_end, max_steps=100)
        logger.info('Agent run complete')
        # Send final progress update
        search_progress.put({
            'type': 'status',
            'message': 'Search complete'
        })
        if hasattr(result, 'final_result'):
            final = result.final_result()
… (truncated)

```

---

## `README.md`

```md
# Contact Finder Agent

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/boshjerns/Browser-Use-Quick-Contact-Search-Module-)

A powerful web application built on top of [browser-use](https://github.com/your-username/browser-use) that leverages AI to automatically search for and collect contact information from across the web. This application uses an intelligent agent to find, extract, and save contact details including email addresses and phone numbers.

## Features

- 🔍 **Intelligent Web Search**: Uses AI-powered agent to search the web for contact information
- 📧 **Contact Extraction**: Automatically extracts emails and phone numbers from web pages
- 💾 **Contact Management**: Saves and deduplicates contacts in a structured JSON format
- 🔄 **Real-time Updates**: Continuously updates contact information as new sources are found
- 🎯 **Targeted Search**: Search for specific individuals, companies, or roles
- 📊 **Source Tracking**: Maintains records of where each contact was found

## Technical Stack

- **Backend**: Flask (Python)
- **AI/ML**: 
  - OpenAI GPT-4
  - LangChain
  - browser-use agent framework
- **Web Automation**: Playwright
- **Data Storage**: JSON-based local storage

## Prerequisites

- Python 3.8+
- OpenAI API key
- Modern web browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/boshjerns/Browser-Use-Quick-Contact-Search-Module-.git
cd Browser-Use-Quick-Contact-Search-Module-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter your search query and OpenAI API key in the web interface
   - The API key is handled securely through the frontend interface
   - No need to set up environment variables

4. The agent will automatically:
   - Search the web for relevant information
   - Extract contact details
   - Save unique contacts to the database
   - Display results in real-time

## How It Works

1. **Search Initiation**: When you submit a search query, the application creates an AI agent with specific instructions to find contact information.

2. **Web Navigation**: The agent uses browser-use to navigate the web, visiting relevant pages and extracting information.

3. **Contact Extraction**: The system automatically extracts:
   - Email addresses
   - Phone numbers
   - Role information
   - Source URLs

4. **Data Processing**: 
   - Contacts are deduplicated
   - Information is structured and validated
   - Results are saved to `contacts.json`

5. **Real-time Updates**: The application continuously updates the contact list as new information is found.

## API Endpoints

- `GET /contacts`: Retrieve all saved contacts
- `POST /search`: Initiate a new contact search
  - Required parameters:
    - `api_key`: Your OpenAI API key (provided through frontend)
    - `query`: Search query for finding contacts

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the [browser-use](https://github.com/your-username/browser-use) framework
- Powered by OpenAI's GPT-4
- Uses LangChain for AI/ML operations

```

---

## `requirements.txt`

```txt
flask==3.0.2
python-dotenv==1.0.1
browser-use
langchain-openai
playwright 
```

---

## `templates\index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1.0" />
<title>Contact Search Console</title>

<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">

<style>
/* ---------- palette & base ---------- */
:root{
  --win-grey:#c0c0c0;
  --win-blue:#0a2fa5;         /* title-bar blue */
  --btn-face:#e0e0e0;
  --btn-shadow:#808080;
  --btn-hilite:#fff;
}
*{box-sizing:border-box;font-family:system-ui,Segoe UI,Tahoma,Verdana,Arial,sans-serif;}
body{
  margin:0;
  background:var(--win-grey);
  display:flex;
  justify-content:center;
  align-items:flex-start;   /* change to center for full V-centering */
  min-height:100vh;
  padding:40px 12px;
}

/* ---------- “window” ---------- */
.container{
  width:100%;max-width:600px;background:#fff;
  border:2px solid #000;                /* outer frame */
  box-shadow:inset 1px 1px 0 var(--btn-hilite),
             inset -1px -1px 0 var(--btn-shadow);
}

/* title bar */
.title-bar{
  background:var(--win-blue);
  color:#fff;
  padding:6px 10px;
  font-weight:bold;
  font-size:0.9rem;
  letter-spacing:0.5px;
  display:flex;justify-content:center;
}

/* content padding */
.inner{
  padding:24px 20px;
}

/* ---------- inputs ---------- */
.input, .btn{
  width:100%;
  font-size:1rem;
  margin-bottom:16px;
}
.input{
  padding:8px 10px;
  border:2px solid #000;
  background:var(--btn-face);
  box-shadow:inset 1px 1px 0 var(--btn-hilite),
             inset -1px -1px 0 var(--btn-shadow);
}
.input:focus{outline:none;border-color:var(--win-blue);}

/* ---------- button ---------- */
.btn{
  padding:10px 0;
  background:var(--btn-face);
  border:2px solid #000;
  cursor:pointer;
  box-shadow:1px 1px 0 var(--btn-shadow),
             -1px -1px 0 var(--btn-hilite);
  font-weight:600;
}
.btn:hover{background:#d7d7d7;}

/* ---------- log & results ---------- */
.box{
  background:#fff;
  border:2px solid #000;
  box-shadow:inset 1px 1px 0 var(--btn-hilite),
             inset -1px -1px 0 var(--btn-shadow);
  padding:10px;
  word-break:break-word;
}
.log{height:90px;overflow-y:auto;font-size:0.85rem;margin-bottom:16px;}
.results{display:none;margin-top:16px;}
.loading{font-weight:bold;text-align:center;margin:12px 0;}

/* ---------- contacts table ---------- */
#contacts-section h2{
  text-align:center;margin:26px 0 10px;font-size:1rem;
}
.contacts-wrapper{
  max-height:260px;overflow-y:auto;
  border:2px solid #000;
  box-shadow:inset 1px 1px 0 var(--btn-hilite),
             inset -1px -1px 0 var(--btn-shadow);
}
#contacts-table{
  width:100%;border-collapse:collapse;font-size:.9rem;
}
#contacts-table thead{
  background:#d9d9d9;
}
#contacts-table th,#contacts-table td{
  border:1px solid #808080;
  padding:6px 8px;
}
#contacts-table tbody tr:nth-child(odd){
  background:#f0f0f0;
}
#contacts-table td{
  max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}

.api-key-section {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.api-key-section .input {
  flex: 1;
}

.api-key-section .btn {
  flex: 0 0 auto;
  white-space: nowrap;
}
</style>
</head>
<body>

<div class="container">
  <div class="title-bar">Contact Search Console</div>
  <div class="inner">
    <div class="api-key-section">
      <input id="apiKey" class="input" type="password" placeholder="Enter OpenAI API Key">
      <button class="btn" onclick="saveApiKey()">Save API Key</button>
      <button class="btn" onclick="clearApiKey()">Clear Saved Key</button>
    </div>

    <input id="query"  class="input" type="text" placeholder="Search for a contact…">
    <button class="btn" onclick="performSearch()">Search</button>

    <div id="logArea" class="box log"></div>
    <div id="loading" class="loading" style="display:none;">Loading…</div>
    <div id="results" class="box results"></div>

    <div id="contacts-section">
      <h2>Saved Contacts</h2>
      <div class="contacts-wrapper">
        <table id="contacts-table">
          <thead>
            <tr><th>Email</th><th>Phone</th><th>Source</th><th>Timestamp</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
let eventSource = null;

function logMessage(msg){
  const area=document.getElementById('logArea');
  area.innerHTML+=`&gt; ${msg}<br>`;
  area.scrollTop=area.scrollHeight;
}

function updateContactsTable(contacts) {
  const tbody = document.querySelector('#contacts-table tbody');
  contacts.forEach(c => {
    tbody.insertAdjacentHTML('beforeend',`
      <tr>
        <td>${c.email||''}</td><td>${c.phone||''}</td>
        <td>${c.source||''}</td><td>${c.timestamp||''}</td>
      </tr>`);
  });
}

function startProgressStream() {
  if (eventSource) {
    eventSource.close();
  }
  
  eventSource = new EventSource('/search-progress');
  
  eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'heartbeat') {
      return; // Ignore heartbeat messages
… (truncated)

```

