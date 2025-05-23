from flask import Flask, render_template, request, jsonify, Response
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
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

# Global queue to store search progress updates (will be used for X.com updates)
search_progress = queue.Queue()

# Initialize encryption key (kept for API key)
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
    interaction_options = data.get('interactions', {})
    logger.info(f'API key provided: {bool(api_key)}, Query: {query}')

    if not api_key or not query:
        logger.warning('Missing API key or query')
        return jsonify({'error': 'API key and query are required'}), 400

    # API key is now handled by the /api-key endpoint and dotenv
    os.environ['OPENAI_API_KEY'] = api_key # Keep this line to set the key for the agent
    logger.info('Set OpenAI API key from request')

    model = ChatOpenAI(model="gpt-4o") # Using gpt-4o as recommended for browser-use
    logger.info('Initialized ChatOpenAI model')

    # Rework the prompt for X.com interaction
    x_com_task_prompt = (
        f"You are an AI agent interacting on X.com (formerly Twitter) as a user. Your goal is to " \
        f"perform the following task: {query}. This might involve browsing the feed, liking posts, " \
        f"bookmarking, reposting, or commenting based on the user's typical interaction style. " \
        f"After navigating to X.com, find and visit the profile of the logged-in user **SOLEY to observe their typical interaction style by examining their Tweets, Replies, Likes, and Media sections.** Do NOT interact with any content while on the user's own profile page. " \
        f"Before interacting with any tweet, perform a thorough check to ensure it is NOT an advertisement. " \
        f"Advertisements can be indicated in many ways on X: a visible label text such as 'Ad', 'Promoted', 'Sponsored', or 'Promoted by', an <span> element with classes similar to css-1jxf684 or generic r-qvutc0 containing those words, an aria-label attribute containing 'Ad', or a sub-label near the account name that reads 'Ad' or 'Promoted'. They may also include a small diagonal-arrow icon and lack normal engagement counts. " \
        f"ALWAYS locate the nearest enclosing tweet/article container element before deciding. Within that container search (case-insensitive) for the words 'ad', 'promoted', or 'sponsored' in either visible text content, aria-labels, alt text or title attributes. If ANY such marker is found, treat the tweet as an advertisement and IGNORE IT COMPLETELY. Do not click, like, bookmark, repost, comment, or even expand it. Scroll past it. " \
        f"If your vision tools cannot conclusively determine that the tweet is *not* an ad, default to **NO INTERACTION**. Interact only with tweets you have positively confirmed to be organic (non-advertisement) content. " \
        f"Based on the task and allowed interactions ({interaction_options}), **after analyzing the profile, navigate to the main feed to browse.** " \
        f"Scroll through the feed sufficiently to find diverse posts from **other users** that are relevant to the initial task and align with the user's observed interaction style and interests. For each post from another user, carefully evaluate its content, relevance to the task, and alignment with the user's observed interaction style and interests BEFORE deciding to like or bookmark. " \
        f"While browsing the main timeline, **actively and repeatedly scroll down** (for example by issuing `scroll` or `page-down` style actions) so that you continuously load fresh tweets well beyond the first screenful. " \
        f"After each scroll, pause just long enough for new content to appear, then interpret the newly visible tweets one-by-one before deciding whether any interaction is appropriate. Continue this scroll-observe-interpret loop until you have assessed a meaningful variety of posts (at least several full viewport heights or roughly 40-50 distinct tweets) or until you have confidently fulfilled the user's task. " \
        f"Specifically, when considering liking or bookmarking, identify and understand the content and author of the specific tweet you are about to interact with. This context is crucial for making appropriate decisions. " \
        f"If commenting is allowed ({interaction_options.get('comments', False)}), perform a deeper analysis of the tweet from another user and its context. Use your intelligence (the underlying language model) to craft a relevant, thoughtful, and contextually appropriate comment that aligns with the user's style. Avoid generic comments. " \
        f"For relevant posts from **other users** that align with the user's preferences, perform allowed interactions: " \
        f"- If liking is allowed ({interaction_options.get('likes', False)}), and the post is suitable, click the like button. " \
        f"- If bookmarking is allowed ({interaction_options.get('bookmarks', False)}), and the post is suitable, click the bookmark button. " \
        f"- If reposting is allowed ({interaction_options.get('reposts', False)}), click the repost button. " \
        f"- If commenting is allowed ({interaction_options.get('comments', False)}), find the comment field and type a relevant comment. " \
        f"Report progress and any significant findings (like posts interacted with, including their content or author if possible) through the progress stream."\
        f"Avoid asking human for help or getting stuck on captchas/logins. Prioritize browsing and interacting."
    )

    async def run_agent():
        # Create a handler for processing steps (reworked for X.com)
        async def step_handler(agent_instance):
            """Callback executed after each agent step to process X.com activity and push SSE updates."""
            # Ensure the agent has state history to inspect
            if not hasattr(agent_instance, 'state') or not agent_instance.state.history:
                return

            history = agent_instance.state.history
            current_url = history.urls()[-1] if history.urls() else "Unknown Source"
            
            # Log agent thoughts and actions
            latest_thought = history.model_thoughts()[-1] if history.model_thoughts() else "No thoughts"
            latest_action = history.model_actions()[-1] if history.model_actions() else "No action"
            latest_output = history.model_outputs()[-1] if history.model_outputs() else "No output"

            logger.info(f"Step_handler: URL: {current_url}, Thought: {latest_thought}, Action: {latest_action}, Output: {latest_output}")

            # Push updates to frontend based on agent's action and state
            status_message = f'Browsing: {current_url}'
            # Add the agent's next goal to the status message
            latest_next_goal = "No defined goal"
            if hasattr(agent_instance, 'state') and hasattr(agent_instance.state, 'current_state') and hasattr(agent_instance.state.current_state, 'next_goal'):
                 latest_next_goal = agent_instance.state.current_state.next_goal

            if latest_next_goal != "No defined goal":
                 status_message += f' | Next Goal: {latest_next_goal}'

            search_progress.put({
                'type': 'status',
                'message': status_message
            })
            
            # Add logic here to identify specific interaction actions and report them
            if latest_action and 'name' in latest_action:
                action_name = latest_action['name']
                # You would need more sophisticated logic here to know *what* was liked/bookmarked etc.
                # This is a basic example just reporting the action name.
                if action_name in ['click', 'type']:
                    # Attempt to get some context about the interaction from the output or history
                    interaction_context = latest_output if latest_output != "No output" else ""
                    message = f'Agent performed action: {action_name}'
                    if interaction_context:
                         message += f' (Details: {interaction_context[:100]}...)' # Limit context length
                    search_progress.put({
                        'type': 'status',
                        'message': message
                    })

        # Configure Browser to connect to a real Chrome instance
        # !! IMPORTANT !!: Replace the path below with the actual path to your Chrome executable.
        # Common paths:
        # Windows: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # macOS: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
        # Linux: '/usr/bin/google-chrome'
        browser_config = BrowserConfig(
            browser_binary_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', # <<-- REPLACE WITH YOUR CHROME PATH
            headless=False # Keep headless=False to see the browser window
        )
        browser = Browser(config=browser_config)
        logger.info('Initialized Browser for real Chrome instance')

        # Instantiate the Agent
        agent = Agent(
            task=x_com_task_prompt,
            llm=model,
            use_vision=True, # Vision is essential for browsing X.com
            browser=browser, # Pass the configured browser instance
            # Start by opening x.com
            initial_actions = [
                {'open_tab': {'url': 'https://x.com'}},
            ]
        )
        logger.info('Initialized Agent for X.com task')

        # Send initial progress update
        search_progress.put({
            'type': 'status',
            'message': 'Starting X.com interaction agent...'
        })

        result = None
        agent_stopped_prematurely = False # Flag to indicate if max_steps was hit before agent was truly done
        try:
            logger.info(f"Attempting to run agent with on_step_end callback for max 40 steps.")
            result = await agent.run(max_steps=40, on_step_end=step_handler) # Set max_steps to 40
            logger.info('Agent run complete')

            http_response_payload = str(result) # Default payload

            if hasattr(result, 'final_result'):
                final = result.final_result()
                logger.info(f'Final result: {final}')
                http_response_payload = final # Set the HTTP response to the final agent output
                # Agent reached its own conclusion of 'done'

            # Check if the agent explicitly finished or hit max steps
            if hasattr(result, 'state') and hasattr(result.state, 'is_done') and result.state.is_done:
                 # Agent successfully completed its task
                 search_progress.put({
                     'type': 'status',
                     'message': 'Task complete!'
                 })
            else:
                 # Agent stopped, likely due to max_steps or an issue, but not explicitly 'done'
                 agent_stopped_prematurely = True
                 # Get the very last reported next goal from the history for the paused message
                 last_goal = result.state.next_goals()[-1] if hasattr(result, 'state') and result.state.next_goals() else "Unknown"
                 search_progress.put({
                     'type': 'status',
                     'message': f'Agent paused. Current goal: {last_goal}. Click Continue on frontend.'
                 })

        except Exception as e:
            logger.error(f'Error during agent run: {e}')
            # Send error progress update
            search_progress.put({
                'type': 'error',
                'message': str(e)
            })
            http_response_payload = str(e)
        finally:
            # Ensure the browser is closed after the run
            logger.info('Closing browser...')
            await browser.close()
            logger.info('Browser closed.')

        return http_response_payload

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Running the async agent in a separate thread to not block the Flask server
        # This requires a bit more sophisticated handling than a simple run_until_complete
        # For simplicity in this example, we'll stick to run_until_complete for now, 
        # but be aware this can block the server for the duration of the agent run.
        # A proper production setup would use a separate thread or process pool.
        logger.info('Running agent asynchronously...')
        result = loop.run_until_complete(run_agent())
        logger.info('Async agent run finished.')

        logger.info('Returning result to client')
        return jsonify({'result': result})

    except Exception as e:
        logger.error(f'Error setting up or running asyncio loop: {e}')
        # Send error progress update
        search_progress.put({
            'type': 'error',
            'message': str(e)
        })
        return jsonify({'error': str(e)}), 500

@app.route('/api-key', methods=['POST'])
def save_api_key():
    data = request.get_json()
    api_key = data.get('api_key')
    
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
        
    try:
        # Encrypt and save the API key
        encrypted_key = encrypt_api_key(api_key)
        with open('.encrypted_api_key', 'w') as f:
            f.write(encrypted_key)
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f'Error saving API key: {e}')
        return jsonify({'error': 'Failed to save API key'}), 500

@app.route('/api-key', methods=['GET'])
def get_api_key():
    try:
        if os.path.exists('.encrypted_api_key'):
            with open('.encrypted_api_key', 'r') as f:
                encrypted_key = f.read().strip()
            api_key = decrypt_api_key(encrypted_key)
            if api_key:
                return jsonify({'api_key': api_key})
    except Exception as e:
        logger.error(f'Error retrieving API key: {e}')
    return jsonify({'api_key': None})

if __name__ == '__main__':
    # Make sure to load dotenv when running the app directly
    load_dotenv()
    app.run(debug=True) 