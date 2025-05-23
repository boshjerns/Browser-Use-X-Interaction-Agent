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

@app.route('/contacts', methods=['DELETE'])
def delete_all_contacts():
    try:
        if os.path.exists("contacts.json"):
            os.remove("contacts.json")
        return jsonify({'status': 'success', 'message': 'All contacts deleted.'})
    except Exception as e:
        logger.error(f'Error deleting contacts: {e}')
        return jsonify({'error': 'Failed to delete contacts'}), 500

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
        "Continue searching and updating the list until no more new contacts are found or all reasonable sources are exhausted. "
        "If you encounter a page requiring human verification (e.g., a captcha or login you cannot bypass), do not use the 'ask_human' action. Instead, navigate back and try a different search result or an alternative strategy to find the information."
    )

    async def run_agent():
        # Create a handler for processing steps
        # This handler receives the agent instance after each step execution
        async def step_handler(agent_instance):
            """Callback executed after each agent step to extract contacts and push SSE updates."""
            # Ensure the agent has history to inspect
            if not hasattr(agent_instance, 'state') or not agent_instance.state.history:
                return

            history = agent_instance.state.history
            # Get the most recent extracted content and URL from the history
            extracted_content_list = history.extracted_content() # Note: This gets all extracted content so far
            current_url_source = history.urls()[-1] if history.urls() else "Unknown Source"

            contacts_for_this_step = [] # Stores all unique contacts found in this specific step processing

            # Process the most recent extracted content block
            if extracted_content_list:
                latest_content_block = extracted_content_list[-1]

                # Set to keep track of (email, phone) tuples from JSON in this step to avoid re-adding by regex
                json_parsed_identifiers_this_step = set()

                # 1. Try to parse JSON array from the latest content block
                try:
                    json_start = latest_content_block.find('[')
                    json_end = latest_content_block.rfind(']')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        json_str = latest_content_block[json_start : json_end + 1]
                        parsed_json_contacts = json.loads(json_str)
                        if isinstance(parsed_json_contacts, list):
                            for c in parsed_json_contacts:
                                if isinstance(c, dict):
                                    if 'source' not in c or not c['source']:
                                        c['source'] = current_url_source
                                    # Ensure essential fields
                                    c.setdefault('role', 'Unknown')
                                    c.setdefault('name', '')
                                    c_email = c.setdefault('email', '')
                                    c_phone = c.setdefault('phone', '')

                                    contacts_for_this_step.append(c)
                                    if c_email:
                                        json_parsed_identifiers_this_step.add((c_email.lower(), 'email'))
                                    if c_phone:
                                        json_parsed_identifiers_this_step.add((c_phone, 'phone'))
                            logger.info(f"Step_handler: Parsed {len(parsed_json_contacts)} contacts from JSON block.")
                except json.JSONDecodeError as e:
                    logger.warning(f'Step_handler: JSONDecodeError parsing content block from agent history: {e}. Will proceed with regex.')
                except Exception as e:
                    logger.error(f'Step_handler: Unexpected error parsing JSON from agent history: {e}. Proceeding with regex.')

                # 2. Scan the same latest_content_block for emails/phones using regex
                emails_from_regex, phones_from_regex = extract_emails_and_phones(latest_content_block)

                for email_r in emails_from_regex:
                    if (email_r.lower(), 'email') not in json_parsed_identifiers_this_step:
                        contacts_for_this_step.append({
                            'role': 'Unknown',
                            'name': '',
                            'email': email_r,
                            'phone': '',
                            'source': current_url_source
                        })
                        logger.info(f"Step_handler: Added new email from regex: {email_r}")

                for phone_r in phones_from_regex:
                    if (phone_r, 'phone') not in json_parsed_identifiers_this_step:
                        contacts_for_this_step.append({
                            'role': 'Unknown',
                            'name': '',
                            'email': '',
                            'phone': phone_r,
                            'source': current_url_source
                        })
                        logger.info(f"Step_handler: Added new phone from regex: {phone_r}")

            if contacts_for_this_step:
                # save_contacts_stepwise handles global deduplication before saving to contacts.json
                # Use agent_instance.task as the target
                save_contacts_stepwise(agent_instance.task if hasattr(agent_instance, 'task') else "Unknown Target", contacts_for_this_step)
                # Send only the unique contacts found *in this step* to the frontend for logging
                search_progress.put({
                    'type': 'contacts_found',
                    'contacts': contacts_for_this_step,
                    'source': current_url_source
                })
                logger.info(
                    f"Step_handler: Pushed {len(contacts_for_this_step)} unique contacts for this step to frontend/save."
                )

        # Instantiate the Agent
        agent = Agent(
            task=robust_prompt,
            llm=model,
            use_vision=True,
        )
        logger.info('Initialized Agent')
        # Send initial progress update
        search_progress.put({
            'type': 'status',
            'message': 'Starting search...'
        })

        # Run the agent
        # Pass the step_handler to the run method
        logger.info("Attempting to run agent with on_step_end callback.")
        result = await agent.run(max_steps=100, on_step_end=step_handler)
        logger.info('Agent run complete')

        http_response_payload = str(result) # Default payload

        if hasattr(result, 'final_result'):
            final = result.final_result()
            logger.info(f'Final result: {final}')

            final_contacts_found = []
            final_source_for_contacts = "Unknown"

            if final:
                # Determine the source URL for these final contacts
                source_match_in_final_text = re.search(r"Source:\s*(https?://[^\s,]+)", final, re.IGNORECASE)
                if source_match_in_final_text:
                    final_source_for_contacts = source_match_in_final_text.group(1)
                elif hasattr(agent, 'state') and agent.state and hasattr(agent.state, 'history') and agent.state.history and agent.state.history.urls():
                    final_source_for_contacts = agent.state.history.urls()[-1]
                else:
                    final_source_for_contacts = "Final result summary"

                # Attempt to parse structured JSON from the final string first
                parsed_contacts_from_final = False
                try:
                    # The agent's output might be a string like "Collected contacts: [{"role": ...}]"
                    # We need to extract the actual JSON part.
                    json_start_index = final.find('[')
                    json_end_index = final.rfind(']')
                    if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
                        json_str_from_final = final[json_start_index : json_end_index + 1]
                        contacts_from_json = json.loads(json_str_from_final)
                        if isinstance(contacts_from_json, list):
                            for c in contacts_from_json:
                                if isinstance(c, dict):
                                    if 'source' not in c or not c['source']:
                                        c['source'] = final_source_for_contacts
                                    # Ensure essential fields exist, even if empty, to prevent downstream errors
                                    c.setdefault('role', 'Unknown')
                                    c.setdefault('name', '')
                                    c.setdefault('email', '')
                                    c.setdefault('phone', '')
                                    final_contacts_found.append(c)
                            parsed_contacts_from_final = True
                            logger.info(f"Successfully parsed {len(contacts_from_json)} contacts from final result JSON string.")
                        else:
                            logger.warning("Parsed JSON from final result was not a list.")
                    else:
                        logger.info("No JSON array found in final result string for direct parsing.")
                except json.JSONDecodeError as e:
                    logger.warning(f'JSONDecodeError when parsing final result: {e}. Will fall back to regex.')
                except Exception as e:
                    logger.error(f'Unexpected error parsing final result JSON: {e}. Will fall back to regex.')

                # If JSON parsing failed or did not happen, fall back to regex extraction for emails/phones
                if not parsed_contacts_from_final:
                    logger.info("Falling back to regex extraction for final result.")
                    emails, phones = extract_emails_and_phones(final)
                    for email in emails:
                        final_contacts_found.append({
                            'role': 'Unknown',
                            'name': '',
                            'email': email,
                            'phone': '',
                            'source': final_source_for_contacts
                        })
                    for phone in phones:
                        final_contacts_found.append({
                            'role': 'Unknown',
                            'name': '',
                            'email': '',
                            'phone': phone,
                            'source': final_source_for_contacts
                        })
            
            if final_contacts_found:
                logger.info(f"Extracted contacts from final result: {final_contacts_found}")
                # Ensure agent.task is appropriate for save_contacts_stepwise, it's the original query
                save_contacts_stepwise(agent.task if hasattr(agent, 'task') else "Unknown Target", final_contacts_found)
                search_progress.put({
                    'type': 'contacts_found',
                    'contacts': final_contacts_found,
                    'source': final_source_for_contacts 
                })
            
            http_response_payload = final # Set the HTTP response to the final agent output
        
        # Send "Search complete!" message AFTER all final processing and contact sending
        search_progress.put({
            'type': 'status',
            'message': 'Search complete!'
        })
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
        result = loop.run_until_complete(run_agent())
        logger.info('Returning result to client')
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f'Error during search: {e}')
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
    app.run(debug=True) 