from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import json
import re

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    async def run_agent():
        agent = Agent(task=robust_prompt, llm=model)
        logger.info('Initialized Agent')
        result = await agent.run(on_step_end=on_step_end, max_steps=100)
        logger.info('Agent run complete')
        if hasattr(result, 'final_result'):
            final = result.final_result()
            logger.info(f'Final result: {final}')
            return final
        return str(result)

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
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 