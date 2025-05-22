# Browser-Use Contact Discovery Agent

A web application that uses browser automation to discover contact information from various sources. The application provides a user-friendly interface to search for and collect contact details like email addresses and phone numbers.

## Features

- Real-time contact discovery using browser automation
- Secure API key storage
- Live progress updates during search
- Contact deduplication
- Persistent storage of discovered contacts
- Modern Windows 95-inspired UI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/boshjerns/BrowserUse--Contact-Discovery-Agent.git
cd BrowserUse--Contact-Discovery-Agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your OpenAI API key in the input field
2. Click "Save API Key" to store it securely (optional)
3. Enter your search query
4. Click "Search" to start the contact discovery process
5. View real-time updates in the log area
6. Discovered contacts will appear in the table below

## Security

- API keys are encrypted before storage
- Sensitive files are excluded from version control
- No API keys or sensitive data are logged

## Requirements

- Python 3.8+
- Flask
- LangChain
- Browser-Use
- Cryptography

## License

MIT License

---

Repository: [https://github.com/boshjerns/BrowserUse--Contact-Discovery-Agent](https://github.com/boshjerns/BrowserUse--Contact-Discovery-Agent)
