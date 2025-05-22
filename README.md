# Browser Use Web Interface

A web interface for Browser Use that allows you to perform searches using a virtual browser.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install
```

3. Run the Flask application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your OpenAI API key in the input field
2. Enter your search query
3. Click the Search button
4. Wait for the results to appear

## Features

- Secure API key handling (never stored)
- Real-time search results
- Modern UI with loading indicators
- Error handling and user feedback

## Requirements

- Python 3.11 or higher
- OpenAI API key
- Internet connection

## Security Note

Your API key is only used for the current session and is never stored. However, please ensure you're using this application in a secure environment. 