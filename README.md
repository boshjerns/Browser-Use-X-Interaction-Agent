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
