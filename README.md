<div align="center">

# 🤖 Browser-Use-X-Interaction-Agent
**AI-Powered Automation for Your X.com (Twitter) Feed**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
[![Issues](https://img.shields.io/github/issues/boshjerns/Browser-Use-X-Interaction-Agent?logo=github)](https://github.com/boshjerns/Browser-Use-X-Interaction-Agent/issues)
[![Stars](https://img.shields.io/github/stars/boshjerns/Browser-Use-X-Interaction-Agent?style=social)](https://github.com/boshjerns/Browser-Use-X-Interaction-Agent/stargazers)

</div>

An AI-powered Flask application that automates your X.com (formerly Twitter) account. Define a task, and the agent will browse your feed, observe your interaction style from your profile, and then engage with relevant, organic content by liking, bookmarking, reposting, or commenting on your behalf. Features real-time progress updates and intelligent ad avoidance.

---

## ✨ Key Features

| ⚡ Feature | 🚀 What It Delivers |
|-----------|--------------------|
| **Intelligent X.com Interaction** | Automates browsing, liking, bookmarking, reposting, and commenting on X.com. |
| **User Style Emulation** | Observes the logged-in user's profile (Tweets, Replies, Likes) to mimic their typical interaction style. |
| **Task-Driven Operation** | Executes actions based on a user-defined task (e.g., "Find and like posts about AI in sports"). |
| **Advanced Ad Detection** | Actively identifies and ignores advertisements in the feed, focusing only on organic content. |
| **Real-Time Web UI** | Provides live feedback and progress updates directly in your browser via Server-Sent Events. |
| **`gpt-4o` Powered Decisions** | Leverages OpenAI's `gpt-4o` for nuanced understanding and context-aware interactions. |
| **Robust Browser Control** | Built on the `browser-use` library for effective and observable browser automation (runs non-headless). |
| **Secure API Key Handling** | Manages your OpenAI API key securely with encryption, allowing input via the UI or an environment variable. |

---

## 🚀 Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/boshjerns/Browser-Use-X-Interaction-Agent.git
    cd Browser-Use-X-Interaction-Agent
    ```

2.  **Install Dependencies:**
    ```bash
    # It's recommended to use a virtual environment
    # python -m venv venv
    # source venv/bin/activate  (or .\venv\Scripts\activate on Windows)
    pip install -r requirements.txt
    ```

3.  **Configure Google Chrome Path:**
    Open `app.py` and update the `browser_binary_path` in the `BrowserConfig` to your local Google Chrome executable path.
    ```python
    # Example in app.py:
    browser_config = BrowserConfig(
        browser_binary_path='C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', # <-- UPDATE THIS PATH
        headless=False
    )
    ```

4.  **Provide Your OpenAI API Key:**
    *   **Option 1 (Recommended - UI):** The application will prompt you to enter your API key in the web interface. It will be stored securely in an encrypted file (`.encrypted_api_key`).
    *   **Option 2 (Environment Variable):**
        ```bash
        # On Linux/macOS
        export OPENAI_API_KEY="sk-..."
        # On Windows PowerShell
        $env:OPENAI_API_KEY="sk-..."
        ```

5.  **Run the Application:**
    ```bash
    python app.py
    ```
    Then open `http://127.0.0.1:5000` in your web browser.

---

## 🔑 Environment Variables

| Variable         | Purpose                                           | Default | How to Set                  |
|------------------|---------------------------------------------------|---------|-----------------------------|
| `OPENAI_API_KEY` | Your OpenAI API key for LLM-driven decision making. | *none*  | Via UI (recommended) or Env Var |

---

## 🏆 Credits & Acknowledgements

This project leverages and extends the excellent **[Browser-Use](https://github.com/browser-use/browser-use)** library by Gregor Žunič & Magnus Müller.

<p align="center">
  <img alt="Browser-Use logo" src="docs/assets/browser-use.png" width="220"> 
  <!-- Ensure this image path is correct or remove if not present -->
</p>

Huge thanks to the Browser-Use community for their foundational work in AI-driven browser automation. Explore their [Discord](https://link.browser-use.com/discord) to see more innovative projects!

---

## 🙌 Contributing

Bug reports, feature requests, and pull requests are welcome! Please open an issue to discuss any significant changes.

---

## 📖 License

Released under the [MIT License](LICENSE).
