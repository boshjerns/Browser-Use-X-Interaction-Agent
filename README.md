<div align="center">

<img src="docs/assets/banner_win98.png" alt="Browser-Use Contact Discovery Agent" width="600"/>

**Real-time, AI-powered contact hunting — now with Windows-95 vibes**  
<br/>

<!-- badges -->
<a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
<img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python">
<a href="https://github.com/boshjerns/BrowserUse--Contact-Discovery-Agent/issues">
  <img alt="Issues" src="https://img.shields.io/github/issues/boshjerns/BrowserUse--Contact-Discovery-Agent?logo=github">
</a>
<a href="https://github.com/boshjerns/BrowserUse--Contact-Discovery-Agent/stargazers">
  <img alt="Stars" src="https://img.shields.io/github/stars/boshjerns/BrowserUse--Contact-Discovery-Agent?style=social">
</a>

<br/><br/>

<img src="docs/assets/demo.gif" alt="30-second demo" width="600"/>

</div>

---

## ✨ Key Features

| ⚡ Feature | 🚀 What It Delivers |
|-----------|--------------------|
| **Live browser automation** | Chrome/Edge driven by [Browser-Use](https://docs.browser-use.com) with smart retries & captcha handling |
| **Instant progress feed**   | WebSocket logs update in real time — no more staring at a spinner |
| **LLM-assisted parsing**    | LangChain + OpenAI extract emails/phones from messy HTML |
| **Deduped contact vault**   | Normalizes & stores contacts in SQLite; swap in Postgres if you like |
| **Secure secrets**          | AES-GCM–encrypted API keys (no plaintext in Git or logs) |
| **Retro Win-95 UI**         | Pixel-perfect buttons, grey panels & a splash of neon |

---

## 📸 Screenshots

<table>
  <tr>
    <td align="center"><img src="docs/screenshots/search.png"  alt="Search UI"  width="260"/></td>
    <td align="center"><img src="docs/screenshots/results.png" alt="Results UI" width="260"/></td>
    <td align="center"><img src="docs/screenshots/db_view.png" alt="DB View"    width="260"/></td>
  </tr>
</table>

<details>
<summary>More screenshots (click to expand)</summary>

<p align="center">
  <img src="docs/screenshots/contact_card.png" width="280" alt="Contact Card"/>
  <img src="docs/screenshots/settings.png"     width="280" alt="Settings"/>
</p>

</details>

---

## 🗺️ How It Works

```mermaid
flowchart TD
    subgraph Frontend (Flask)
        UI[Retro Win-95 UI] --> WS(WebSocket)
    end
    subgraph Backend
        WS --> Worker["Async Task<br/>(Browser-Use + LangChain)"]
        Worker --> DB[(SQLite)]
        Worker -->|contact JSON| UI
    end
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/boshjerns/BrowserUse--Contact-Discovery-Agent.git
cd BrowserUse--Contact-Discovery-Agent
pip install -r requirements.txt
python app.py            # then open http://127.0.0.1:5000
```

> **Heads-up:** first run grabs the Docker Chromium image (≈ 100 MB) — subsequent launches are instant.

---

## 🙌 Contributing

Bug reports and PRs welcome!  
If you build something cool on top of this, let me know — I’ll add a “made-with” showcase.

## 📖 License

[MIT](LICENSE)
