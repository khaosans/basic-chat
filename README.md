# BasicChat: Your Intelligent Local AI Assistant

<div align="center">

![BasicChat Logo](assets/brand/logo/elron-logo-full.png)

**🔒 Privacy-First • 🧠 Advanced Reasoning • 🔬 Deep Research • ⚡ High Performance**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLMs-green.svg)](https://ollama.ai)
[![Redis](https://img.shields.io/badge/Redis-Task%20Queue-orange.svg)](https://redis.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An intelligent, private AI assistant that runs entirely on your local machine*

</div>

---

## 🎥 Demo

<div align="center">

![BasicChat Demo](assets/demo_seq_0.6s.gif)

*Real-time reasoning and document analysis with local AI models*

</div>

---

## 🚀 Quick Start

> **🟢 TL;DR:**
> 1. **Install [Ollama](https://ollama.ai)** and [Python 3.11+](https://python.org)
> 2. **Clone this repo**: `git clone ... && cd basic-chat`
> 3. **Create venv**: `python -m venv venv && source venv/bin/activate`
> 4. **Install deps**: `pip install -r requirements.txt`
> 5. **Pull models**: `ollama pull mistral` and `ollama pull nomic-embed-text`
> 6. **Start app**: `./start_basicchat.sh` (recommended) or `./start_dev.sh`
> 7. **Visit**: [http://localhost:8501](http://localhost:8501) (or your chosen port)

### Prerequisites
- **Ollama** (local LLM server)
- **Python 3.11+**
- **Redis** (optional, for background tasks)
- **Git**

### Install Required Models
```bash
ollama pull mistral              # Reasoning model
ollama pull nomic-embed-text     # Embedding model for RAG
ollama pull llava                # (Optional) Vision model for images
```

### Clone & Setup
```bash
git clone https://github.com/khaosans/basic-chat-template.git
cd basic-chat-template
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start the Application
```bash
# Start Ollama (if not running)
ollama serve &

# Start app with all services (recommended)
./start_basicchat.sh
# or for dev mode
./start_dev.sh
```

**App URLs:**
- Main App: [http://localhost:8501](http://localhost:8501)
- Task Monitor (Flower): [http://localhost:5555](http://localhost:5555)
- Redis: `localhost:6379`
- Ollama: `localhost:11434`

---

## 🏆 Best Practices & Pro Tips

<div style="background:#e3f2fd; padding:1em; border-radius:8px; border-left:5px solid #1976d2;">

- **🟢 Always use local code for E2E and dev** (not Docker)
- **🟢 Run health checks before E2E**: `poetry run python scripts/e2e_health_check.py`
- **🟢 Use `.env.local` for secrets and config** (never commit keys)
- **🟢 Use streaming for best UX**
- **🟢 Monitor logs and Flower for background tasks**
- **🟢 Stop all Docker containers before local dev/E2E**
- **🟢 Ensure backwards compatibility for stored data**
- **🟢 Store all important keys in `.env.local`**
- **🟢 Use `bunx` for TypeScript, `poetry` for Python**
- **🟢 Prefer `0.0.0.0` over `localhost` for server hosts**
- **🟢 Check `progress.md` for always-up-to-date tips**

</div>

---

## 🛠️ Development & Testing

- **Run all tests**: `pytest -n auto`
- **E2E tests**: `bunx playwright test --reporter=list`
- **Coverage**: `pytest --cov=app --cov-report=html`
- **Pre-commit hooks**: `pre-commit run --all-files`
- **Type checking**: `mypy . --strict`
- **CI/CD**: Always starts from source, not Docker

---

## 🤖 LLM Judge Quality Assurance

BasicChat includes an intelligent LLM Judge that evaluates code quality, test coverage, documentation, architecture, security, and performance.

### Quick Start
```bash
# Automatic setup
./scripts/setup_local_llm_judge.sh

# Quick evaluation (smart backend selection)
make llm-judge-quick

# Full evaluation (smart backend selection)
make llm-judge
```

### Features
- **Smart Backend Selection**: Automatically chooses Ollama (local) or OpenAI (remote/CI)
- **Comprehensive Evaluation**: 6 categories with weighted scoring
- **Actionable Reports**: Prioritized improvement plans
- **Multiple Backends**: Ollama (local) and OpenAI (cloud) with automatic fallback
- **CI/CD Integration**: Automated quality gates with OpenAI for remote environments
- **Deterministic Results**: Consistent evaluation standards

### Generated Reports
- `llm_judge_action_items.md` - Prioritized action plan
- `llm_judge_improvement_tips.md` - Specific improvement tips
- `llm_judge_results.json` - Detailed evaluation data

📖 **Full Documentation**: [Local LLM Judge Setup](docs/LOCAL_LLM_JUDGE.md)

## 🧩 Troubleshooting

- **Redis not running?**: `brew services start redis` or `sudo systemctl start redis`
- **Ollama not running?**: `ollama serve`
- **Port in use?**: `lsof -i :8501` then `kill -9 <PID>`
- **Permission issues?**: `chmod +x *.sh`
- **Check logs**: `tail -f basicchat.log`
- **Health check**: `poetry run python scripts/e2e_health_check.py`
- **LLM Judge issues?**: See [Local LLM Judge Setup](docs/LOCAL_LLM_JUDGE.md#troubleshooting)

---

## 🏗️ Architecture Overview

```ascii
+-------------------+      +-------------------+      +-------------------+
|   User Interface  | ---> |  Reasoning Engine | ---> |   Ollama/Tools    |
+-------------------+      +-------------------+      +-------------------+
        |                        |                            |
        v                        v                            v
+-------------------+      +-------------------+      +-------------------+
| Document Uploads  | ---> | Document Processor| ---> |  ChromaDB/Vector  |
+-------------------+      +-------------------+      +-------------------+
```

<details>
<summary>Mermaid: System Architecture</summary>

```mermaid
graph TB
    subgraph "🎨 User Interface"
        UI[Web Interface]
        AUDIO[Audio Processing]
    end
    subgraph "🧠 Core Logic"
        RE[Reasoning Engine]
        DP[Document Processor]
        TR[Tool Registry]
    end
    subgraph "⚡ Services"
        AO[Ollama Client]
        VS[Vector Store]
        CS[Cache Service]
        WS[Web Search]
    end
    subgraph "🗄️ Storage"
        CHROMA[Vector Database]
        CACHE[Memory Cache]
        FILES[File Storage]
    end
    subgraph "🌐 External"
        OLLAMA[LLM Server]
        DDG[Search Engine]
    end
    UI --> RE
    UI --> DP
    AUDIO --> RE
    RE --> AO
    RE --> VS
    RE --> TR
    DP --> VS
    TR --> WS
    AO --> OLLAMA
    VS --> CHROMA
    CS --> CACHE
    WS --> DDG
    CHROMA --> FILES
    CACHE --> FILES
```
</details>

---

## 📚 Documentation & Further Reading

- [Startup Guide](STARTUP_GUIDE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Features Overview](docs/FEATURES.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Technical Overview](docs/TECHNICAL_OVERVIEW.md)
- [Planning & Roadmap](docs/ROADMAP.md)
- [Evaluators & LLM Judge](docs/EVALUATORS.md)
- [Local LLM Judge Setup](docs/LOCAL_LLM_JUDGE.md)
- [progress.md](progress.md) — always up-to-date best practices

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with ❤️ using modern Python, async/await, and best practices for production-ready AI applications.
</div>
