# AI Assistant with DeFi Insights

An AI-powered chatbot that combines general knowledge with real-time crypto and DeFi data.

## Features

- Real-time conversations with AI
- Internet search capabilities
- Cryptocurrency price checking
- DeFi protocol TVL tracking
- Document generation
- Streaming responses

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd robotv4
poetry install
```

3. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key
```

## Running the Chatbot

```bash
poetry run streamlit run src/app.py
```

## Example Usage

Try these example queries:
- "What's the current price of Bitcoin?"
- "Show me the TVL of Aave protocol"
- "Search for the latest news about Ethereum"
- "Create a document summarizing DeFi trends"

## Configuration

Required API keys:
- OpenAI API key
- SerpAPI key (for web searches)

The application uses environment variables for configuration. Create a `.env` file in the project root with the required API keys.
# basic-chat
