# Features Overview

This document provides an overview of the key features available in BasicChat.

[← Back to README](../README.md)

## 🧠 AI & Reasoning

-   **Multiple Reasoning Modes**: Choose from different reasoning strategies to best solve your problem:
    -   **Standard Mode**: For direct answers to simple questions.
    -   **Chain-of-Thought**: See a step-by-step thought process, which is great for understanding complex logic.
    -   **Multi-Step Agent**: The AI automatically breaks down complex questions and uses its tools to find the answer.
-   **Local & Private**: All reasoning is performed locally on your machine using Ollama. No data is sent to external cloud providers.
-   **Model Selection**: Easily switch between different Ollama models through the UI.

## 📄 Document & Image Processing (RAG)

-   **Multi-Format Support**: Upload and analyze various file types:
    -   PDFs (`.pdf`)
    -   Text files (`.txt`)
    -   Markdown files (`.md`)
    -   Images (`.png`, `.jpg`) for text extraction (OCR).
-   **Chat with Your Data**: Ask questions about your uploaded documents and get context-aware answers.
-   **Vector Search**: Uses a local ChromaDB vector store to perform fast, semantic searches over your documents.

## 🛠️ Built-in Tools

-   **Smart Calculator**:
    -   Handles complex mathematical expressions.
    -   Includes advanced functions like trigonometry, logarithms, and factorials.
    -   Provides safe evaluation of expressions.
-   **Web Search**:
    -   Integrated with DuckDuckGo to get real-time information from the web.
    -   Results are cached to speed up repeated queries.
-   **Time Tools**:
    -   Full timezone support for time conversions.
    -   Calculate time differences between different locations.

## 🚀 Performance & User Experience

-   **Fast & Responsive**: Built with an asynchronous architecture to handle multiple operations without blocking.
-   **Intelligent Caching**: A multi-layer cache (in-memory and Redis) significantly speeds up responses for repeated queries.
-   **Streaming Responses**: See the AI's response generate in real-time.
-   **Modern UI**: A clean and intuitive interface built with Streamlit.
-   **Database Cleanup**: Includes a utility script to manage and clean up old ChromaDB databases. 