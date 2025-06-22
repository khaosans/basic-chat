# Release Notes - v1.1.0

This release focuses on improving the stability of the build, enhancing the developer experience, and completely overhauling the project documentation for clarity and ease of use.

## ✨ Key Highlights

-   **Build Stability**: Fixed critical dependency issues that were causing the GitHub Actions build to fail. The `requirements.txt` file has been updated to include all necessary packages (`langchain-chroma`, `langchain-community`), ensuring a reliable and reproducible environment.

-   **Comprehensive Documentation Overhaul**: All documentation has been rewritten from the ground up to be more concise, relevant, and user-friendly. This includes:
    -   A streamlined `README.md` with a clear "Getting Started" guide.
    -   A practical `DEVELOPMENT.md` focused on contributor workflows.
    -   A simplified `ARCHITECTURE.md` with a clear system diagram.
    -   A scannable `FEATURES.md` and a high-level `ROADMAP.md`.
    -   Redundant documents and academic jargon have been removed to reduce confusion.

-   **Robust Database Management**:
    -   A new utility script, `scripts/cleanup_chroma.py`, has been added to help developers manage and clean up local ChromaDB databases.
    -   The `.gitignore` file has been updated to correctly ignore all ChromaDB directories.

## Fixes

-   Resolved `ModuleNotFoundError` for `langchain_chroma` and `langchain_community` by adding them to `requirements.txt`.
-   Corrected text splitter initialization in `document_processor.py`.

## What's Next?

With a stable build and clean documentation, the project is now in a great position for future feature development. See the updated [Project Roadmap](ROADMAP.md) for our plans.

## 📚 References

### **Research Papers**
Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *arXiv preprint arXiv:2201.11903*, 2022.

Lewis, Mike, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems*, vol. 33, 2020, pp. 9459-9474.

Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-Scale Similarity Search with GPUs." *IEEE Transactions on Big Data*, vol. 7, no. 3, 2019, pp. 535-547.

### **Core Technologies**
Ollama. "Local Large Language Model Server." *Ollama.ai*, 2024, https://ollama.ai.

Streamlit. "Web Application Framework." *Streamlit.io*, 2024, https://streamlit.io.

LangChain. "LLM Application Framework." *LangChain.com*, 2024, https://langchain.com.

ChromaDB. "Vector Database for AI Applications." *ChromaDB.ai*, 2024, https://chromadb.ai.

### **Development Resources**
Beazley, David M., and Brian K. Jones. *Python Cookbook*. 3rd ed., O'Reilly Media, 2013.

Martin, Robert C. *Clean Architecture: A Craftsman's Guide to Software Structure and Design*. Prentice Hall, 2017.

---

[← Back to README](../README.md) | [Architecture →](ARCHITECTURE.md) | [Features →](FEATURES.md) | [Development →](DEVELOPMENT.md) 