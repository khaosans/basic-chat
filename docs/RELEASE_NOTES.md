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

---

[← Back to README](../README.md) | [Architecture →](ARCHITECTURE.md) | [Features →](FEATURES.md) | [Development →](DEVELOPMENT.md) 
