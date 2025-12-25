# 🧞 UMLGenie - AI-Powered UML Architect

UMLGenie is an intelligent, AI-driven UML diagram generation and editing tool. It combines the power of Large Language Models (LLMs) via OpenRouter/Gemini with a specialized RAG (Retrieval-Augmented Generation) system to generate accurate Mermaid.js and PlantUML diagrams.

## ✨ Key Features

*   **🗣️ Natural Language Chat**: Describe your system in plain English, and the AI will draft the UML components.
*   **🧙‍♂️ Visual Editor & Magic Fix**: Write code manually with an IDE-like interface. Use the "Magic Fix" button to have the AI auto-correct syntax errors using its knowledge base.
*   **📚 RAG-Powered Precision**: A custom RAG engine indexes Mermaid documentation, PlantUML references, and OMG UML standards to provide context-aware answers and valid syntax.
*   **🖼️ Live Preview**:
    *   **Mermaid**: Local, interactive JavaScript rendering with Zoom, Pan, and Theme Toggling (Light/Dark).
    *   **PlantUML**: Cloud-based rendering (or local server configurable).
*   **🧠 Smart Scanner**: Analyzes existing diagrams (code) to explain them or suggest improvements.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/UMLGenie.git
    cd UMLGenie
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Secrets**:
    *   Copy the example secrets file:
        ```bash
        cp .streamlit/secrets.example.toml .streamlit/secrets.toml
        ```
    *   Edit `.streamlit/secrets.toml` and add your API keys:
        *   `GOOGLE_API_KEY`: Required for embeddings (using `Qwen` or Gemini) and generative AI if selected.
        *   `OPENROUTER_API_KEY`: Required if using OpenRouter models.

## 🚀 Usage

1.  **Initialize the Knowledge Base (RAG)**:
    Before using the app, you need to ingest the documentation documents to build the vector database.
    ```bash
    python ingest_docs.py
    ```
    *This may take a few minutes as it scrapes web docs (Mermaid + PlantUML official docs + PlantUML stdlib/C4 + UML theory) and processes PDFs (PlantUML language reference + OMG UML).*

### Embedding model + device (fixing CUDA out-of-memory)

If you see a `torch.OutOfMemoryError: CUDA out of memory`, your GPU is too small for the embedding model. By default, UMLGenie uses a smaller embedding model and will prefer CPU on small GPUs.

You can control embeddings via environment variables:

- `UMLGENIE_EMBED_MODEL`: embedding model name (default: `Qwen/Qwen3-Embedding-0.6B`)
- `UMLGENIE_EMBED_DEVICE`: `auto` (default) / `cpu` / `cuda`
- `UMLGENIE_EMBED_BATCH_SIZE`: encoding batch size (default: `8`)

On Windows (cmd.exe), to force CPU:

```bat
set UMLGENIE_EMBED_DEVICE=cpu
python ingest_docs.py
```

2.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```
    *Or use the provided `run_app.bat` script on Windows.*

## 📂 Project Structure

*   `app.py`: Main application entry point and sidebar navigation.
*   `features.py`: UI components for Chat, Visual Editor, and Scanner.
*   `ai_service.py`: Logic for communicating with LLM providers (Google/OpenRouter).
*   `rag_engine.py`: Core RAG system (FAISS + SentenceTransformers).
*   `ingest_docs.py`: Pipeline for scraping URLs and parsing PDFs into vector indices.
*   `uml_utils.py`: Helpers for rendering diagrams and handling local/web logic.

## 🤖 Supported Models

*   **Google Gemini** (1.5 Pro/Flash) - Recommended for Vision capabilities.
*   **OpenRouter** - Access to DeepSeek, Claude, GPT-4, etc.

---
*Built with [Streamlit](https://streamlit.io).*
