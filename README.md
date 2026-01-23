# 🏥 Medical Information Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that provides trusted medical information from verified sources using local LLMs.

## Features

- **Multilingual Support**: English and German responses
- **Voice Input**: Speech-to-text for hands-free queries
- **Smart Suggestions**: AI-generated follow-up questions
- **Confidence Scoring**: Shows reliability of answers
- **Source Citations**: References for all information
- **Conversation History**: Save and load chat sessions
- **Incremental Updates**: Fast knowledge base refreshes

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Mistral-Nemo (via Ollama) |
| Embeddings | nomic-embed-text |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Web Interface | Gradio |
| Web Crawling | BeautifulSoup, RecursiveUrlLoader |

## Data Sources

- 📄 Local PDF medical documents
- 🏥 [arzt-auskunft.de](https://www.arzt-auskunft.de) - German doctor directory
- 📚 [gesundheitsinformation.de](https://www.gesundheitsinformation.de) - Health information
- 🏛️ [gesund.bund.de](https://gesund.bund.de) - Federal Ministry of Health

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd medical-_chatbots

# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running with required models
ollama pull mistral-nemo
ollama pull nomic-embed-text
```

## Usage

```bash
# Start the web interface
python app.py
```

Then open http://localhost:7860 in your browser.

### First-Time Setup

1. Go to the **⚙️ Admin** tab
2. Click **🔄 Full Refresh** to build the knowledge base
3. Switch to **💬 Chat** tab and start asking questions

## Project Structure

```
medical-_chatbots/
├── app.py              # Gradio web interface
├── chatbot.py          # RAG chatbot logic
├── ingest.py           # Data ingestion pipeline
├── crawler.py          # Web crawler with structured extraction
├── config.py           # Configuration settings
├── doctors.jsonl       # Structured doctor data
├── data/               # PDF documents
├── chroma_db/          # Vector store
└── utils/
    └── metrics.py      # Performance tracking
```

## Requirements

- Python 3.9+
- Ollama running locally
- ~4GB RAM for embeddings
- Internet connection for web crawling

## License

MIT
