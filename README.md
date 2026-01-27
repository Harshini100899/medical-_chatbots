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
- **Pydantic Validation**: Structured data extraction with type safety

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Mistral 7B (via Ollama) |
| Embeddings | nomic-embed-text |
| Vector Store | ChromaDB |
| Framework | LangChain |
| Web Interface | Gradio |
| Web Crawling | BeautifulSoup, RecursiveUrlLoader |
| Data Validation | Pydantic v2 |

## Data Sources

- 📄 Local PDF medical documents
- 🏥 [arzt-auskunft.de](https://www.arzt-auskunft.de) - German doctor directory (structured extraction)
- 📚 [gesundheitsinformation.de](https://www.gesundheitsinformation.de) - Health information
- 🏛️ [gesund.bund.de](https://gesund.bund.de) - Federal Ministry of Health

## Structured Data Schema

Doctor profiles are validated using Pydantic with the following schema:

```python
class DoctorProfile(BaseModel):
    doctor_name: str          # Required, min 2 chars
    specialties: str | None   # e.g., "Allgemeinmedizin"
    practice_address: str | None
    phone: str | None         # Auto-cleaned
    avg_rating: float | None  # 0-5 scale
    review_count: int         # Default 0
    profile_url: str          # Required, must be valid URL
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd medical-_chatbots

# Install dependencies
pip install -r requirements.txt

# Pull required Ollama models
ollama pull mistral:latest
ollama pull nomic-embed-text
```

## Usage

```bash
# Step 1: Crawl doctor data first
python crawler.py

# Step 2: Run the app
python app.py
```

Then open http://localhost:7861 in your browser.

### First-Time Setup

1. Run `python crawler.py` to populate `doctors.jsonl`
2. Open the app and go to **⚙️ Admin** tab
3. Click **🔄 Full Refresh** to build the knowledge base
4. Switch to **💬 Chat** tab and start asking questions

## Project Structure

```
medical-_chatbots/
├── app.py              # Gradio web interface
├── chatbot.py          # RAG chatbot logic
├── ingest.py           # Data ingestion pipeline
├── crawler.py          # Web crawler with structured extraction
├── models.py           # Pydantic models for validation
├── config.py           # Configuration settings
├── doctors.jsonl       # Structured doctor data (auto-generated)
├── data/               # PDF documents
├── chroma_db/          # Vector store
└── utils/
    └── metrics.py      # Performance tracking
```

## Requirements

- Python 3.10+
- Ollama running locally with `mistral` and `nomic-embed-text`
- ~8GB RAM for Mistral 7B
- Internet connection for web crawling

## License

MIT
