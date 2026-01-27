"""Configuration for the Medical Chatbot."""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data sources
PDF_DIR = BASE_DIR
PDF_FILES = ["Medical_book.pdf"]

CRAWL_URLS = [
    "https://www.gesundheitsinformation.de/",
    "https://gesund.bund.de/",
    "https://www.arzt-auskunft.de/oberhausen-rheinland/",
]

# Vector store
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "medical_docs"

# Ollama models
LLM_MODEL = "mistral:latest"  # Ollama model name for Mistral 7B
EMBEDDING_MODEL = "nomic-embed-text"

# Mistral model parameters (optimized for medical chatbot)
LLM_TEMPERATURE = 0.3
LLM_TOP_P = 0.9
LLM_TOP_K = 40
LLM_NUM_CTX = 8192  # Larger context window for 12B model
LLM_REPEAT_PENALTY = 1.1  # Reduce repetition

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# RAG parameters
TOP_K_RESULTS = 4

# Crawling settings
MAX_PAGES_PER_SITE = 100
CRAWL_DELAY = 0.3
