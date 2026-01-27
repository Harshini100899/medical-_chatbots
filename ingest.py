"""Data ingestion module for PDF and web content."""
import os
import json
import glob
from typing import List, Optional, Callable
import urllib3
from bs4 import BeautifulSoup as Soup

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set User-Agent
os.environ["USER_AGENT"] = "MedicalChatbot/1.0"

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Config
from config import CHROMA_DB_DIR

# Pydantic models
from models import DoctorProfile, load_doctors_from_jsonl

# Import crawler for structured data extraction
from crawler import crawl_all_sources, DOCTORS_JSONL_FILE

# --- CONSTANTS & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Constants
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCTORS_FILE = os.path.join(BASE_DIR, "doctors.jsonl")
EMBEDDING_MODEL = "nomic-embed-text"


# Non-doctor URLs (health info sites only)
HEALTH_INFO_URLS = [
    "https://www.gesundheitsinformation.de/", 
    "https://gesund.bund.de/",
]


def load_vector_store():
    """Get the existing vector store instance."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="medical_knowledge",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    return vector_store


def load_structured_doctors(file_path: str) -> List[Document]:
    """Load doctors.jsonl as validated Pydantic documents."""
    print(f"👉 Looking for doctors file at: {file_path}")
    
    if not os.path.exists(file_path):
        print("⚠️ doctors.jsonl not found. Run crawler first.")
        return []
    
    print(f"Loading structured data with Pydantic validation...")
    
    # Use the new loader function
    doctor_list = load_doctors_from_jsonl(file_path)
    
    docs = []
    for doctor in doctor_list:
        docs.append(Document(
            page_content=doctor.to_searchable_text(),
            metadata=doctor.to_metadata()
        ))
    
    print(f"✅ Loaded {len(docs)} validated doctors (❌ {doctor_list.extraction_errors} failed validation)")
    return docs


def simple_extractor(html: str) -> str:
    """Extract clean text from HTML."""
    try:
        soup = Soup(html, "html.parser")
        # Remove scripts, styles, nav, footer
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.extract()
        return soup.get_text(separator="\n").strip()
    except Exception:
        return html


def crawl_health_info_sites(urls: List[str], max_depth: int = 2) -> List[Document]:
    """Crawl health information websites (NOT doctor directories)."""
    print(f"Crawling {len(urls)} health info sources (max_depth={max_depth})...")
    docs = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    }

    for url in urls:
        print(f"  - Starting crawl of {url}")
        try:
            loader = RecursiveUrlLoader(
                url=url, 
                max_depth=max_depth,
                extractor=simple_extractor,
                headers=headers,
                prevent_outside=True,
                timeout=15,
                check_response_status=True,
                continue_on_failure=True,
            )
            site_docs = loader.load()
            print(f"    ✅ Collected {len(site_docs)} documents from {url}")
            docs.extend(site_docs)
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    # Clean up
    for doc in docs:
        doc.metadata["type"] = "web"
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        doc.page_content = "\n".join(lines)
    
    docs = [d for d in docs if len(d.page_content) > 100]
    
    print(f"Total health info documents: {len(docs)}")
    return docs


def ingest_all(include_web: bool = True, incremental: bool = False, progress_callback: Optional[Callable] = None):
    """Main ingestion function."""
    all_chunks = []
    
    # 1. Load PDFs
    print("Step 1/5: Loading PDFs...")
    print(f"👉 Looking for PDFs in: {DATA_DIR}")
    
    if os.path.exists(DATA_DIR):
        try:
            pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            print(f"📄 Found {len(pdf_docs)} PDF pages.")
            all_chunks.extend(pdf_docs)
        except Exception as e:
            print(f"Warning: Error loading PDFs: {e}")
    else:
        print(f"Creating data directory at {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)

    # 2. CRAWL doctor directory FIRST (this populates doctors.jsonl)
    if include_web:
        print("\nStep 2/5: Crawling Doctor Directory (arzt-auskunft.de)...")
        print("This will extract structured doctor data and save to doctors.jsonl")
        try:
            crawl_all_sources(incremental=incremental, max_pages_per_site=50)
            print("✅ Doctor directory crawl complete!")
        except Exception as e:
            print(f"⚠️ Crawler error: {e}")

    # 3. Load Structured Data (Doctors) from JSONL
    print("\nStep 3/5: Loading Structured Doctor Data...")
    doctor_docs = load_structured_doctors(DOCTORS_FILE)
    all_chunks.extend(doctor_docs)
    
    # 4. Crawl health info websites
    if include_web:
        print("\nStep 4/5: Crawling Health Info Websites...")
        web_docs = crawl_health_info_sites(HEALTH_INFO_URLS, max_depth=2)
        all_chunks.extend(web_docs)
    else:
        print("\nStep 4/5: Skipping web crawling (disabled)")

    # CHECK: Did we actually find anything?
    if not all_chunks:
        raise ValueError(
            "❌ No documents found! \n"
            "1. Put PDFs in the 'data' folder.\n"
            "2. Enable web crawling to fetch doctor data.\n"
            "3. Check internet connection."
        )

    # 5. Text Splitting & Indexing
    print("\nStep 5/5: Splitting and Indexing...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        add_start_index=True
    )
    final_chunks = text_splitter.split_documents(all_chunks)
    print(f"📦 Total chunks to process: {len(final_chunks)}")
    
    if progress_callback:
        progress_callback(0, total_chunks=len(final_chunks))

    print(f"Initializing Vector Store in {CHROMA_DB_DIR}...")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="medical_knowledge",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    batch_size = 50
    total = len(final_chunks)
    
    for i in range(0, total, batch_size):
        batch = final_chunks[i:i+batch_size]
        try:
            vector_store.add_documents(batch)
            print(f"Processed batch {i // batch_size + 1}/{(total // batch_size) + 1}")
        except Exception as e:
             print(f"Error adding batch: {e}")
        
        if progress_callback:
            progress_callback(min(len(batch), total - i))
            
    print("✅ Ingestion successfully completed.")


if __name__ == "__main__":
    ingest_all(include_web=True)
