"""Data ingestion module for PDF and web content."""
import os
import json
import glob
from typing import List, Optional, Callable
import urllib3
from bs4 import BeautifulSoup as Soup # Needed for extraction

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

# --- CONSTANTS & PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Constants
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCTORS_FILE = os.path.join(BASE_DIR, "doctors.jsonl")
EMBEDDING_MODEL = "nomic-embed-text"


MEDICAL_URLS = [
    "https://www.gesundheitsinformation.de/", 
    "https://gesund.bund.de/",              
    "https://www.arzt-auskunft.de/oberhausen-rheinland/",                
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
    """Load doctors.jsonl as formatted documents."""
    docs = []
    print(f"👉 Looking for doctors file at: {file_path}")
    
    if not os.path.exists(file_path):
        if os.path.exists("doctors.jsonl"):
            file_path = "doctors.jsonl"
        else:
            print("⚠️ doctors.jsonl missing. Skipping.")
            return []
        
    print(f"Loading structured data...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    text = (
                        f"DOCTOR PROFILE:\n"
                        f"Name: {item.get('doctor_name', 'N/A')}\n"
                        f"Specialties: {item.get('specialties', 'N/A')}\n"
                        f"Address: {item.get('practice_address', 'N/A')}\n"
                        f"Phone: {item.get('phone', 'N/A')}\n"
                        f"URL: {item.get('profile_url', '')}"
                    )
                    metadata = {
                        "source": "doctors_directory",
                        "id": item.get("profile_url", item.get("doctor_name")),
                        "type": "structured"
                    }
                    docs.append(Document(page_content=text, metadata=metadata))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"❌ Error reading doctors file: {e}")
        
    print(f"✅ Created {len(docs)} documents from doctor profiles.")
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


def crawl_medical_sites(urls: List[str], max_depth: int = 3) -> List[Document]:
    """Crawl medical websites recursively with configurable depth."""
    print(f"Crawling {len(urls)} web sources (Recursive max_depth={max_depth})...")
    docs = []
    
    # Headers to mimic a real browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    }

    for url in urls:
        print(f"  - Starting crawl of {url} (depth={max_depth})")
        try:
            # RecursiveUrlLoader with increased depth
            loader = RecursiveUrlLoader(
                url=url, 
                max_depth=max_depth,  # Increased for deeper crawling
                extractor=simple_extractor,
                headers=headers,
                prevent_outside=True,  # Stay on domain
                timeout=15,
                check_response_status=True,
                continue_on_failure=True,  # Don't stop on single page errors
            )
            site_docs = loader.load()
            print(f"    ✅ Collected {len(site_docs)} documents from {url}")
            docs.extend(site_docs)
            
        except Exception as e:
            print(f"    ❌ Failed recursive crawl for {url}: {e}")
            # Fallback for single page if recursive fails
            try:
                print(f"    ↪️ Trying fallback single-page load...")
                web_loader = WebBaseLoader(url, header_template=headers, verify_ssl=False)
                fallback_docs = web_loader.load()
                docs.extend(fallback_docs)
                print(f"    ✅ Fallback collected {len(fallback_docs)} documents")
            except Exception as fallback_e:
                print(f"    ❌ Fallback also failed: {fallback_e}")
    
    # Clean up results
    for doc in docs:
        doc.metadata["type"] = "web"
        doc.metadata["crawled_at"] = "now"
        # Reduce whitespace and filter empty lines
        lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
        doc.page_content = "\n".join(lines)
    
    # Remove documents with very little content
    docs = [d for d in docs if len(d.page_content) > 100]
    
    print(f"Total valid documents collected: {len(docs)}")
    return docs


def ingest_all(include_web: bool = True, incremental: bool = False, progress_callback: Optional[Callable] = None):
    """Main ingestion function."""
    all_chunks = []
    
    # 1. Load PDFs
    print("Step 1/4: Loading PDFs...")
    print(f"👉 Looking for PDFs in: {DATA_DIR}")
    
    if os.path.exists(DATA_DIR):
        try:
            pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            print(f"📄 Found {len(pdf_docs)} PDF pages.")
            all_chunks.extend(pdf_docs)
        except Exception as e:
            print(f"Warning: Error loading PDFs (is the folder empty?): {e}")
    else:
        print(f"Creating data directory at {DATA_DIR}")
        os.makedirs(DATA_DIR, exist_ok=True)

    # 2. Load Structured Data (Doctors)
    print("Step 2/4: Loading Structured Data...")
    doctor_docs = load_structured_doctors(DOCTORS_FILE)
    all_chunks.extend(doctor_docs)
    
    # 3. Web Crawling
    if include_web:
        print("Step 3/4: Crawling Websites...")
        web_docs = crawl_medical_sites(MEDICAL_URLS, max_depth=3)  # Increased depth
        all_chunks.extend(web_docs)

    # CHECK: Did we actually find anything?
    if not all_chunks:
        raise ValueError(
            "❌ No documents found! \n"
            "1. Put PDFs in the 'data' folder.\n"
            "2. Ensure 'doctors.jsonl' is in the project root.\n"
            "3. Check internet connection for web crawling."
        )

    # 4. Text Splitting
    print("Step 4/4: Splitting and Indexing...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        add_start_index=True
    )
    final_chunks = text_splitter.split_documents(all_chunks)
    print(f"📦 Total chunks to process: {len(final_chunks)}")
    
    if progress_callback:
        progress_callback(0, total_chunks=len(final_chunks))

    # 5. Embedding & Indexing
    print(f"Initializing Vector Store in {CHROMA_DB_DIR}...")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="medical_knowledge",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    # Batch process
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
