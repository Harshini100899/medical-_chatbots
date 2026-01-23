"""Web crawler for medical websites with structured data extraction."""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import List, Dict, Optional
import re
import json
import hashlib
import os

from config import CRAWL_URLS, BASE_DIR

# Cache file for incremental updates
CRAWL_CACHE_FILE = os.path.join(BASE_DIR, "crawl_cache.json")


def load_crawl_cache() -> Dict:
    """Load the crawl cache from disk."""
    if os.path.exists(CRAWL_CACHE_FILE):
        try:
            with open(CRAWL_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"urls": {}, "content_hashes": {}}


def save_crawl_cache(cache: Dict):
    """Save the crawl cache to disk."""
    with open(CRAWL_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def get_content_hash(content: str) -> str:
    """Generate a hash of content for change detection."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def clean_text(text: str) -> str:
    """Clean extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\!\?\-äöüßÄÖÜ:/]', '', text)
    return text.strip()


# =============================================================================
# STRUCTURED DATA EXTRACTION FOR DOCTOR DIRECTORY
# =============================================================================

def extract_doctor_info(soup: BeautifulSoup, url: str) -> List[Dict]:
    """Extract structured doctor information from arzt-auskunft.de."""
    doctors = []
    
    # Look for doctor listing elements (common patterns for arzt-auskunft)
    # Includes search results and profile page containers
    doctor_elements = soup.find_all(['div', 'article', 'li'], 
        class_=re.compile(r'doctor|arzt|listing|result|entry|card|treffer', re.I))
    
    # Fallback for generic rows or if site structure is flatter (e.g. detail pages)
    if not doctor_elements:
        doctor_elements = soup.find_all('div', class_=re.compile(r'row|item|profil|detail', re.I))
    
    for elem in doctor_elements:
        doctor = {
            "doctor_name": None,
            "specialties": None,
            "practice_address": None,
            "phone": None,
            "avg_rating": None,
            "review_count": 0,
            "profile_url": url
        }
        
        # Extract name
        name_elem = elem.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b', 'a'], 
            class_=re.compile(r'name|title|heading|titel', re.I))
        if not name_elem and elem.name in ['h1', 'h2']: 
            name_elem = elem # Sometimes the element itself is the name
            
        if name_elem:
            doctor['doctor_name'] = clean_text(name_elem.get_text())
            # Try to grab profile URL from link on name
            link = name_elem if name_elem.name == 'a' else name_elem.find('a')
            if link and link.get('href'):
                doctor['profile_url'] = urljoin(url, link.get('href'))
        
        # Extract address
        addr_elem = elem.find(['address', 'p', 'span', 'div'], 
            class_=re.compile(r'address|addr|street|location|anschrift', re.I))
        if addr_elem:
            doctor['practice_address'] = clean_text(addr_elem.get_text())
        else:
            # Fallback: Look for text matching zip code pattern
            for s in elem.strings:
                if re.search(r'\b\d{5}\b', s):
                    doctor['practice_address'] = clean_text(s)
                    break
        
        # Extract phone
        phone_elem = elem.find(['a', 'span', 'p'], href=re.compile(r'tel:', re.I)) or \
                     elem.find(['span', 'p', 'div'], class_=re.compile(r'phone|tel|telefon', re.I))
        if phone_elem:
            phone_text = phone_elem.get('href', '') if phone_elem.name == 'a' else phone_elem.get_text()
            phone_text = re.sub(r'^tel:', '', phone_text)
            doctor['phone'] = clean_text(phone_text)
        
        # Extract specialty
        spec_elem = elem.find(['span', 'p', 'div'], 
            class_=re.compile(r'specialty|fach|specialization|category|rubrik', re.I))
        if spec_elem:
            doctor['specialties'] = clean_text(spec_elem.get_text())
        
        # Only add if we found at least a name
        if doctor.get('doctor_name') and len(doctor.get('doctor_name', '')) > 2:
            doctors.append(doctor)
    
    return doctors


def format_doctor_as_text(doctor: Dict) -> str:
    """Format structured doctor info as searchable text."""
    parts = []
    
    if doctor.get('doctor_name'):
        parts.append(f"Arzt/Doctor: {doctor['doctor_name']}")
    if doctor.get('specialties'):
        parts.append(f"Fachgebiet/Specialty: {doctor['specialties']}")
    if doctor.get('practice_address'):
        parts.append(f"Adresse/Address: {doctor['practice_address']}")
    if doctor.get('phone'):
        parts.append(f"Telefon/Phone: {doctor['phone']}")
    if doctor.get('profile_url'):
        parts.append(f"URL: {doctor['profile_url']}")
    
    return "\n".join(parts)


# =============================================================================
# GENERAL WEB CRAWLING
# =============================================================================

def extract_text_from_page(soup: BeautifulSoup) -> str:
    """Extract meaningful text from a BeautifulSoup object."""
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        element.decompose()
    
    main_content = soup.find('main') or soup.find('article') or \
                   soup.find('div', class_=re.compile(r'content|main|article'))
    
    if main_content:
        text = main_content.get_text(separator=' ')
    else:
        text = soup.get_text(separator=' ')
    
    return clean_text(text)


def is_doctor_directory(url: str) -> bool:
    """Check if URL is from the doctor directory site."""
    return 'arzt-auskunft.de' in url


def crawl_page(url: str, visited: set, cache: Dict, 
               max_pages: int = 100, incremental: bool = False) -> List[Dict]:
    """Crawl a page with support for incremental updates."""
    documents = []
    pages_to_visit = [url]
    base_domain = urlparse(url).netloc
    new_or_changed = 0
    skipped = 0
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'de-DE,de;q=0.9,en;q=0.8',
    }
    
    # Patterns to skip (non-content pages)
    skip_patterns = [
        r'/login', r'/register', r'/signup', r'/cart', r'/checkout',
        r'/account', r'/admin', r'#', r'\?sort=', r'\?order=',
        r'/print/', r'/feed/', r'/rss', r'mailto:', r'javascript:',
        r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.css$', r'\.js$'
    ]
    skip_regex = re.compile('|'.join(skip_patterns), re.I)
    
    while pages_to_visit and len(visited) < max_pages:
        current_url = pages_to_visit.pop(0)
        
        if current_url in visited:
            continue
        
        # Skip non-content URLs
        if skip_regex.search(current_url):
            continue
            
        try:
            response = requests.get(current_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue
            
            content_hash = get_content_hash(response.text)
            
            # Check if content has changed (incremental mode)
            if incremental and current_url in cache.get('content_hashes', {}):
                if cache['content_hashes'][current_url] == content_hash:
                    print(f"⏭️  Skipping (unchanged): {current_url}")
                    visited.add(current_url)
                    skipped += 1
                    continue
            
            print(f"📥 Crawling [{len(visited)+1}/{max_pages}]: {current_url}")
            soup = BeautifulSoup(response.content, 'lxml')
            visited.add(current_url)
            
            # Update cache
            cache['content_hashes'][current_url] = content_hash
            
            # Extract content based on site type
            if is_doctor_directory(current_url):
                # Structured extraction for doctor directory
                doctors = extract_doctor_info(soup, current_url)
                for doctor in doctors:
                    documents.append({
                        'content': format_doctor_as_text(doctor),
                        'source': current_url,
                        'title': f"Dr. {doctor.get('doctor_name', 'Unknown')}",
                        'type': 'structured',
                        'metadata': doctor
                    })
                if doctors:
                    new_or_changed += 1
            else:
                # Unstructured extraction for health info
                text = extract_text_from_page(soup)
                if len(text) > 100:
                    documents.append({
                        'content': text,
                        'source': current_url,
                        'title': soup.title.string if soup.title else current_url,
                        'type': 'unstructured'
                    })
                    new_or_changed += 1
            
            # Find ALL links to follow (improved link discovery)
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Skip anchors and empty hrefs
                if not href or href.startswith('#'):
                    continue
                
                full_url = urljoin(current_url, href)
                parsed = urlparse(full_url)
                
                # Normalize URL (remove fragments)
                normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    # Keep only essential query params
                    normalized_url += f"?{parsed.query}"
                
                # Check if same domain and not visited
                if parsed.netloc == base_domain and normalized_url not in visited:
                    if not skip_regex.search(normalized_url):
                        if normalized_url not in pages_to_visit:
                            pages_to_visit.append(normalized_url)
            
            # Be respectful - rate limiting
            time.sleep(0.3)
            
        except requests.exceptions.Timeout:
            print(f"⏱️  Timeout: {current_url}")
            continue
        except requests.exceptions.HTTPError as e:
            print(f"🚫 HTTP Error {e.response.status_code}: {current_url}")
            continue
        except Exception as e:
            print(f"❌ Error crawling {current_url}: {e}")
            continue
    
    if len(visited) >= max_pages:
        print(f"⚠️  Reached max pages limit ({max_pages}). {len(pages_to_visit)} URLs remaining.")
    
    if incremental:
        print(f"   📊 New/changed: {new_or_changed}, Skipped: {skipped}")
    
    return documents


def crawl_all_sources(incremental: bool = False, max_pages_per_site: int = 100) -> List[Dict]:
    """Crawl all configured URLs with optional incremental mode."""
    all_documents = []
    visited = set()
    
    # Load cache for incremental updates
    cache = load_crawl_cache() if incremental else {"urls": {}, "content_hashes": {}}
    
    print(f"\n{'='*50}")
    print(f"Crawling mode: {'INCREMENTAL' if incremental else 'FULL'}")
    print(f"Max pages per site: {max_pages_per_site}")
    print(f"{'='*50}\n")
    
    for url in CRAWL_URLS:
        print(f"\n--- Starting crawl of {url} ---")
        site_visited = set()
        docs = crawl_page(url, site_visited, cache, max_pages=max_pages_per_site, incremental=incremental)
        all_documents.extend(docs)
        visited.update(site_visited)
        print(f"✅ Collected {len(docs)} documents from {url} ({len(site_visited)} pages visited)")
    
    # Save updated cache
    save_crawl_cache(cache)
    
    print(f"\n{'='*50}")
    print(f"Total pages visited: {len(visited)}")
    print(f"Total documents collected: {len(all_documents)}")
    print(f"{'='*50}")
    
    return all_documents


if __name__ == "__main__":
    # Test with incremental=False for first run, higher page limit
    documents = crawl_all_sources(incremental=False, max_pages_per_site=100)
    for doc in documents[:5]:
        print(f"\n{'='*40}")
        print(f"Title: {doc['title']}")
        print(f"Source: {doc['source']}")
        print(f"Type: {doc.get('type', 'unknown')}")
        print(f"Content preview: {doc['content'][:300]}...")
