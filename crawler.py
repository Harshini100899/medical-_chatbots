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

from config import BASE_DIR
from models import DoctorProfile, DoctorList

# Cache file for incremental updates
CRAWL_CACHE_FILE = os.path.join(BASE_DIR, "crawl_cache.json")
# JSONL file for storing validated doctor data
DOCTORS_JSONL_FILE = os.path.join(BASE_DIR, "doctors.jsonl")

# Only crawl doctor directory
DOCTOR_CRAWL_URLS = [
    "https://www.arzt-auskunft.de/oberhausen-rheinland/",
]

# Blacklist patterns - these are NOT doctor names
BLACKLIST_PATTERNS = [
    r'^Sind Sie',
    r'^Finden Sie',
    r'^Die Arzt-Auskunft',
    r'^Ein Service',
    r'^Werden Sie',
    r'^Meinungen zu',
    r'^Steigern Sie',
    r'^Ihre Kunden',
    r'^Benötigen Sie',
    r'^Patienten-Services',
    r'^Termine finden',
    r'^Barrierefreie',
    r'^Fragen',
    r'^Fachgebiet',
    r'^Filter$',
    r'^Kostenfreie',
    r'^Ist Ihre Praxis',
    r'^Reichweiten',
    r'^Zeigen Sie',
    r'^Patientenservice',
    r'^Patientenzufriedenheit',
    r'^Eine starke',
    r'^Unsere Qualität',
    r'^Wissenschaftlich',
    r'^Faire Kommunikation',
    r'^Sie haben einen',
    r'^Angaben gem',
    r'^Einen Arzt',
    r'^Oder Sie nutzen',
    r'^Ihre Vorteile',
    r'^Machen Sie sich',
    r'^Treffen Sie',
    r'^\d+\s*Millionen',
    r'^Heilberufler',
    r'^Kliniken',
    r'^Apotheken',
]


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


def is_valid_doctor_name(name: str) -> bool:
    """Check if the text looks like a real doctor name."""
    if not name or len(name) < 5 or len(name) > 80:
        return False
    
    # Must contain a title OR be in format "Vorname Nachname"
    has_title = bool(re.search(r'\b(Dr\.|Prof\.|Herr|Frau|med\.|dent\.)\b', name, re.I))
    has_two_words = len(name.split()) >= 2
    
    if not (has_title or has_two_words):
        return False
    
    # Check against blacklist
    for pattern in BLACKLIST_PATTERNS:
        if re.search(pattern, name, re.I):
            return False
    
    # Must contain at least one capital letter (likely a name)
    if not re.search(r'[A-ZÄÖÜ]', name):
        return False
    
    # Should not be too long (likely a sentence, not a name)
    if len(name.split()) > 6:
        return False
    
    # Should not contain certain words that indicate it's not a name
    bad_words = ['finden', 'suchen', 'service', 'kontakt', 'impressum', 'datenschutz', 
                 'millionen', 'zahlen', 'auskunft', 'stiftung', 'gesundheit', 'praxis',
                 'termin', 'online', 'patient', 'klinik', 'apotheke', 'hilfe']
    name_lower = name.lower()
    for word in bad_words:
        if word in name_lower:
            return False
    
    return True


def extract_name_from_url(url: str) -> str:
    """Extract doctor name from arzt-auskunft.de profile URL.
    
    Example URL: https://www.arzt-auskunft.de/arzt/innere-medizin/oberhausen/mansour-ahmad-6796781
    Returns: Mansour Ahmad
    """
    try:
        # Extract the last part of the URL path (before the ID)
        path = urlparse(url).path.rstrip('/')
        parts = path.split('/')
        
        if len(parts) >= 2:
            # Last part contains name-id, e.g., "mansour-ahmad-6796781"
            name_with_id = parts[-1]
            
            # Remove the trailing numeric ID
            # Split by '-' and remove last part if it's all digits
            name_parts = name_with_id.split('-')
            
            # Remove numeric ID at the end
            while name_parts and name_parts[-1].isdigit():
                name_parts.pop()
            
            if name_parts:
                # Convert to title case and join
                name = ' '.join(part.capitalize() for part in name_parts)
                
                # Fix common German name prefixes
                name = name.replace(' Von ', ' von ')
                name = name.replace(' Van ', ' van ')
                name = name.replace(' De ', ' de ')
                
                return name
    except Exception:
        pass
    
    return None


def extract_specialty_from_url(url: str) -> str:
    """Extract specialty from arzt-auskunft.de profile URL.
    
    Example URL: https://www.arzt-auskunft.de/arzt/innere-medizin/oberhausen/name-123
    Returns: Innere Medizin
    """
    try:
        path = urlparse(url).path.rstrip('/')
        parts = path.split('/')
        
        # URL structure: /arzt/specialty/city/name-id
        if len(parts) >= 3 and parts[1] == 'arzt':
            specialty_slug = parts[2]
            # Convert slug to readable name
            specialty = specialty_slug.replace('-', ' ').title()
            
            # Fix German umlauts common in specialties
            replacements = {
                'Ue': 'ü', 'Ae': 'ä', 'Oe': 'ö',
                'ue': 'ü', 'ae': 'ä', 'oe': 'ö',
            }
            for old, new in replacements.items():
                specialty = specialty.replace(old, new)
            
            return specialty
    except Exception:
        pass
    
    return None


def extract_city_from_url(url: str) -> str:
    """Extract city from arzt-auskunft.de profile URL."""
    try:
        path = urlparse(url).path.rstrip('/')
        parts = path.split('/')
        
        # URL structure: /arzt/specialty/city/name-id
        if len(parts) >= 4 and parts[1] == 'arzt':
            city_slug = parts[3]
            # If the city_slug contains name-id pattern, use parts[3] as city
            # But actually city is in parts[3], name is in parts[4]
            # Let me re-check: /arzt/innere-medizin/oberhausen/mansour-ahmad-6796781
            # parts = ['', 'arzt', 'innere-medizin', 'oberhausen', 'mansour-ahmad-6796781']
            city_slug = parts[3]
            
            # Check if this looks like a city (not a name-id)
            if not city_slug[-1].isdigit():
                city = city_slug.replace('-', ' ').title()
                return city
    except Exception:
        pass
    
    return None


# =============================================================================
# STRUCTURED DATA EXTRACTION FOR DOCTOR DIRECTORY (WITH PYDANTIC)
# =============================================================================

def extract_doctor_info(soup: BeautifulSoup, url: str) -> DoctorList:
    """Extract structured doctor information from arzt-auskunft.de with Pydantic validation."""
    doctor_list = DoctorList(source_url=url)
    
    # Find all links to doctor profile pages
    # Pattern: /arzt/specialty/city/name-id (where id is numeric)
    profile_links = soup.find_all('a', href=re.compile(r'/arzt/[^/]+/[^/]+/[^/]+-\d+$', re.I))
    
    # Track unique URLs to avoid duplicates
    seen_urls = set()
    
    print(f"    Found {len(profile_links)} profile links")
    
    for link in profile_links:
        href = link.get('href', '')
        full_url = urljoin(url, href)
        
        # Skip duplicates
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        
        # Extract data from URL (more reliable than page text)
        doctor_name = extract_name_from_url(full_url)
        specialty = extract_specialty_from_url(full_url)
        city = extract_city_from_url(full_url)
        
        if not doctor_name or len(doctor_name) < 3:
            continue
        
        # Try to get address from parent element
        address = city
        phone = None
        
        parent = link.find_parent(['div', 'li', 'article', 'tr'])
        if parent:
            # Look for postal code in parent
            for text in parent.stripped_strings:
                if re.search(r'\b\d{5}\b', text) and text != doctor_name:
                    address = clean_text(text)
                    break
            
            # Look for phone
            phone_link = parent.find('a', href=re.compile(r'tel:', re.I))
            if phone_link:
                phone = phone_link.get('href', '').replace('tel:', '')
        
        raw_data = {
            "doctor_name": doctor_name,
            "specialties": specialty,
            "practice_address": address,
            "phone": phone,
            "avg_rating": None,
            "review_count": 0,
            "profile_url": full_url
        }
        
        success = doctor_list.add_doctor(**raw_data)
        if success:
            print(f"      ✓ {doctor_name} ({specialty})")
    
    return doctor_list


def format_doctor_as_text(doctor: DoctorProfile) -> str:
    """Format validated doctor info as searchable text."""
    return doctor.to_searchable_text()


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
               max_pages: int = 100, incremental: bool = False,
               save_doctors: bool = True) -> List[Dict]:
    """Crawl a page with support for incremental updates."""
    documents = []
    pages_to_visit = [url]
    base_domain = urlparse(url).netloc
    new_or_changed = 0
    skipped = 0
    validation_errors = 0
    
    # Collect all doctors for batch saving
    all_doctors = DoctorList(source_url=url)
    
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
        r'\.pdf$', r'\.jpg$', r'\.png$', r'\.gif$', r'\.css$', r'\.js$',
        r'/impressum', r'/datenschutz', r'/nutzungsbedingungen', r'/kontakt',
        r'/fuer-aerzte', r'/premium', r'/tipps-und-faq', r'\?form=',
    ]
    skip_regex = re.compile('|'.join(skip_patterns), re.I)
    
    while pages_to_visit and len(visited) < max_pages:
        current_url = pages_to_visit.pop(0)
        
        if current_url in visited:
            continue
        
        if skip_regex.search(current_url):
            continue
            
        try:
            response = requests.get(current_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                continue
            
            content_hash = get_content_hash(response.text)
            
            if incremental and current_url in cache.get('content_hashes', {}):
                if cache['content_hashes'][current_url] == content_hash:
                    print(f"⏭️  Skipping (unchanged): {current_url}")
                    visited.add(current_url)
                    skipped += 1
                    continue
            
            print(f"📥 Crawling [{len(visited)+1}/{max_pages}]: {current_url}")
            soup = BeautifulSoup(response.content, 'lxml')
            visited.add(current_url)
            
            cache['content_hashes'][current_url] = content_hash
            
            # Extract doctors with strict validation
            if is_doctor_directory(current_url):
                doctor_list = extract_doctor_info(soup, current_url)
                validation_errors += doctor_list.extraction_errors
                
                for doctor in doctor_list:
                    all_doctors.doctors.append(doctor)
                    documents.append({
                        'content': doctor.to_searchable_text(),
                        'source': current_url,
                        'title': doctor.doctor_name,
                        'type': 'structured',
                        'metadata': doctor.to_metadata()
                    })
                
                if len(doctor_list) > 0:
                    print(f"    ✅ Found {len(doctor_list)} doctors")
                    for doc in doctor_list.doctors:
                        print(f"      ✓ {doc.doctor_name} ({doc.specialties or 'N/A'})")
                    new_or_changed += 1
            
            # Find links to follow (focus on doctor listing pages)
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                if not href or href.startswith('#'):
                    continue
                
                full_url = urljoin(current_url, href)
                parsed = urlparse(full_url)
                
                # Only follow links that look like doctor listings
                if parsed.netloc == base_domain:
                    path = parsed.path
                    # Prioritize specialty/city pages
                    if re.search(r'/[a-z-]+/[a-z-]+/?$', path) or re.search(r'/arzt/', path):
                        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if normalized_url not in visited and not skip_regex.search(normalized_url):
                            if normalized_url not in pages_to_visit:
                                pages_to_visit.append(normalized_url)
            
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
    
    # Save all collected doctors to JSONL file
    if save_doctors and len(all_doctors) > 0:
        all_doctors.save_to_jsonl(DOCTORS_JSONL_FILE, append=True)
    
    if len(visited) >= max_pages:
        print(f"⚠️  Reached max pages limit ({max_pages}). {len(pages_to_visit)} URLs remaining.")
    
    print(f"   📊 Doctors found: {len(all_doctors)}, Pages: {new_or_changed}, Skipped: {skipped}")
    
    return documents


def crawl_all_sources(incremental: bool = False, max_pages_per_site: int = 100) -> List[Dict]:
    """Crawl doctor directory URLs."""
    all_documents = []
    visited = set()
    
    cache = load_crawl_cache() if incremental else {"urls": {}, "content_hashes": {}}
    
    print(f"\n{'='*50}")
    print(f"🏥 DOCTOR DIRECTORY CRAWLER")
    print(f"Mode: {'INCREMENTAL' if incremental else 'FULL'}")
    print(f"Max pages per site: {max_pages_per_site}")
    print(f"Output: {DOCTORS_JSONL_FILE}")
    print(f"{'='*50}\n")
    
    for url in DOCTOR_CRAWL_URLS:
        print(f"\n--- Crawling {url} ---")
        site_visited = set()
        docs = crawl_page(url, site_visited, cache, max_pages=max_pages_per_site, 
                         incremental=incremental, save_doctors=True)
        all_documents.extend(docs)
        visited.update(site_visited)
        print(f"✅ Collected {len(docs)} doctor records from {url}")
    
    save_crawl_cache(cache)
    
    # Count doctors in JSONL
    doctor_count = 0
    if os.path.exists(DOCTORS_JSONL_FILE):
        with open(DOCTORS_JSONL_FILE, 'r', encoding='utf-8') as f:
            doctor_count = sum(1 for line in f if line.strip())
    
    print(f"\n{'='*50}")
    print(f"📊 CRAWL SUMMARY")
    print(f"Pages visited: {len(visited)}")
    print(f"Documents collected: {len(all_documents)}")
    print(f"Total doctors in JSONL: {doctor_count}")
    print(f"{'='*50}")
    
    return all_documents


if __name__ == "__main__":
    # Clear existing doctors.jsonl for fresh crawl
    if os.path.exists(DOCTORS_JSONL_FILE):
        os.remove(DOCTORS_JSONL_FILE)
        print(f"🗑️ Cleared existing {DOCTORS_JSONL_FILE}")
    
    documents = crawl_all_sources(incremental=False, max_pages_per_site=50)
