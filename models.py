"""Pydantic models for structured data validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import re
import json
import os


class DoctorProfile(BaseModel):
    """Validated doctor profile from arzt-auskunft.de."""
    
    doctor_name: str = Field(..., min_length=2, description="Full name of the doctor")
    specialties: Optional[str] = Field(default=None, description="Medical specialties")
    practice_address: Optional[str] = Field(default=None, description="Practice address")
    phone: Optional[str] = Field(default=None, description="Contact phone number")
    avg_rating: Optional[float] = Field(default=None, ge=0, le=5, description="Average rating 0-5")
    review_count: int = Field(default=0, ge=0, description="Number of reviews")
    profile_url: str = Field(..., description="URL to doctor's profile page")
    
    @field_validator('doctor_name')
    @classmethod
    def clean_doctor_name(cls, v: str) -> str:
        """Clean and validate doctor name."""
        # Remove extra whitespace
        v = ' '.join(v.split())
        # Remove common prefixes that might be duplicated
        v = re.sub(r'^(Dr\.\s*)+', 'Dr. ', v)
        # Must contain at least one letter
        if not re.search(r'[a-zA-ZäöüßÄÖÜ]', v):
            raise ValueError('Doctor name must contain letters')
        return v
    
    @field_validator('phone')
    @classmethod
    def clean_phone(cls, v: Optional[str]) -> Optional[str]:
        """Clean phone number."""
        if not v or v.lower() in ['n/a', 'na', '-', '']:
            return None
        # Remove common prefixes
        v = re.sub(r'^tel:', '', v, flags=re.I)
        # Keep only digits, spaces, +, -, /
        v = re.sub(r'[^\d\s\+\-\/\(\)]', '', v).strip()
        return v if v else None
    
    @field_validator('practice_address')
    @classmethod
    def clean_address(cls, v: Optional[str]) -> Optional[str]:
        """Clean and validate address."""
        if not v or v.lower() in ['n/a', 'na', '-', '']:
            return None
        # Remove extra whitespace
        v = ' '.join(v.split())
        return v if len(v) > 2 else None
    
    @field_validator('specialties')
    @classmethod
    def clean_specialties(cls, v: Optional[str]) -> Optional[str]:
        """Clean specialties field."""
        if not v or v.lower() in ['n/a', 'na', '-', '']:
            return None
        return ' '.join(v.split()) if len(v) > 1 else None
    
    @field_validator('profile_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL is valid."""
        if not v.startswith('http') and not v.startswith('line_'):
            raise ValueError('Profile URL must start with http')
        return v
    
    def to_jsonl_dict(self) -> dict:
        """Convert to dict for JSONL storage."""
        return {
            "doctor_name": self.doctor_name,
            "specialties": self.specialties or "N/A",
            "practice_address": self.practice_address or "N/A",
            "phone": self.phone or "N/A",
            "avg_rating": self.avg_rating,
            "review_count": self.review_count,
            "profile_url": self.profile_url
        }
    
    def to_searchable_text(self) -> str:
        """Convert to searchable text format for vector store."""
        parts = ["DOCTOR PROFILE:", f"Name: {self.doctor_name}"]
        if self.specialties:
            parts.append(f"Fachgebiet/Specialty: {self.specialties}")
        if self.practice_address:
            parts.append(f"Adresse/Address: {self.practice_address}")
        if self.phone:
            parts.append(f"Telefon/Phone: {self.phone}")
        if self.avg_rating is not None:
            parts.append(f"Bewertung/Rating: {self.avg_rating}/5 ({self.review_count} reviews)")
        parts.append(f"URL: {self.profile_url}")
        return "\n".join(parts)
    
    def to_metadata(self) -> dict:
        """Convert to metadata dict for document storage."""
        return {
            "source": "doctors_directory",
            "type": "structured",
            "doctor_name": self.doctor_name,
            "specialties": self.specialties or "N/A",
            "practice_address": self.practice_address or "N/A",
            "phone": self.phone or "N/A",
            "profile_url": self.profile_url
        }


class DoctorList(BaseModel):
    """Container for multiple validated doctor profiles."""
    
    doctors: List[DoctorProfile] = Field(default_factory=list)
    source_url: str = Field(..., description="URL where doctors were extracted from")
    extraction_errors: int = Field(default=0, description="Count of failed extractions")
    
    def add_doctor(self, **kwargs) -> bool:
        """Try to add a doctor, returns True if valid, False otherwise."""
        try:
            doctor = DoctorProfile(**kwargs)
            self.doctors.append(doctor)
            return True
        except Exception as e:
            print(f"⚠️ Validation failed: {e}")
            self.extraction_errors += 1
            return False
    
    def save_to_jsonl(self, file_path: str, append: bool = True):
        """Save all validated doctors to JSONL file."""
        mode = 'a' if append else 'w'
        
        # Load existing URLs to avoid duplicates
        existing_urls = set()
        if append and os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                existing_urls.add(item.get('profile_url', ''))
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass
        
        new_count = 0
        with open(file_path, mode, encoding='utf-8') as f:
            for doctor in self.doctors:
                # Skip duplicates
                if doctor.profile_url in existing_urls:
                    continue
                
                json_line = json.dumps(doctor.to_jsonl_dict(), ensure_ascii=False)
                f.write(json_line + '\n')
                existing_urls.add(doctor.profile_url)
                new_count += 1
        
        print(f"💾 Saved {new_count} new doctors to {file_path} (skipped {len(self.doctors) - new_count} duplicates)")
        return new_count
    
    def __len__(self) -> int:
        return len(self.doctors)
    
    def __iter__(self):
        return iter(self.doctors)


def load_doctors_from_jsonl(file_path: str) -> DoctorList:
    """Load and validate doctors from JSONL file."""
    doctor_list = DoctorList(source_url=f"file://{file_path}")
    
    if not os.path.exists(file_path):
        return doctor_list
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                doctor_list.add_doctor(**item)
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️ Error loading doctor: {e}")
    
    return doctor_list
