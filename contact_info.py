from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

class ContactInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[HttpUrl] = None
    social_links: List[HttpUrl] = []
    source_url: HttpUrl
    found_at: datetime
    additional_info: Optional[str] = None

class ContactSearchResults(BaseModel):
    contacts: List[ContactInfo]
    search_query: str
    total_found: int
    search_completed_at: datetime 