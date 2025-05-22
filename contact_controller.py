from browser_use import Controller, ActionResult, Browser
from contact_info import ContactInfo, ContactSearchResults
from datetime import datetime
import re
from typing import List, Optional
from urllib.parse import urlparse
import json
from pathlib import Path
from playwright.async_api import Page

class ContactController(Controller):
    def __init__(self, output_file: str = "contacts.json"):
        super().__init__(output_model=ContactSearchResults)
        self.output_file = output_file
        self.contacts: List[ContactInfo] = []
        self.search_query: Optional[str] = None
        
    def save_contacts(self):
        """Save contacts to JSON file"""
        results = ContactSearchResults(
            contacts=self.contacts,
            search_query=self.search_query or "",
            total_found=len(self.contacts),
            search_completed_at=datetime.now()
        )
        
        with open(self.output_file, 'w') as f:
            json.dump(results.model_dump(), f, indent=2, default=str)
        
        return f"Saved {len(self.contacts)} contacts to {self.output_file}"

    @Controller.action("Extract contact information from current page")
    async def extract_contacts(self, browser: Browser) -> str:
        page = await browser.get_current_page()
        content = await page.content()
        
        # Extract emails
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', content)
        
        # Extract phone numbers (basic pattern)
        phones = re.findall(r'\+?[\d\s-()]{10,}', content)
        
        # Extract social media links
        social_patterns = [
            r'https?://(?:www\.)?linkedin\.com/[^\s<>"]+',
            r'https?://(?:www\.)?twitter\.com/[^\s<>"]+',
            r'https?://(?:www\.)?facebook\.com/[^\s<>"]+',
            r'https?://(?:www\.)?instagram\.com/[^\s<>"]+'
        ]
        social_links = []
        for pattern in social_patterns:
            social_links.extend(re.findall(pattern, content))
        
        # Create contact info
        contact = ContactInfo(
            email=emails[0] if emails else None,
            phone=phones[0] if phones else None,
            social_links=social_links,
            source_url=page.url,
            found_at=datetime.now()
        )
        
        if contact.email or contact.phone or contact.social_links:
            self.contacts.append(contact)
            return f"Found contact info: {contact.model_dump_json()}"
        
        return "No contact information found on this page"

    @Controller.action("Search for contact information")
    async def search_contacts(self, query: str, browser: Browser) -> str:
        self.search_query = query
        page = await browser.get_current_page()
        
        # Search for contact-related terms
        search_terms = [
            "contact us",
            "about us",
            "team",
            "staff",
            "people",
            "contact information",
            "get in touch"
        ]
        
        for term in search_terms:
            try:
                # Use a more reliable selector
                elements = await page.query_selector_all(f'text="{term}"')
                if elements:
                    await elements[0].click()
                    await page.wait_for_load_state('networkidle')
                    await self.extract_contacts(browser)
            except Exception as e:
                print(f"Error clicking {term}: {str(e)}")
                continue
        
        return self.save_contacts()

    @Controller.action("Follow contact page links")
    async def follow_contact_links(self, browser: Browser) -> str:
        page = await browser.get_current_page()
        
        # Look for common contact page link patterns
        contact_links = await page.query_selector_all('a[href*="contact"], a[href*="about"], a[href*="team"]')
        
        for link in contact_links:
            try:
                href = await link.get_attribute('href')
                if href:
                    # Use the same page instead of creating new ones
                    await page.goto(href)
                    await page.wait_for_load_state('networkidle')
                    await self.extract_contacts(browser)
            except Exception as e:
                print(f"Error following link: {str(e)}")
                continue
        
        return self.save_contacts() 