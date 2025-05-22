from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from contact_controller import ContactController
from dotenv import load_dotenv
import asyncio
import argparse
from playwright.async_api import BrowserContext
import streamlit as st

load_dotenv()

async def find_contacts(search_query: str, output_file: str = "contacts.json", max_steps: int = 20, headless: bool = False):
    # Initialize the model
    llm = ChatOpenAI(model="gpt-4o")
    
    # Create custom controller
    controller = ContactController(output_file=output_file)
    
    # Configure browser settings
    browser_config = BrowserConfig(
        headless=headless,
        new_context_config=BrowserContextConfig(
            window_width=1280,
            window_height=800,
            wait_for_network_idle_page_load_time=2.0
        )
    )
    
    # Create a single browser instance
    browser = Browser(config=browser_config)
    
    try:
        # Create a single browser context
        async with await browser.new_context() as context:
            # Create the agent with the shared context
            agent = Agent(
                task=f"Search for contact information about '{search_query}'. Look for emails, phone numbers, and social media links. Visit multiple pages and save all found contact information.",
                llm=llm,
                controller=controller,
                browser_context=context  # Use the shared context
            )
            
            # Run the agent with progress tracking
            history = await agent.run(max_steps=max_steps)
            
            # Print results
            if history.final_result():
                if 'st' in globals():
                    st.success(f"Contact search completed. Results saved to {output_file}")
                else:
                    print(f"\nContact search completed. Results saved to {output_file}")
            else:
                if 'st' in globals():
                    st.warning("No contact information found.")
                else:
                    print("\nNo contact information found.")
                    
            return history
    finally:
        # Ensure browser is closed
        await browser.close()

def main():
    parser = argparse.ArgumentParser(description="Find contact information for a given search query")
    parser.add_argument("query", help="The search query to find contact information for")
    parser.add_argument("--output", "-o", default="contacts.json", help="Output file for contact information")
    parser.add_argument("--max-steps", "-m", type=int, default=20, help="Maximum number of search steps")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    
    args = parser.parse_args()
    
    asyncio.run(find_contacts(
        args.query,
        args.output,
        max_steps=args.max_steps,
        headless=args.headless
    ))

if __name__ == "__main__":
    main() 