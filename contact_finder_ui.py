import streamlit as st
import asyncio
import json
from datetime import datetime
from contact_finder import find_contacts
from contact_info import ContactInfo, ContactSearchResults
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Contact Finder",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .contact-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🔍 Contact Finder")
st.markdown("""
    Find contact information from websites automatically. Enter a search query and the tool will search for:
    - Email addresses
    - Phone numbers
    - Social media profiles
    - Contact pages
""")

# Create two columns for input and results
col1, col2 = st.columns([1, 2])

with col1:
    # Search input
    search_query = st.text_input("Search Query", placeholder="Enter company name or website...")
    
    # Advanced options in an expander
    with st.expander("Advanced Options"):
        output_file = st.text_input("Output File", value="contacts.json")
        max_steps = st.slider("Maximum Search Steps", 5, 30, 20)
        headless = st.checkbox("Run in Background", value=False)
    
    # Search button
    if st.button("🔍 Start Search", type="primary"):
        if search_query:
            with st.spinner("Searching for contacts..."):
                # Create a unique output file for this search
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"contacts_{timestamp}.json"
                
                # Run the search
                asyncio.run(find_contacts(
                    search_query=search_query,
                    output_file=output_path
                ))
                
                # Store the output path in session state
                st.session_state.last_output = output_path
                st.session_state.search_completed = True
        else:
            st.error("Please enter a search query")

with col2:
    # Results section
    st.subheader("Search Results")
    
    if 'search_completed' in st.session_state and st.session_state.search_completed:
        try:
            # Load and display results
            with open(st.session_state.last_output, 'r') as f:
                results = json.load(f)
                
            # Display summary
            st.success(f"Found {results['total_found']} contacts!")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Contact List", "Data Table"])
            
            with tab1:
                # Display contacts in cards
                for contact in results['contacts']:
                    with st.container():
                        st.markdown(f"""
                            <div class="contact-card">
                                <h4>Contact Information</h4>
                                <p><strong>Source:</strong> {contact['source_url']}</p>
                                {f"<p><strong>Email:</strong> {contact['email']}</p>" if contact['email'] else ""}
                                {f"<p><strong>Phone:</strong> {contact['phone']}</p>" if contact['phone'] else ""}
                                {f"<p><strong>Social Links:</strong> {', '.join(contact['social_links'])}</p>" if contact['social_links'] else ""}
                                <p><strong>Found at:</strong> {contact['found_at']}</p>
                            </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                # Convert to DataFrame for better display
                df = pd.DataFrame(results['contacts'])
                st.dataframe(df, use_container_width=True)
                
                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"contacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
    else:
        st.info("Enter a search query and click 'Start Search' to begin")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Contact Finder - Powered by Browser Use</p>
    </div>
""", unsafe_allow_html=True) 