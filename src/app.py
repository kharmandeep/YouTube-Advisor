import streamlit as st
from rag_chain_setup import get_rag_chain_with_memory
from document_loader import load_and_add_documents
from dotenv import load_dotenv
import weaviate
import os

@st.cache_resource
def initialize_weaviate_client():
    """Initialize Weaviate client only"""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY environment variable not set. Please add it to your .env file.")
        st.stop()

    try:
        client = weaviate.connect_to_local()
        return client
    except Exception as e:
        st.error(f"Failed to initialize Weaviate client: {e}")
        st.stop()

def main():
    """
    The main function for the Streamlit application with memory.
    """
    st.set_page_config(page_title="YouTube Advisor", page_icon="📺")
    st.title("📺 YouTube Advisor")
    st.write("Ask questions about your YouTube video transcripts and get insights!")

    # Initialize Weaviate client once
    client = initialize_weaviate_client()

    # Initialize chat history for the session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get the query processor with current conversation history
                query_processor = get_rag_chain_with_memory(client, st.session_state.messages)
                
                # Process the query
                result = query_processor(prompt)
                
                # Handle different return types (memory vs RAG)
                if isinstance(result, tuple):
                    response, sources = result
                    st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
                else:
                    response = result
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                st.markdown(response)

if __name__ == "__main__":
    main()
