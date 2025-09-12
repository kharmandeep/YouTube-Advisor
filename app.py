import os
import streamlit as st
from dotenv import load_dotenv
from src.chatbot import initialize_chatbot

# Load environment variables
load_dotenv()

# --- Initialize Chatbot in Session State ---
# This ensures the chatbot is only initialized once per session,
# which is crucial for performance.
if "qa_chain" not in st.session_state:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables.")
        st.stop()
    st.session_state.qa_chain = initialize_chatbot(groq_api_key)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Streamlit UI ---
st.title("YouTube Advisor Chatbot")
st.write("I'm an expert assistant on YouTube content creation. Ask me about video intros, storytelling, or anything else from the transcripts!")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        # Get response from the chatbot
        try:
            response = st.session_state.qa_chain.invoke({"question": prompt})
            full_response = response['answer']
        except Exception as e:
            full_response = f"An error occurred: {e}"

        # Display the full response
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
