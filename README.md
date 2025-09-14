# YouTube-Advisor
Chatbot that gives creators practical advice on how to improve their YouTube channel
YouTube Advisor Chatbot (LangChain Version)
This is a multi-turn chatbot that provides advice on YouTube video creation, specifically on topics like storytelling and video introductions. This version uses LangChain to handle the Retrieval-Augmented Generation (RAG) pipeline and manage conversational memory.

Folder Structure
main.py: The entry point of the application. It now initializes and starts the LangChain-based chatbot.

src/: Contains the core logic of the application.

chatbot.py: Holds the main chatbot logic, setting up the LangChain QA chain with conversational memory.

data/transcripts/: A folder for storing the raw transcript data.

Prerequisites
Before running the application, you need to install the required libraries and set up your API key.

Install dependencies:

pip install -r requirements.txt

Set up your Groq API key:
Create a .env file in the project's root directory and add your key:

GROQ_API_KEY ="your-api-key-here"

How to Run
Make sure you have all prerequisites installed and your API key is set.

Navigate to the project's root directory.

Run the main application from your terminal:

#TODO
python main.py

Example Usage
Once the chatbot is running, you can have a natural, multi-turn conversation. You can ask follow-up questions without having to repeat the full context.

"How can I make my video intros better?"

"What's the key to good storytelling?"

"Can you give me a summary of what you just told me?"

"What was the timestamp for that?"
