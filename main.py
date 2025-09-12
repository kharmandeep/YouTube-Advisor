import os
from dotenv import load_dotenv

from src.chatbot import initialize_chatbot

# Load environment variables from .env file
load_dotenv()

# Get the API key for Groq from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")


def main():
    """
    Main function to run the multi-turn chatbot.
    """
    print("Initializing chatbot...")
    # Initialize the LangChain-based chatbot with the Groq API key
    qa = initialize_chatbot(GROQ_API_KEY)

    print("\nChatbot Initialized! You can start a conversation now.")
    print("Type 'exit' to end the chat.")

    # Start the conversational loop
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Get the chatbot's response
        try:
            response = qa.invoke({"question": user_input})
            print(f"Chatbot: {response['answer']}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("calling main")
    main()
