from src.rag_chain_setup1 import get_rag_chain
from document_loader import load_and_add_documents
from dotenv import load_dotenv
import weaviate
import os

def run_test_query():
    """
    A simple test script to run a query through the RAG chain on the command line.
    """
    print("Initializing RAG chain for a test query...")
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    with weaviate.connect_to_local() as client:
        # Uncomment the line below if you need to load documents before testing
        # load_and_add_documents(client)

        rag_chain = get_rag_chain(client)
        question = "What's the key to good storytelling?"
        # The chain now expects a dictionary input, just like in the Streamlit app.
        # We wrap the question in a dictionary to match this expectation.
        response = rag_chain.invoke({"question": question})

        print("\n--- Test Question ---")
        print(f"Question: {question}")
        print("\n--- RAG Chain Response ---")
        print(response)

if __name__ == "__main__":
    try:
        run_test_query()
    except Exception as e:
        print(f"An error occurred during the test script: {e}")
        print("Please ensure your Weaviate instance is running and your .env file has the GROQ_API_KEY set.")
