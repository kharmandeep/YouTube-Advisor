import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import weaviate
from weaviate import WeaviateClient

from utils import parse_transcript_to_documents

def load_and_add_documents(client: WeaviateClient):
    try:
        # 1. Load transcripts from specific files
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'transcripts')
        all_documents = []

        # Check if the data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"The data directory was not found: {data_dir}")
        
        # Load specific transcript files by name
        file_path_1 = os.path.join(data_dir, "video_1_transcript.txt")
        file_path_2 = os.path.join(data_dir, "video_2_transcript.txt")

        # Check if each transcript file exists
        if not os.path.exists(file_path_1):
            raise FileNotFoundError(f"Transcript file not found: {file_path_1}")
        if not os.path.exists(file_path_2):
            raise FileNotFoundError(f"Transcript file not found: {file_path_2}")
        
        all_documents.extend(parse_transcript_to_documents(file_path_1, "video_1_transcript.txt"))
        all_documents.extend(parse_transcript_to_documents(file_path_2, "video_2_transcript.txt"))

        print(f"Loaded {len(all_documents)} raw documents.")
        

        # 2. Split documents into chunks for more effective retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        docs = text_splitter.split_documents(all_documents)
        print(f"Split into {len(docs)} document chunks.")

        # 3. Create the embeddings and add documents to Weaviate
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        vector_store = WeaviateVectorStore.from_documents(
            documents = docs,
            embedding = embeddings,
            client = client,
            index_name="Transcripts"
        )
        print(f"Vector Store {vector_store}")
        print("Documents successfully added to the Weaviate database.")
        

    except Exception as e:
        print(f"An error occurred during document loading and embedding: {e}")
        raise
        
