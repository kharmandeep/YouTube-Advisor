import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate


def initialize_chatbot(api_key):
    """
    Initializes the LangChain chatbot with Groq's API.

    Args:
        api_key (str): The Groq API key.

    Returns:
        ConversationalRetrievalChain: The initialized chatbot chain.
    """
    
    # 1. Load transcripts from specific files
        # The path is modified to correctly point to the 'transcripts' folder
    # which is at the same level as the 'src' folder.
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'transcripts')
    texts = []
    
    # Load specific transcript files by name
    file_path_1 = os.path.join(data_dir, "video_1_transcript.txt")
    file_path_2 = os.path.join(data_dir, "video_2_transcript.txt")
    
    # Use TextLoader to load each file individually
    loader_1 = TextLoader(file_path_1)
    loader_2 = TextLoader(file_path_2)

    texts.extend(loader_1.load())
    texts.extend(loader_2.load())

    # 2. Split documents into chunks for more effective retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = text_splitter.split_documents(texts)

    # 3. Create embeddings using SentenceTransformer model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create a FAISS vector store to index the document embeddings
    vector_store = FAISS.from_documents(docs, embeddings)

    # 5. Initialize the language model using Groq's API
    # We are using Mixtral as a powerful alternative to the Qwen model.
    llm = ChatGroq(temperature=0.7, groq_api_key=api_key, model_name="llama-3.1-8b-instant")

    # 6. Set up the memory to handle conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 7. Define a prompt template to give the chatbot more context and instructions
    # You can edit this template to make the prompt as detailed as you need.
    prompt_template = """You are a helpful and expert assistant on the topic of a YouTube video transcript. 
    Use the provided context ONLY to answer the user's question. 
    Your goal is to answer the user's question by extracting key points from the provided video transcripts. 
    Format your answer as a list of bullet points. 
    Responses must cite the relevant video(s) and timestamp ranges. Choose a clear, machine-checkable format, e.g.:
    [source: video_1 t=00:12:34–00:13:10].For each bullet point, cite the video transcript it came from.
    Only cite from "video_1_transcript.txt" or "video_2_transcript.txt". Do not create new or fake video names.
    Do not make up information. If you cannot find the answer, just say 'I don't know'.
    
    Context: {context}
    Question: {question}"""

    # 7. Create the conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
    )

    return qa
