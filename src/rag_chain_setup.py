from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_weaviate.vectorstores import WeaviateVectorStore
import os
from weaviate import WeaviateClient
import weaviate

def get_rag_chain_with_memory(client: WeaviateClient, conversation_history: list):
    """
    Initializes and returns the RAG chain with memory support.
    """
    try:
        load_dotenv()
        print("Connecting to Weaviate to get vector store...")
        
        print(f"Weaviate client ready: {client.is_ready()}")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vector_store = WeaviateVectorStore(
            client=client,
            index_name="Transcripts",
            embedding=embeddings,
            text_key="text"
        )
        
        retriever = vector_store.as_retriever()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        
        llm = ChatGroq(
            temperature=0,
            model="openai/gpt-oss-120b",
            model_kwargs={"seed": 0}
        )
        
        print("LLM ready")
        
        def process_query_with_memory(query: str):
            """Process query with memory - check if it references conversation history"""
            memory_keywords = ['summary', 'summarize', 'last response', 'previous answer', 'what did you say', 'what you told me', 'earlier']
            
            if any(keyword in query.lower() for keyword in memory_keywords):
                # Handle memory-based queries - get last assistant response
                if not conversation_history:
                    return "I don't have any previous responses to reference."
                
                # Find the last assistant message
                last_assistant_response = None
                for msg in reversed(conversation_history):
                    if msg["role"] == "assistant":
                        last_assistant_response = msg["content"]
                        break
                
                if not last_assistant_response:
                    return "I don't have any previous responses to reference."
                
                # For summary requests, directly process the last response
                if 'summary' in query.lower() or 'summarize' in query.lower():
                    summary_template = """Please provide a concise summary of the following response:

{last_response}

Summary:"""
                    
                    summary_prompt = ChatPromptTemplate.from_template(summary_template)
                    
                    response = (summary_prompt | llm | StrOutputParser()).invoke({
                        "last_response": last_assistant_response
                    })
                    
                    return response
                else:
                    # For other memory queries, use conversation context
                    history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-4:]])
                    
                    memory_template = """Based on our conversation history, answer the user's question.
                    
                    Conversation History:
                    {history}
                    
                    Question: {question}
                    
                    Answer based only on the conversation history above."""
                    
                    memory_prompt = ChatPromptTemplate.from_template(memory_template)
                    
                    response = (memory_prompt | llm | StrOutputParser()).invoke({
                        "history": history_context,
                        "question": query
                    })
                    
                    return response
            else:
                # Regular RAG query with context
                context = retriever.invoke(query)
                
                template = """You are a helpful and expert assistant on the topic of a YouTube video transcript. Use the provided context ONLY to answer the user's question.
                            Responses must cite the relevant video(s) and timestamp ranges. For each bullet point, cite the video transcript it came from.
                            Do not create new or fake video names. If you cannot find the answer, just say 'I don't know'.
                            
                            Context: {context}
                            Question: {question}"""
                
                prompt = ChatPromptTemplate.from_template(template)
                
                response = (prompt | llm | StrOutputParser()).invoke({
                    "context": context,
                    "question": query
                })
                
                return response
        
        print("RAG chain with memory successfully initialized.")
        return process_query_with_memory
        
    except Exception as e:
        print(f"An error occurred during RAG chain initialization: {e}")
        raise
