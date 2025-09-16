# RAG Evaluation System

A Retrieval-Augmented Generation (RAG) system with comprehensive evaluation framework for querying YouTube video transcripts. Built to demonstrate RAG implementation and systematic performance measurement.

## Features

- **Smart Document Retrieval**: Uses Weaviate vector database with semantic search
- **Contextual Response Generation**: Powered by Groq LLMs with automatic citation
- **Comprehensive Evaluation**: 6-metric assessment framework for response quality
- **Memory Support**: Handles conversation history and follow-up questions
- **Automated Scoring**: LLM-as-a-judge evaluation with numerical metrics
- **Interactive Web Interface**: Streamlit-based chat interface for easy usage

## Tech Stack

- **Web Interface**: Streamlit
- **Vector Database**: Weaviate
- **Embeddings**: HuggingFace `sentence-transformers/all-mpnet-base-v2` 
- **LLM**: Groq (`openai/gpt-oss-120b` for generation, `llama-3.3-70b-versatile` for evaluation)
- **Framework**: LangChain
- **Evaluation**: BERTScore + Custom LLM judges

## Installation

### Prerequisites
- Python 3.8+
- Groq API key
- Weaviate (local installation)

### Setup



1. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
# Create .env file
GROQ_API_KEY=your_groq_api_key_here
```

4. Start Weaviate (local)
```bash
# Follow Weaviate local installation guide
docker compose up -d
```

## Usage

### Web Interface (Recommended)
Start the Streamlit app for an interactive chat interface:
```bash
streamlit run src/app.py
```
Then open your browser to `http://localhost:8501/`

### Running Evaluation
```bash
python src/eval.py
```

### Sample Output
```
--- Evaluating question: 'How can I make my video intros better?' ---
Chatbot response: Focus on making the intro captivating – a compelling opening keeps viewers engaged...
Fluency Score: 9.00
Relevance Score: 8.00
Groundedness Score: 7.50
Faithfulness Score: 8.50
Coherence Score: 9.00
BERTScore F1: 0.82
```


## Sample Questions

- "How can I make my video intros better?"
- "What's the key to good storytelling?"
- "What are the best practices for YouTube content?"


## Future Improvements

- [ ] Expand evaluation dataset
- [ ] Expand chatbot functionality to answer questions like "Summarize the whole transcript" 
- [ ] Implement retrieval optimization
- [ ] Add more sophisticated chunking strategies
- [ ] Include human evaluation benchmarks
- [ ] Enhanced Streamlit UI with evaluation dashboard