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
1. Clone the repository
```bash
git clone <repository-url>
cd rag-evaluation-system
```

2. Install dependencies
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
docker run -d -p 8080:8080 weaviate/weaviate:latest
```

## Usage

### Web Interface (Recommended)
Start the Streamlit app for an interactive chat interface:
```bash
streamlit run src/app.py
```
Then open your browser to `http://localhost:8501/`

**Features:**
- Interactive chat interface
- Real-time responses from video transcripts
- Automatic citations with timestamps
- Conversation history support

### Programmatic Usage
```python
from rag_chain_setup import get_rag_chain_with_memory
import weaviate

# Initialize
client = weaviate.connect_to_local()
rag_chain = get_rag_chain_with_memory(client, conversation_history=[])

# Ask questions
response = rag_chain("How can I make my video intros better?")
print(response)
```

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

## Evaluation Metrics

| Metric | Description | Scale |
|--------|-------------|-------|
| **Fluency** | Grammar and readability | 0-10 |
| **Relevance** | How well answer addresses question | 0-10 |
| **Groundedness** | Answer supported by context | 0-10 |
| **Faithfulness** | Accuracy to source, no hallucination | 0-10 |
| **Coherence** | Logical flow and consistency | 0-10 |
| **BERTScore** | Semantic similarity to expected answer | 0-1 |


## Key Design Decisions

- **Different LLMs** for generation vs evaluation to reduce bias
- **768-dimensional embeddings** for semantic search capability
- **Timestamp-based chunking** to preserve context and enable citations
- **Multi-metric evaluation** to capture different aspects of response quality
- **Streamlit interface** for user-friendly interaction and demonstration

## Sample Questions

- "How can I make my video intros better?"
- "What's the key to good storytelling?"
- "What are the best practices for YouTube content?"

## Performance

Current baseline performance on evaluation dataset:
- Average Fluency: 9.5/10
- Average Relevance: 6.5/10  
- Average Groundedness: 3.8/10
- Average BERTScore F1: 0.83

## Future Improvements

- [ ] Expand evaluation dataset
- [ ] Implement retrieval optimization
- [ ] Add more sophisticated chunking strategies
- [ ] Include human evaluation benchmarks
- [ ] Enhanced Streamlit UI with evaluation dashboard