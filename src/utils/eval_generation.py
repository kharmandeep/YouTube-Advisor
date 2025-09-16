import re
from typing import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_llm_judge():
    """Get the LLM judge instance with API key from environment."""
    return ChatGroq(model="openai/gpt-oss-120b")

def _get_llm_response(prompt: ChatPromptTemplate, input_data: dict) -> str:
    """Helper function to get a response from an LLM for evaluation."""
    llm_judge = get_llm_judge()
    return (prompt | llm_judge | StrOutputParser()).invoke(input_data)

def evaluate_fluency(generated_text: str) -> float:
    """
    Evaluates the fluency of the generated text.
    For this basic implementation, we will use an LLM to judge fluency.
    """
    prompt = ChatPromptTemplate.from_template(
        """Rate the following text on a scale of 0 to 10 for fluency and grammatical correctness.
        A score of 10 is perfectly fluent and grammatically correct.
        
        Text: {text}
        
        Return ONLY the numerical score."""
    )
    
    score_text = _get_llm_response(prompt, {"text": generated_text})
    
    try:
        score = float(score_text.strip())
        return score
    except ValueError:
        # Fallback if the LLM does not return a number
        return 0.0

def evaluate_relevance(generated_text: str, expected_answer: str) -> float:
    """
    Evaluates the relevance of the generated text to the expected answer.
    """
    prompt = ChatPromptTemplate.from_template(
        """Rate the following generated answer on a scale of 0 to 10 for relevance to the expected answer.
        A score of 10 means the generated answer is highly relevant.
        
        
        Expected Answer: {expected_answer}
        
        Generated Answer: {generated_text}
        
        Return ONLY the numerical score."""
    )
    
    score_text = _get_llm_response(prompt, {"expected_answer": expected_answer, "generated_text": generated_text})
    
    try:
        score = float(score_text.strip())
        return score
    except ValueError:
        # Fallback if the LLM does not return a number
        return 0.0
