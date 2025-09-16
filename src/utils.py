from langchain.docstore.document import Document
import re

def parse_transcript_to_documents(file_path: str, source_name: str) -> list[Document]:
    """
    Parses a transcript file, extracting text and timestamps to create a list of Documents.
    Each speaker's turn becomes a separate document.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to find each block of dialogue with its timestamp
    # It captures the start and end timestamps and the dialogue text
    pattern = re.compile(r'^\d+\s*([\d:.]+)\s*-->\s*([\d:.]+)\s*([\s\S]+?)(?=\n^\d+|\Z)', re.MULTILINE)
    
    documents = []
    for match in pattern.finditer(content):
        start_time, end_time, text_content = match.groups()
        
        # Clean up the text content
        cleaned_text = text_content.strip()
        
        if cleaned_text:
            # Create a Document for each speaker's turn with metadata
            doc = Document(
                page_content=cleaned_text,
                metadata={
                    "source": source_name,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
            documents.append(doc)
            
    return documents