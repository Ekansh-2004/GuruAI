import json
import random
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from src.core.config import GOOGLE_API_KEY
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from src.rag.embedder import load_existing_vectorstore

class QuizQuestion(BaseModel):
    topic: str = Field(description="A broad, high-level, generalized category name (like 'Software Metrics' or 'Data Structures'). Avoid highly specific sub-topics. Limit to 1-3 words.")
    question: str = Field(description="The question text")
    options: List[str] = Field(description="Exactly 4 options for the question")
    correct_index: int = Field(description="Index of the correct option (0 to 3)")
    explanation: str = Field(description="Detailed explanation of why the answer is correct and others are wrong")

class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(description="List of 4 relevant questions based on chat history")

def generate_quiz_for_session_db(session_id: str) -> dict:
    """Generates a JSON-structured quiz drawing directly from the session's vectorstore."""
    db = load_existing_vectorstore(session_id)
    if not db:
        return {"questions": []}
        
    # Extract raw documents stored inside the local FAISS index
    doc_dict = db.docstore._dict
    docs = list(doc_dict.values())
    
    if not docs:
        return {"questions": []}
        
    # Sample up to 10 random chunks to create a comprehensive quiz
    sample_size = min(10, len(docs))
    sampled_docs = random.sample(docs, sample_size)
    text_corpus = "\n\n---\n\n".join([doc.page_content for doc in sampled_docs])
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
    parser = JsonOutputParser(pydantic_object=Quiz)
    
    prompt = PromptTemplate(
        template="""You are Study AI. Based ONLY on the following textbook excerpts uploaded by the student, generate a high-quality multiple choice quiz with exactly 4 questions to test the student's understanding of the facts precisely detailed within.

CRITICAL INSTRUCTION FOR TOPICS:
When generating the 'topic' field for each question, you MUST use very broad, high-level academic categories (e.g., "Software Engineering", "Machine Learning", "Data Structures", "Networking"). DO NOT use highly specific sub-topics or detailed concept names. For example, instead of "Halstead Formulas" or "Halstead Variables", just use "Software Metrics". Keep it generalized.

Textbook Excerpts:
{text_corpus}

{format_instructions}
Do not include markdown outside the JSON block.""",
        input_variables=["text_corpus"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | model | parser
    
    try:
        return chain.invoke({"text_corpus": text_corpus})
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return {"questions": []}
