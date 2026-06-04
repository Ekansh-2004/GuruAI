import json
import random
from langchain_core.prompts import PromptTemplate
from src.core.llm import llm_creative
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from src.rag.embedder import load_existing_vectorstore

class QuizQuestion(BaseModel):
    subject: str = Field(description="The broad subject this question belongs to.")
    topic: str = Field(description="The specific sub-topic or concept. Limit to 1-3 words.")
    question: str = Field(description="The question text")
    options: List[str] = Field(description="Exactly 4 options for the question")
    correct_index: int = Field(description="Index of the correct option (0 to 3)")
    explanation: str = Field(description="Detailed explanation of why the answer is correct and others are wrong")

class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(description="List of 4 relevant questions based on chat history")

def generate_quiz_for_session_db(session_id: str, user_subjects: Optional[List[str]] = None) -> dict:
    """Generates a JSON-structured quiz drawing directly from the session's vectorstore."""
    db = load_existing_vectorstore(session_id)
    if not db:
        return {"questions": []}
        
    doc_dict = db.docstore._dict
    docs = list(doc_dict.values())
    
    if not docs:
        return {"questions": []}
        
    sample_size = min(10, len(docs))
    sampled_docs = random.sample(docs, sample_size)
    text_corpus = "\n\n---\n\n".join([doc.page_content for doc in sampled_docs])
    
    # Build subject constraint section
    if user_subjects and len(user_subjects) > 0:
        subject_list = ", ".join(f'"{s}"' for s in user_subjects)
        subject_instruction = f"""CRITICAL SUBJECT CONSTRAINT:
The student is ONLY studying these subjects: {subject_list}.
You MUST assign every question's 'subject' field to one of these EXACT names (case-sensitive).
Do NOT invent or use any other subject name whatsoever."""
    else:
        subject_instruction = """CRITICAL INSTRUCTION FOR CATEGORIZATION:
When generating the 'subject' field, use a broadly recognized CS subject (e.g., "Cyber Security", "Data Structures", "Machine Learning", "Operating Systems", "Computer Networks").
Then specify the precise sub-topic in the 'topic' field."""

    model = llm_creative
    parser = JsonOutputParser(pydantic_object=Quiz)

    if user_subjects and len(user_subjects) > 0:
        subject_list = ", ".join(f'"{s}"' for s in user_subjects)
        subject_label_rule = f"""SUBJECT LABELLING RULE (for the 'subject' field ONLY):
After writing each question from the textbook content, assign the 'subject' field to whichever of these matches best: {subject_list}.
This is ONLY a label — it does NOT affect what the question is about. The question content must still come entirely from the textbook excerpts above."""
    else:
        subject_label_rule = """SUBJECT LABELLING RULE (for the 'subject' field ONLY):
Assign a broad CS subject label (e.g., "Cyber Security", "Machine Learning", "Data Structures") to each question based on what the textbook content is about."""

    prompt = PromptTemplate(
        template="""You are Study AI generating a quiz STRICTLY from the following uploaded textbook content.

RULE 1 — CONTENT SOURCE (most important):
Every single question, option, and answer MUST be based ONLY on the textbook excerpts below.
Do NOT use your general knowledge. Do NOT add information not present in the text.
If the text is about V2X communication, ALL 4 questions must be about V2X communication.

RULE 2 — SUBJECT LABELLING:
{subject_label_rule}

RULE 3 — TOPIC FIELD:
The 'topic' field should be a specific concept from the textbook (1-3 words, e.g. "DSRC Protocol", "V2X Architecture").

Textbook Excerpts:
{text_corpus}

{format_instructions}
Do not include markdown outside the JSON block.""",
        input_variables=["text_corpus", "subject_label_rule"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | model | parser

    try:
        return chain.invoke({"text_corpus": text_corpus, "subject_label_rule": subject_label_rule})
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return {"questions": []}
