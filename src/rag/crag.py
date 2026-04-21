"""
Corrective RAG (CRAG) — Relevance Grader
=========================================
Uses Google Gemini (gemini-3.1-flash-lite) to evaluate each retrieved document
against the user's question.  Documents that are graded "no" (not relevant)
are silently discarded before the answer is generated.

Two public helpers are exposed:
  • grade_documents(question, docs) → list[Document]
      Returns only the docs that Gemini considers relevant.
  • build_crag_context(retriever, question) → (str, str)
      Returns (formatted_context, source_label) ready for prompt injection.
      source_label is "[Textbook]" or "[General Knowledge — no relevant
      textbook content found]" so the chain can tell the user.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.core.config import GOOGLE_API_KEY


# ── Gemini grader model (lightweight flash for speed) ────────────────────────
def _get_grader_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_retries=2,
    )


# ── Relevance grading prompt ──────────────────────────────────────────────────
_GRADER_PROMPT = PromptTemplate(
    template="""You are a strict relevance grader for a Retrieval-Augmented Generation (RAG) pipeline.

Your task: decide whether the retrieved DOCUMENT contains information that is
useful for answering the USER QUESTION.

Respond with ONLY a single word — either:
  yes   → the document is relevant and useful
  no    → the document is irrelevant or does not help answer the question

Do not explain. Do not add any other text.

USER QUESTION:
{question}

RETRIEVED DOCUMENT:
{document}

Relevance verdict (yes/no):""",
    input_variables=["question", "document"],
)

_grader_chain = _GRADER_PROMPT | _get_grader_model() | StrOutputParser()


# ── Public API ────────────────────────────────────────────────────────────────

def grade_documents(question: str, docs: List[Document]) -> List[Document]:
    """
    Grade each document for relevance to `question` using Gemini.
    Returns the subset of docs rated 'yes'.
    """
    relevant = []
    for doc in docs:
        try:
            verdict = _grader_chain.invoke({
                "question": question,
                "document": doc.page_content,
            }).strip().lower()
            if verdict.startswith("yes"):
                relevant.append(doc)
                print(f"[CRAG] ✅ Doc graded RELEVANT (first 80 chars): {doc.page_content[:80]!r}")
            else:
                print(f"[CRAG] ❌ Doc graded IRRELEVANT — discarded.")
        except Exception as e:
            # On grader error, be conservative: keep the doc
            print(f"[CRAG] ⚠️  Grader error ({e}), keeping doc by default.")
            relevant.append(doc)
    return relevant


def build_crag_context(retriever, question: str) -> Tuple[str, str]:
    """
    Full CRAG retrieval + grading pipeline.

    Steps:
      1. Retrieve top-k docs from the FAISS retriever.
      2. Grade each doc with Gemini.
      3. If ≥1 relevant doc is found → return their combined text + "[Textbook]".
      4. If 0 relevant docs → return a fallback notice + "[General Knowledge]"
         so the LLM knows to answer from its own training data.

    Returns:
        (context_text, source_label)
    """
    raw_docs: List[Document] = retriever.invoke(question)
    print(f"[CRAG] Retrieved {len(raw_docs)} docs, grading now …")

    relevant_docs = grade_documents(question, raw_docs)
    total = len(raw_docs)
    kept  = len(relevant_docs)
    print(f"[CRAG] Kept {kept}/{total} docs after Gemini grading.")

    if relevant_docs:
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        source_label = "[Textbook]"
    else:
        context = (
            "No relevant textbook content was found for this question after "
            "Corrective RAG filtering. Answer from your general knowledge as a "
            "CS tutor, but explicitly tell the student that this information does "
            "NOT come from their uploaded documents."
        )
        source_label = "[General Knowledge — no relevant textbook content found]"

    return context, source_label
