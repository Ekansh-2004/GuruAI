"""
Corrective RAG (CRAG) — Relevance Grader
=========================================
Uses Google Gemini to evaluate all retrieved documents in a single batch
against the user's question. Documents that are graded as not relevant
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
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        max_retries=2,
    )


# ── Relevance grading prompt ──────────────────────────────────────────────────
_GRADER_PROMPT = PromptTemplate(
    template="""You are a  relevance grader for a Retrieval-Augmented Generation (RAG) pipeline.

Your task: decide which of the retrieved DOCUMENTS contain information that is
useful for answering the USER QUESTION.

Respond with ONLY a comma-separated list of the integer IDs of the relevant documents (e.g., 0, 2).
If NONE of the documents are relevant, respond with the exact word "none".
Do not explain. Do not add any other text.

USER QUESTION:
{question}

RETRIEVED DOCUMENTS:
{documents}

Relevant document IDs:""",
    input_variables=["question", "documents"],
)

_grader_chain = _GRADER_PROMPT | _get_grader_model() | StrOutputParser()


# ── Public API ────────────────────────────────────────────────────────────────

def grade_documents(question: str, docs: List[Document]) -> List[Document]:
    """
    Grade all documents in a single batch for relevance to `question` using Gemini.
    Returns the subset of docs rated relevant.
    """
    if not docs:
        return []

    formatted_docs = ""
    for i, doc in enumerate(docs):
        formatted_docs += f"--- Document ID: {i} ---\n{doc.page_content}\n\n"

    try:
        verdict = _grader_chain.invoke({
            "question": question,
            "documents": formatted_docs,
        }).strip().lower()

        if verdict == "none":
            print(f"[CRAG] ❌ All docs graded IRRELEVANT — discarded.")
            return []

        import re
        relevant_indices = set(int(idx) for idx in re.findall(r'\d+', verdict))

        relevant = []
        for i, doc in enumerate(docs):
            if i in relevant_indices:
                relevant.append(doc)
                print(f"[CRAG] ✅ Doc {i} graded RELEVANT (first 80 chars): {doc.page_content[:80]!r}")
            else:
                print(f"[CRAG] ❌ Doc {i} graded IRRELEVANT — discarded.")
        return relevant
    except Exception as e:
        # On grader error, be conservative: keep all docs
        print(f"[CRAG] ⚠️  Grader error ({e}), keeping all docs by default.")
        return docs


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
