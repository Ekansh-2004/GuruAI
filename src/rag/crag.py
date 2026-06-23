"""
Corrective RAG (CRAG) — Relevance Grader
=========================================
Uses Google Gemini to evaluate all retrieved documents in a single batch
against the user's question. Documents that are graded as not relevant
are silently discarded before the answer is generated.

Two public helpers are exposed:
  • grade_documents(question, docs) → list[Document]
      Returns only the docs that Gemini considers relevant.
  • build_crag_context(retriever, question) → (str, str, list)
      Returns (formatted_context, source_label, sources_metadata) ready for
      prompt injection and UI display.
      source_label is "[Textbook]", "[Web Search]", or "[General Knowledge]".
      sources_metadata is a list of dicts with keys: type, title, snippet, url.
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

_grader_chain = None

def _get_grader_chain():
    global _grader_chain
    if _grader_chain is None:
        _grader_chain = _GRADER_PROMPT | _get_grader_model() | StrOutputParser()
    return _grader_chain


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
        verdict = _get_grader_chain().invoke({
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


def _web_search(query: str, max_results: int = 3):
    """
    Perform a web search using ddgs.
    Returns (formatted_context_str, raw_results_list).
    raw_results_list items have keys: title, href, body.
    """
    try:
        from ddgs import DDGS
        print(f"[CRAG Web Search] Querying: {query!r}")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            print("[CRAG Web Search] No results returned from DuckDuckGo.")
            return "", []

        formatted_results = []
        for i, r in enumerate(results):
            title = r.get("title", "No Title")
            href  = r.get("href", "")
            body  = r.get("body", "")
            formatted_results.append(
                f"--- Web Search Result {i+1}: {title} ---\nSource URL: {href}\nContent: {body}\n"
            )

        return "\n".join(formatted_results), results
    except Exception as e:
        print(f"[CRAG Web Search] Error during web search: {e}")
        return "", []


def build_crag_context(retriever, question: str) -> Tuple[str, str, list]:
    """
    Full CRAG retrieval + grading pipeline.

    Steps:
      1. Retrieve top-k docs from the FAISS retriever.
      2. Grade each doc with Gemini.
      3. If ≥1 relevant doc is found → return their combined text + "[Textbook]".
      4. If 0 relevant docs → trigger web search fallback.
      5. If web search succeeds → return search context + "[Web Search]".
      6. If web search has no results/fails → return a fallback notice + "[General Knowledge]".

    Returns:
        (context_text, source_label, sources_metadata)
        sources_metadata is a list of dicts:
            { "type": "textbook"|"web", "title": str, "snippet": str, "url": str|None }
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
        # Build metadata from LangChain Document.metadata (source filename + page)
        sources_metadata = []
        for doc in relevant_docs:
            meta  = doc.metadata or {}
            fname = meta.get("source", "Textbook")
            # Normalise to just the basename so we don't leak full server paths
            import os as _os
            fname = _os.path.basename(fname) if fname else "Textbook"
            page  = meta.get("page")
            title = f"{fname} · Page {page + 1}" if page is not None else fname
            sources_metadata.append({
                "type":    "textbook",
                "title":   title,
                "snippet": doc.page_content[:300],
                "url":     None,
            })
    else:
        print("[CRAG] ❌ All docs graded IRRELEVANT. Triggering Web Search fallback...")
        web_context, web_results = _web_search(question)
        if web_context:
            print("[CRAG] ✅ Web Search fallback succeeded.")
            context = web_context
            source_label = "[Web Search]"
            sources_metadata = [
                {
                    "type":    "web",
                    "title":   r.get("title", "Web Result"),
                    "snippet": r.get("body", "")[:300],
                    "url":     r.get("href", None),
                }
                for r in web_results
            ]
        else:
            print("[CRAG] ❌ Web Search fallback returned no results or failed. Falling back to General Knowledge.")
            context = (
                "No relevant textbook content was found for this question after "
                "Corrective RAG filtering, and web search did not return results. "
                "Answer from your general knowledge as a CS tutor, but explicitly "
                "tell the student that this information does NOT come from their "
                "uploaded documents."
            )
            source_label = "[General Knowledge — no relevant textbook content found]"
            sources_metadata = []

    return context, source_label, sources_metadata
