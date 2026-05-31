from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from src.core.config import GROQ_API_KEY
from src.rag.crag import build_crag_context           # ← CRAG grading step


def build_rag_chain(retriever, knowledge_profile_summary: str = "", user_memory_context: str = ""):
    # Same Groq / Llama model as before — only the retrieval step changes.
    model = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, max_retries=0)

    # ── 1. User Memory ────────────────────────────────────────────────────────
    memory_section = ""
    if user_memory_context:
        memory_section = f"""ABOUT THIS STUDENT (treat as ground truth):
{user_memory_context}

MEMORY APPLICATION RULES — read carefully:
- "likes analogies / simple explanations" → ALWAYS shape your explanation style accordingly
- "likes C++ / Python / any language" → this is a code PREFERENCE, NOT a reason to add code
  ▶ ONLY write code if the user's message explicitly asks for code, implementation, or an example
  ▶ A conceptual question like "explain X" or "what is X" → NO code, ever
- Background facts (year, field) → calibrate depth only
- Questions about own preferences → answer from this memory directly

"""

    # ── 2. Core tutor behaviour ────────────────────────────────────────────────
    base_system = f"""{memory_section}You are Study AI — a sharp, concise CS tutor.
TEACHING RULES:
- Use the Textbook Context to gather the facts, but NEVER just copy-paste the textbook verbatim.
- Act like a human tutor: synthesize the information and explain it in your own words so the student actually understands it.
- If the question requires an enumeration (like "list the uses"), make sure to include all points from the context, but explain them conceptually instead of rigidly copying original numbering.

PRECISION & COMPLETENESS RULES:
- If a user asks for a list, categories, or "what are the uses", you MUST extract ALL relevant points from the Textbook Context. 
- Do not summarize a multi-point list into a single sentence.
- Maintain the original numbering/lettering (e.g., a, b, c...) if present in the context.

RESPONSE LENGTH RULES:
- If the question is simple, be short and crisp.
- If the question requires an enumeration (like "list the uses"), provide the FULL detailed list regardless of length.
- Do NOT pad with intros like "Great question!"
- NO code unless explicitly requested.

SOURCE LABELLING & CITATION:
- Use [Textbook] if answering from provided document context.
- Use [General knowledge] otherwise.
- AT THE VERY END of your response, you MUST append a section titled "### Context/Sources Used:" and display the exact excerpts or comments from the Textbook Facts that you relied on to generate the answer. If you used General Knowledge, just state General Knowledge used."""

    # ── 3. Adaptive depth ─────────────────────────────────────────────────────
    adaptive_section = ""
    if knowledge_profile_summary:
        adaptive_section = f"""

STUDENT KNOWLEDGE PROFILE:
{knowledge_profile_summary}

CRITICAL RULES FOR ADAPTATION (YOU MUST OBEY THESE STRICTLY):
1. Immediately cross-reference the student's question topic with the STUDENT KNOWLEDGE PROFILE above.
2. If the topic is found with a score < 50% (WEAK):  You MUST radically shift your tone to be extremely simple. Use a child-like, real-world analogy. Avoid all technical jargon. Move painfully slow step-by-step.
3. If the topic is found with a score > 75% (STRONG):  You MUST radically shift your tone to be highly concise and extremely technical. Provide zero analogies. Assume the student is already an expert and only wants advanced nuances.
4. If it's 50-75% (AVERAGE) Give a balanced response.
5. If the topic is missing from the profile: Answer normally, do NOT prepend any tag.

Your tone MUST drastically change depending on the score. A WEAK explanation and a STRONG explanation MUST sound like they were written by two entirely different people."""

    system_prompt = base_system + adaptive_section

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", (
            "Use the following textbook facts as your ground-truth to teach the student.\n\n"
            "Textbook Facts (Corrective RAG — only relevant documents kept):\n{context}\n\n"
            "Student Question: {question}\n"
            "Your Explanation:"
        ))
    ])

    # ── 4. CRAG-aware chain ────────────────────────────────────────────────────
    # We override the context lambda to run the full CRAG pipeline:
    #   retrieve → Gemini grade → filter → format
    # Everything else (prompt, model, parser) is identical to before.
    chain = (
        {
            # build_crag_context returns (context_str, source_label).
            # We only need the context text here; source_label is printed to
            # the server log so the developer can see whether CRAG fired.
            "context": lambda x: build_crag_context(retriever, x["question"])[0],
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return chain