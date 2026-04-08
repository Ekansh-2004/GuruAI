from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.core.config import GOOGLE_API_KEY


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, knowledge_profile_summary: str = "", user_memory_context: str = ""):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

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

RESPONSE LENGTH RULES (strictly enforce):
- Keep answers SHORT and CRISP. Aim for 3–6 sentences for most questions.
- Do NOT pad with intros like "Great question!" or outros like "I hope this helps!"
- Do NOT add bullet-pointed lists of "things to consider" unless directly asked
- If genuinely complex, you may go up to 2 short paragraphs — no more
- NO code unless the user explicitly asks for code, an implementation, or a program

SOURCE LABELLING:
- Use [Textbook] if answering from provided document context
- Use [General knowledge] otherwise

Follow Up questions (mandatory — always end every response with this exact block):
After your answer, on a new line write exactly:
1. <most natural next question>
2. <second likely question>
3. <third likely question>

These should be the 3 most probable questions the student would ask next based on your answer."""

    # ── 3. Adaptive depth ─────────────────────────────────────────────────────
    adaptive_section = ""
    if knowledge_profile_summary:
        adaptive_section = f"""

STUDENT KNOWLEDGE PROFILE (adapt depth accordingly):
{knowledge_profile_summary}
- WEAK topic (<50%): use analogies, avoid jargon, step-by-step
- AVERAGE (50-75%): balanced, reinforce key points
- STRONG (>75%): concise, technical, focus on nuance
- Not in profile: standard CS student level"""

    system_prompt = base_system + adaptive_section

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Textbook Context (use only if directly relevant):\n{context}\n\nQuestion: {question}\nAnswer:")
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"]
        }
        | prompt
        | model
        | StrOutputParser()
    )
    return chain
