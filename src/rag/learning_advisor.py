"""Learning-guidance advisor — a dedicated pre-processing LLM call, separate
from answer generation.

Given the student's question, their full knowledge profile, and the
prerequisite graph for the subjects they're tracking, it produces a short
plain-text briefing on how deep/paced the eventual answer should be — e.g.
"the student is weak on this topic's prerequisite, slow down and review that
first" or "the student is strong here, be terse and technical".

This is a deliberate extra API call (not folded into the answer-generation
prompt) so the mastery/prerequisite reasoning stays fully separate from, and
inspectable independently of, the final answer — at the cost of one more
sequential call (and its latency) before the chat response starts streaming.

The model is NOT asked to match topic names exactly against the profile or
graph — it's handed both as loosely-worded text and expected to reason past
naming differences itself (e.g. "Newton's 2nd Law" vs. "Newtons Laws Of
Motion"), the same way chain.py's adaptive section already does today.
"""
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.llm import llm_default
from src.personalization import mastery, topic_graph

_ADVISOR_PROMPT = PromptTemplate(
    template="""You are a learning-guidance advisor for a CS tutoring system. You do NOT
answer the student's question — you produce a short briefing that another
model will use to decide how to answer it.

STUDENT QUESTION:
{question}

STUDENT KNOWLEDGE PROFILE (topic names are informal, may not match the graph exactly):
{profile_summary}

SUBJECT PREREQUISITE GRAPH (topic names are broad/canonical, may not match the profile exactly):
{graph_context}

TASK:
1. Identify which profile topic(s) this question is most likely about, even if the wording differs.
2. Check the prerequisite graph for that topic. If a prerequisite exists and the profile shows it's weak
   (or the profile doesn't mention it at all — treat as unproven), flag that explicitly.
3. State in 2-4 short sentences: how proficient the student likely is on this question's topic, whether a
   prerequisite gap should be addressed first, and whether the answer should be foundational/slow-paced or
   terse/advanced.

If the profile has nothing relevant to this question, or the graph has no data for the relevant subject,
say so briefly and recommend a normal, unmodified explanation.

Output ONLY the briefing text — no headers, no markdown, no preamble.""",
    input_variables=["question", "profile_summary", "graph_context"],
)


def build_learning_guidance(user_id: int, question: str) -> str:
    """Returns a short guidance briefing, or "" if there's nothing to reason
    about yet (new user, no quiz history) or the advisor call fails — callers
    should treat an empty string as "fall back to normal behavior", not an error.
    """
    profile_summary = mastery.build_profile_summary(user_id)
    if not profile_summary:
        return ""  # nothing to reason about — skip the call entirely

    subjects = list(mastery.load_global_profile(user_id).keys())
    graph_context = topic_graph.format_graph_context(subjects)

    chain = _ADVISOR_PROMPT | llm_default | StrOutputParser()
    try:
        return chain.invoke({
            "question": question,
            "profile_summary": profile_summary,
            "graph_context": graph_context or "(no prerequisite graph data for these subjects yet)",
        }).strip()
    except Exception as e:
        print(f"[learning_advisor] Guidance generation failed: {e}")
        return ""
