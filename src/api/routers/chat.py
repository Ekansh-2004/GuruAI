"""Streaming RAG chat endpoint."""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.api import retriever_cache
from src.api.deps import get_current_user, verify_session_ownership
from src.api.schemas import ChatRequest
from src.personalization import mastery, user_memory
from src.rag.chain import build_rag_chain
from src.rag.crag import build_crag_context
from src.sessions import store

router = APIRouter(prefix="/api", tags=["chat"])

# Only the last few turns are replayed to the LLM, to keep prompt size bounded.
_HISTORY_TURNS = 4


@router.post("/chat")
def chat(req: ChatRequest, user_id: int = Depends(get_current_user)):
    """Answer a question against the session's documents, streamed as SSE.

    Emits `{"chunk": ...}` frames as the model produces tokens, then a single
    `{"sources": [...]}` frame, then `[DONE]`. The full answer and its sources
    are persisted once the stream completes.
    """
    verify_session_ownership(req.session_id, user_id)
    retriever = retriever_cache.get(req.session_id)
    if not retriever:
        raise HTTPException(
            status_code=400,
            detail="No database built for this session. Please upload documents first."
        )

    profile_summary = mastery.build_profile_summary(user_id)
    print(profile_summary)
    memory_context = user_memory.get_memory_as_system_context(user_id)
    chain = build_rag_chain(
        retriever,
        knowledge_profile_summary=profile_summary,
        user_memory_context=memory_context,
    )

    history_raw = store.get_session_messages(req.session_id)
    # Filter out quiz messages from LLM history
    text_history = [m for m in history_raw if m["role"] in ("user", "assistant")]

    # Token optimization: limit history to the last 2 QA pairs
    if len(text_history) > _HISTORY_TURNS:
        text_history = text_history[-_HISTORY_TURNS:]

    if not text_history:
        title = req.question[:25] + "..." if len(req.question) > 25 else req.question
        store.update_session_title(req.session_id, title)

    store.add_message(req.session_id, "user", req.question)

    formatted_history = []
    for h in text_history:
        if h["role"] == "user":
            formatted_history.append(HumanMessage(content=h["content"]))
        else:
            formatted_history.append(AIMessage(content=h["content"]))

    # Run CRAG before streaming so we capture sources metadata up front
    context_text, source_label, sources_metadata = build_crag_context(retriever, req.question)
    print(f"[Chat] Source label: {source_label} | Sources count: {len(sources_metadata)}")

    def generate():
        full_response = ""
        try:
            for chunk in chain.stream({
                "question": req.question,
                "context": context_text,
                "chat_history": formatted_history,
            }):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            # Emit sources metadata before [DONE] so the UI can attach the drawer
            yield f"data: {json.dumps({'sources': sources_metadata})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if full_response:
                store.add_message(req.session_id, "assistant", full_response, sources=sources_metadata)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
