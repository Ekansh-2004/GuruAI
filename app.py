import streamlit as st
import os
from src.rag.loader import load_documents
from src.rag.embedder import create_vectorstore, load_existing_vectorstore
from src.rag.chain import build_rag_chain
import src.personalization.tracker as tracker
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Study AI", page_icon="📚", layout="wide")

if "quizzes" not in st.session_state:
    st.session_state.quizzes = {}
if "answered_questions" not in st.session_state:
    st.session_state.answered_questions = set()

# Load all historical sessions
all_sessions = tracker.load_all_sessions()

# Initialize the current active session
if "current_session_id" not in st.session_state:
    if not all_sessions:
        st.session_state.current_session_id = tracker.create_session("New Chat")
        all_sessions = tracker.load_all_sessions() 
    else:
        st.session_state.current_session_id = list(all_sessions.keys())[-1]

# Load the DB specifically for the current session ---
@st.cache_resource
def load_chain_for_session(session_id):
    existing_db = load_existing_vectorstore(session_id)
    if existing_db:
        retriever = existing_db.as_retriever(search_kwargs={"k": 4})
        return build_rag_chain(retriever)
    return None

# Update the active chain based on the session we are looking at
st.session_state.rag_chain = load_chain_for_session(st.session_state.current_session_id)

# --- Sidebar ---
with st.sidebar:
    st.header("💬 Chat Sessions")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = tracker.create_session("New Chat")
        st.session_state.current_session_id = new_id
        st.rerun()
        
    st.divider()
    st.subheader("Previous Chats")
    
    # Loop through sessions
    for sid, session_data in reversed(list(all_sessions.items())):
        btn_label = f"🟢 {session_data['title']}" if sid == st.session_state.current_session_id else f"📄 {session_data['title']}"
        if st.button(btn_label, key=sid, use_container_width=True):
            st.session_state.current_session_id = sid
            st.rerun()
            
    if st.button("🗑️ Delete Active Chat", use_container_width=True):
        tracker.delete_session(st.session_state.current_session_id)
        if "current_session_id" in st.session_state:
            del st.session_state.current_session_id
        st.rerun()

    st.divider()
    st.header("📚 Knowledge Base")
    db_path = f"faiss_index_db/{st.session_state.current_session_id}"
    if os.path.exists(db_path) and os.path.exists(os.path.join(db_path, "index.faiss")):
        st.success("✅ Active Database Loaded for this chat!")
    else:
        st.info("No database built for this chat yet.")

    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True, 
        key=f"uploader_{st.session_state.current_session_id}"
    )
    if st.button("Build Database"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                file_data = [(f.name, f.read()) for f in uploaded_files]
                docs = load_documents(file_data)
                
                # Pass the active session_id so it saves in the right folder!
                vectorstore = create_vectorstore(docs, st.session_state.current_session_id)
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                st.session_state.rag_chain = build_rag_chain(retriever)
                load_chain_for_session.clear()
                st.success("Database built and saved for this chat!")

    st.divider()
    st.header("📝 Generate Assessment")
    if st.button("Generate Quiz from Book", use_container_width=True):
        if not os.path.exists(f"faiss_index_db/{st.session_state.current_session_id}/index.faiss"):
            st.warning("Please build the Knowledge Base database first using the button above.")
        else:
            with st.spinner("Analyzing the uploaded documents and crafting 4 questions..."):
                from src.rag.quiz import generate_quiz_for_session_db
                quiz = generate_quiz_for_session_db(st.session_state.current_session_id)
                if quiz and quiz.get("questions"):
                    st.session_state.quizzes[st.session_state.current_session_id] = quiz
                    st.success("Quiz generated successfully!")
                else:
                    st.error("Failed to generate quiz. The document might be fully empty or API limits exceeded.")

    st.divider()
    st.header("🧠 Knowledge Profile")
    profile_data = tracker.get_performance_areas()
    if profile_data["strong"] or profile_data["average"] or profile_data["weak"]:
        with st.expander("Show Performance Areas", expanded=True):
            if profile_data["strong"]:
                st.markdown("🟢 **Strong Areas**")
                for topic, score, c, t in profile_data["strong"]:
                    st.write(f"- {topic} ({c}/{t})")
            if profile_data["average"]:
                st.markdown("🟡 **Average Areas**")
                for topic, score, c, t in profile_data["average"]:
                    st.write(f"- {topic} ({c}/{t})")
            if profile_data["weak"]:
                st.markdown("🔴 **Weak Areas**")
                for topic, score, c, t in profile_data["weak"]:
                    st.write(f"- {topic} ({c}/{t})")
    else:
        st.info("Complete quizzes to build your profile!")

# --- Main UI ---
st.title("Study AI Tutor 🤖")

if st.session_state.rag_chain is None:
    st.warning("Please upload documents and build the database to start.")

# Fetch ONLY the active session's history
active_history = tracker.get_session_messages(st.session_state.current_session_id)

# Display active chat history
for msg in active_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Display Quiz if available for this session
if st.session_state.current_session_id in st.session_state.quizzes:
    quiz_data = st.session_state.quizzes[st.session_state.current_session_id]
    st.divider()
    st.subheader("🎯 Assessment Quiz")
    
    for i, q in enumerate(quiz_data.get("questions", [])):
        st.markdown(f"**Q{i+1}: {q['question']}** *(Topic: {q.get('topic', 'General')})*")
        
        key = f"quiz_{st.session_state.current_session_id}_q{i}"
        
        selected_idx = st.radio(
            "Select an answer:",
            options=range(len(q["options"])),
            format_func=lambda x: q["options"][x],
            key=f"radio_{key}",
            index=None,
            label_visibility="collapsed"
        )
        
        if selected_idx is not None:
             # Lock answer inside our knowledge profile just once
             if key not in st.session_state.answered_questions:
                 is_correct = (selected_idx == q["correct_index"])
                 tracker.update_topic_performance(st.session_state.current_session_id, q.get("topic", "General"), is_correct)
                 st.session_state.answered_questions.add(key)
                 
             if selected_idx == q["correct_index"]:
                 st.success("✅ **Correct!**\n\n" + q["explanation"])
             else:
                 st.error(f"❌ **Incorrect.** The correct answer was: **{q['options'][q['correct_index']]}**")
                 st.info("💡 **Explanation:**\n\n" + q["explanation"])
    st.divider()

# Chat Input
if question := st.chat_input("Ask a question..."):
    # If this is the first message in the session, automatically rename the session title!
    if not active_history:
        short_title = question[:25] + "..." if len(question) > 25 else question
        tracker.update_session_title(st.session_state.current_session_id, short_title)

    with st.chat_message("user"):
        st.markdown(question)
        
    # Save user message to tracker
    tracker.add_message(st.session_state.current_session_id, "user", question)
    
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # Format ONLY the active history for LangChain
                formatted_history = []
                for h in active_history:
                    if h["role"] == "user":
                        formatted_history.append(HumanMessage(content=h["content"]))
                    else:
                        formatted_history.append(AIMessage(content=h["content"]))
                
                # Invoke chain
                answer = st.session_state.rag_chain.invoke({
                    "question": question,
                    "chat_history": formatted_history
                })
                
                st.markdown(answer)
                
        # Save assistant message to tracker
        tracker.add_message(st.session_state.current_session_id, "assistant", answer)
        st.rerun()