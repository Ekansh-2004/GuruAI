

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.core.config import GOOGLE_API_KEY

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
    # Your custom persona prompt
    system_prompt = """You are Study AI, a helpful CS tutor that combines textbook accuracy with general knowledge.

RULES:
1. When textbook context RELEVANTLY answers the question → Use ONLY context + cite it
2. When context IRRELEVANT → Answer normally from your knowledge  
3. When context WRONG/CONFUSING → Trust your knowledge instead
4. ALWAYS be clear which source you're using
5. Teach like a patient TA - examples, simple language, code when helpful

Format:
- [Textbook] for document answers  
- [General knowledge] for other cases"""

    # We build the prompt with the system rules, the chat history, and the user's current question
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Textbook Context (use if relevant):\n{context}\n\nQuestion: {question}\nAnswer:")
    ])
    
    # We explicitly route the dictionary values so LangChain doesn't crash!
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
