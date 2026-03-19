from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from src.core.config import GOOGLE_API_KEY, TOP_K

def build_rag_chain(retriever):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
    
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """Textbook Context (use if relevant):
{context}

Question: {question}
Answer:""")
    ])
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
         "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain