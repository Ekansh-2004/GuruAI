import os
import tempfile
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(file_data: List[Tuple[str, bytes]]) -> List[Document]:
    """Load and chunk multiple files."""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=45)
    
    for name, data in file_data:
        ext = os.path.splitext(name)[1].lower()
        loaders = {'.pdf': PyPDFLoader, '.txt': TextLoader, '.docx': Docx2txtLoader}
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        
        docs = loaders[ext](tmp_path).load()
        all_docs.extend(splitter.split_documents(docs))
        os.unlink(tmp_path)
    
    return all_docs