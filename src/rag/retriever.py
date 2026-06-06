from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.vectorstores import FAISS

def reciprocal_rank_fusion(dense_results: List[Document], sparse_results: List[Document], k: int = 60) -> List[Document]:
    """
    Applies Reciprocal Rank Fusion (RRF) to merge dense and sparse search results.
    """
    rrf_scores = {}
    doc_map = {}
    
    for rank, doc in enumerate(dense_results):
        doc_content = doc.page_content
        doc_map[doc_content] = doc
        r = rank + 1
        rrf_scores[doc_content] = rrf_scores.get(doc_content, 0.0) + (1.0 / (k + r))
        
    for rank, doc in enumerate(sparse_results):
        doc_content = doc.page_content
        doc_map[doc_content] = doc
        r = rank + 1
        rrf_scores[doc_content] = rrf_scores.get(doc_content, 0.0) + (1.0 / (k + r))
        
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[content] for content, _ in sorted_docs]

class HybridRetriever(BaseRetriever):
    vectorstore: FAISS
    tfidf_retriever: TFIDFRetriever
    top_n: int = 4
    
    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        print(f"\n--- [HybridRetriever] Processing Query: '{query}' ---")
        
        # 1. Retrieve from dense (FAISS)
        dense_docs = self.vectorstore.similarity_search(query, k=self.top_n * 2)
        print(f"[Dense (FAISS) Retrieved {len(dense_docs)} documents]:")
        for i, doc in enumerate(dense_docs):
            print(f"  {i+1}: {doc.page_content[:80]}...")
        
        # 2. Retrieve from sparse (TF-IDF)
        self.tfidf_retriever.k = self.top_n * 2
        sparse_docs = self.tfidf_retriever.invoke(query)
        print(f"[Sparse (TF-IDF) Retrieved {len(sparse_docs)} documents]:")
        for i, doc in enumerate(sparse_docs):
            print(f"  {i+1}: {doc.page_content[:80]}...")
        
        # 3. Merge using RRF
        merged_docs = reciprocal_rank_fusion(dense_docs, sparse_docs)
        final_docs = merged_docs[:self.top_n]
        
        print(f"[Final RRF Merged Top {len(final_docs)} documents]:")
        for i, doc in enumerate(final_docs):
            print(f"  {i+1}: {doc.page_content[:80]}...")
        print("---------------------------------------------------\n")
        
        return final_docs