import numpy as np
import torch
from langchain_core.documents import Document
from typing import List, Dict, Tuple

from data_ingestion_pipeline import DataIngestionPipeline

if __name__ == "__main__":
    vdb_type = "faiss"  # Change to "qdrant", "faiss", "chroma"
    # Example usage with automatic parameter selection
    db_pipeline = DataIngestionPipeline(
        db_type=vdb_type,
        directory_path="/home/jmitchall/vllm-srv/test_docs",
        embedding_model="BAAI/bge-large-en-v1.5",
        use_embedding_server=True,
        safety_level="max"  # Using safe mode
    )
    vect_db, embed_mgr = db_pipeline.generate_vector_db_and_embedding_mgr()

    top_n_answers = 5
    if vect_db and embed_mgr:
        # Example: Search for documents similar to a sample query
        sample_query = "What is Vitae?"
        print(f"\n🔍 Searching for: '{sample_query}'")
        query_embedding = embed_mgr.get_embedding(sample_query)
        results = vect_db.search(query_embedding, top_k=top_n_answers)

        print(f"\n🏆 Top {top_n_answers} similar documents (using {vdb_type.upper()}):")
        for i, doc in enumerate(results):
            similarity = doc.metadata.get('similarity_score', 0)
            print(f"\nDocument {i + 1} (similarity: {similarity:.3f}):")
            print(f"Source: {doc.metadata.get('source', 'unknown')}")
            print(f"Content: {doc.page_content}...")
    else:
        print("❌ Failed to create vector database")
