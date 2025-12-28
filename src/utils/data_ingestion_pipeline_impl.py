import os
from ingestion.data_ingestion_pipeline import DataIngestionPipeline

if __name__ == "__main__":
    vdb_type = "qdrant"  # Change to "qdrant", "faiss", "chroma"
    # Example usage with automatic parameter selection

    document_collections = [
        "vtm",
        "dnd",
        "all"
    ]
    root_data_path = "/home/jmitchall/vllm-srv/data"
    # copy all documents from /home/jmitchall/vllm-srv/data/vtm and /home/jmitchall/vllm-srv/data/dnd into /home/jmitchall/vllm-srv/data/all before running this
    for collection_name in document_collections:
        pattern = os.path.join(root_data_path, collection_name, "*")
        files = [f for f in os.listdir(os.path.join(root_data_path, collection_name)) if
                 os.path.isfile(os.path.join(root_data_path, collection_name, f))]
        print(f"Collection '{collection_name}' has {len(files)} files: {files}")
        # copy files into 'all' collection
        if collection_name != "all":
            all_collection_path = os.path.join(root_data_path, "all")
            if not os.path.exists(all_collection_path):
                os.makedirs(all_collection_path)
            for f in files:
                src = os.path.join(root_data_path, collection_name, f)
                dst = os.path.join(all_collection_path, f)
                if not os.path.exists(dst):
                    os.symlink(src, dst)  # Create a symlink to avoid duplication

    for collection_name in document_collections:
        db_pipeline = DataIngestionPipeline(
            db_type=vdb_type,
            directory_path=f"{root_data_path}/{collection_name}",
            embedding_model="BAAI/bge-large-en-v1.5",
            safety_level="max"  # Using safe mode
        )
        print(f"\n=== Processing collection: {collection_name} ===")
        # Generate vector database and embedding manager for the specified collection
        vect_db, embed_mgr = db_pipeline.generate_vector_db_and_embedding_mgr(collection_name=collection_name + "_docs")
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
