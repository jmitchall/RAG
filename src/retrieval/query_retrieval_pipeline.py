from vectordatabases.vector_db_interface import VectorDBInterface
from vectordatabases.vector_db_factory import VectorDBFactory
from embedding.embedding_manager import EmbeddingManager

class QueryRetrievalPipeline:
    """
    A pipeline to ingest documents, chunk them, generate embeddings, and store in a vector database.
    """

    def __init__(self, vector_db_path:str = "./vector_db_qdrant",
                 embedding_model: str ="BAAI/bge-large-en-v1.5",
                 #use_embedding_server: bool = False,
                 #embedding_model_host: str = None,
                 safety_level: str = "recommended"
                 ):
        """
        Initialize the pipeline with configuration parameters.
        Args:
            vector_db_path: Path to store or load the vector database
            embedding_model: Model name for generating embeddings
            embedding_dim: Dimension of the embeddings
            use_embedding_server: Whether to use an external embedding server
            max_tokens: Maximum tokens for chunking documents
        """
        self.db_path = vector_db_path        
        self.embedding_model_name = embedding_model
        #self.embedding_model_host = embedding_model_host
        self.embedding_mgr: EmbeddingManager = None
        self.vector_db: VectorDBInterface = None
        self.embedding_dim: int = None
        self.safety_level=safety_level
    

    def load_vector_db_and_embedding_mgr(self):
        """
        Load or create the vector database and embedding manager.
        """
        self.vector_db = self.get_persisted_vector_db()
        self.embedding_dim = VectorDBFactory.get_actual_db_embedding_dim(self.vector_db)
        self.embedding_mgr = self.get_embedding_manager()
        # Step 3: Verify the dimension (optional double-check)
        verified_dim = self.embedding_mgr.get_actual_embedding_dimension()
        if verified_dim !=  self.embedding_dim:
            print(f"⚠️  Dimension mismatch! Vector DB embedding dimension: {self.embedding_dim} != Verified: {verified_dim}")
            self.embedding_dim = verified_dim  # Use the verified dimension

        return self.vector_db, self.embedding_mgr
    
    def get_persisted_vector_db(self, db_path: str=None) -> VectorDBInterface:
        """
        Retrieve a persisted vector database from the specified path.
        """
        if db_path is None:
            db_path= self.db_path
        self.vector_db = VectorDBFactory.get_vector_db(db_path)
        return self.vector_db
    

    def get_embedding_manager(self)-> EmbeddingManager:
        """
        Retrieve the embedding model based on configuration.
        """
        # Step 1: Choose optimal max_tokens based on chosen embedding  model
        self.max_tokens , suggested_embedding_dim = EmbeddingManager.estimate_optimal_max_token_actual_embeddings(
            embedding_model=self.embedding_model_name,
            batch_processing=False,  # Changed to False since we're forcing individual processing
            safety_level=self.safety_level
        )

        print(f"📏 Using max_tokens: {self.max_tokens}")

        # Step 2: Initialize the EmbeddingManager
        #if self.use_embedding_server:
            # Logic to connect to an external embedding server
            # __init__ in embedding_impl.py example parameter
            # model_name: str ="BAAI/bge-base-en-v1.5" , embedding_dim: int = 1024
            # , model_host: str ="http://localhost:8001", use_server: bool = True
            # , max_tokens: int = 400
        #    self.embedding_mgr = EmbeddingManager(
        #        model_name=self.embedding_model_name,
        #        model_host=self.embedding_model_host,
        #        max_tokens=self.max_tokens
        #    )
        #else:
            # Logic to load a local embedding model
        self.embedding_mgr = EmbeddingManager(
            model_name=self.embedding_model_name,
            max_tokens=self.max_tokens
        )
        print(f"recommended {self.embedding_model_name} dimension:{suggested_embedding_dim}")
        return self.embedding_mgr
    
       
    def retrieve(self, query_text: str, top_k: int =5) -> list:
        """
        Query the vector database for similar documents.
        Args:
            query_text: The input query string
            top_k: Number of top similar documents to retrieve
        Returns:
            List of similar documents
        """
        if not self.vector_db or not self.embedding_mgr:
            raise ValueError("Vector DB and Embedding Manager must be initialized. Call load_vector_db_and_embedding_mgr() first.")
        
        chunk_size = self.vector_db.get_max_document_length()
        print(f"Max document length in vector DB: {chunk_size} characters")
        query_text_input = query_text
        if len(query_text_input) > chunk_size:
            print(f"⚠️  Query text length ({len(query_text)}) exceeds max document length in vector DB ({chunk_size}). Truncating query.")
            query_text_input = query_text_input[:chunk_size]

        # Step 1: Generate embedding for the query
        query_embedding = self.embedding_mgr.get_embedding(query_text_input)
        
        # Step 2: Search the vector database
        results = self.vector_db.search(query_embedding, top_k=top_k)
        
        # Step 3: turn query_text into a Document and add it to results
        from langchain_core.documents import Document
        query_doc = Document(page_content=query_text_input, metadata={"source": "query"})
        results.insert(0, query_doc)  # Insert the query document at the beginning of the results

        return results
    


if __name__ == "__main__":    
    pipeline = QueryRetrievalPipeline(
        vector_db_path="/home/jmitchall/vllm-srv/vector_db_all_docs_faiss",
        embedding_model="BAAI/bge-large-en-v1.5",
        safety_level="max"  # Using safe mode
    )
    pipeline.load_vector_db_and_embedding_mgr()
    top_n_answers = 3
    if pipeline.vector_db and pipeline.embedding_mgr:
        # Example: Search for documents similar to a sample query
        sample_query = "What is Quintessence?"
        print(f"\n🔍 Searching for: '{sample_query}'")
 
        query_embedding = pipeline.embedding_mgr.get_embedding(sample_query)
        results = pipeline.vector_db.search(query_embedding, top_k=top_n_answers)

        context =""
        print(f"\n🏆 Top {top_n_answers} similar documents (using Qdrant):")
        for i, doc in enumerate(results):

            context += doc.page_content + "\n\n"
            #similarity = doc.metadata.get('similarity_score', 0)
            # print(f"\nDocument {i + 1} (similarity: {similarity:.3f}):")
            #print(f"Source: {doc.metadata.get('source', 'unknown')}")
            #print(f"Content: {doc.page_content}...")

        #if context consists of whitespace and carriage returns, print a warning
        if context.strip() == "":
            print("⚠️ No relevant documents found for the query.")
        else:
            print(f"\n📄 Context retrieved for query '{sample_query}':\n{context}...")  

        prompt= f""" Use the following 
Context: ----
{context}
----
to answer the following 
Question: ----
{sample_query}
----
"""
        print(f"\n🤖 Generated prompt for LLM:\n{prompt}")
    
    