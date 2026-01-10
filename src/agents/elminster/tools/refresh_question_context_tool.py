# Create a tool wrapper that doesn't include self in the schema
from embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings
from vectordatabases.qdrant_vector_db_commands import QdrantClientSmartPointer, quadrant_does_collection_exist ,get_quadrant_client, get_qdrant_retriever
from vectordatabases.fais_vector_db_commands import create_faiss_vectorstore, get_faiss_retriever
from vectordatabases.chroma_vector_db_commands import get_chroma_vectorstore, get_chroma_retriever
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Optional
from agents.elminster.knowledge import collection_names
import os
from refection_logger import logger

class RefreshQuestionContextInput(BaseModel):
    question: str = Field(description="The question to search for context")
    sources: List[str] = Field(description="""Sources ranging from 
                "Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond" ,
                "Vampire-the-Masquerade",
                "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond",
                "Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond",
                "Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond" """)
    db_type: str = Field(description="""Vector database types types are "qdrant","faiss", or "qdrant" """)
    root_path: str = Field(description="Root path for the vector database")
    critique: Optional[str] = Field(default=None, description="A critique of the original question based on an evalation of the previous answer. Include this when refining a previous answer.")
    improved_question: Optional[str] = Field(default=None, description="An improved version of the original question based on critique. Include this when critique is provided.")

class RefreshQuestionContextTool(BaseTool):
    name: str = "refresh_question_context"
    description: str = """ Takes in a {question} and converts the following context {sources}
A) Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond
B) Monster Manual - Dungeons & Dragons - Sources - D&D Beyond
C) Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond
D) Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond
E) Vampire-the-Masquerade
and one of the vector database types {db_type} in the list of choices "qdrant","faiss", or "qdrant"
in order to generate a {context} associated to the {question}

        Args:
            sources: Sources ranging from 
                "Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond" ,
                "Vampire-the-Masquerade",
                "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond",
                "Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond",
                "Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond"
            db_type: Vector database types types are "qdrant","faiss", or "qdrant"
            root_path: vector db root path 
            critique: A critique of the original question based on an evalation of the previous answer.
            improved_question: An improved version of the original question based on critique.
        Returns: a String indicating the USER'S QUERY along with the CONTEXT that should be considered. if "critique" is defined  "improved_question" will be populated with a question improved based on the critique.
else
    do not pass the "improved_question" argument to the tool.
        """
    args_schema: Type[BaseModel] = RefreshQuestionContextInput
    similarity_threshold: float = 0.7
    
   
    def get_retriever_and_vector_stores( self,  vdb_type:str, vector_db_persisted_path:str, 
                                    collection_ref:str , retriever_embeddings) :
        langchain_retriever = None
        qdrant_client: QdrantClientSmartPointer = None
        # test persisted vector store loading and retriever creation
        logger.info(
            f"\n🔍 Testing loading of persisted vector store for collection '{collection_ref}' from path: {vector_db_persisted_path} ...")
        match vdb_type:
            case "qdrant":
                # Reconnect to the persisted Qdrant database
                qdrant_client: QdrantClientSmartPointer = get_quadrant_client(vector_db_persisted_path)
                if quadrant_does_collection_exist(qdrant_client, collection_ref):
                    langchain_retriever = get_qdrant_retriever(qdrant_client, collection_ref, embeddings=retriever_embeddings, k=5)
                    logger.info(f"✅ Created Qdrant retriever wrapper for collection '{collection_ref}'")
                else:
                    logger.info(f"⚠️  Collection '{collection_ref}' does not exist yet. Skipping retrieval test.")
                    langchain_retriever = None
            case "faiss":
                loaded_vectorstore_wrapper = create_faiss_vectorstore(
                    vector_db_persisted_path,
                    retriever_embeddings
                )
                langchain_retriever = get_faiss_retriever(loaded_vectorstore_wrapper, k=5)
            case "chroma": 
                loaded_vectorstore = get_chroma_vectorstore(collection_ref, vector_db_persisted_path, 
                                                            retriever_embeddings)
                langchain_retriever =  get_chroma_retriever(loaded_vectorstore, k=5)
            case _:
                raise ValueError(
                    f"Unsupported DATABASE_TYPE: {vdb_type}. Supported types are 'qdrant', 'faiss', 'chroma'.")
        return langchain_retriever, qdrant_client


    def get_retriever(self , collection_name:str, dbtype:str , root_path:str  ):
         if dbtype and collection_name and root_path:
            dbtype=  dbtype.lower()
            vector_db_persisted_path =f"{root_path}/db/langchain_{collection_name}_{dbtype}"
            #check if directory exists
            embeddings: HuggingFaceOfflineEmbeddings = HuggingFaceOfflineEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            if os.listdir(vector_db_persisted_path):
                return self.get_retriever_and_vector_stores(dbtype, vector_db_persisted_path , collection_name, embeddings)
            else:
                raise ValueError(f" No Directory {vector_db_persisted_path}")



    def format_list_documents_as_string(self, **kwargs) -> str:
        """Format a list of Documents into a human-readable string with metadata.
        
        Args:
            results: List of Document objects with page_content and metadata
            similarity_threshold: Minimum similarity score to include a document (default: 0.8)
            
        Returns:
            Formatted string with each document's content, source, similarity score, and rank
        """
        results = kwargs.get('results', [])
        if not results:
            return ''
        context =""
        for i, doc in enumerate(results, 1):
            
            similarity = doc.metadata.get("similarity_score", .7)
            if similarity < self.similarity_threshold:
                continue
            logger.info(f" Adding Context :{ doc.metadata.get("source")}, page:{doc.metadata.get("page")} based on Threshold {self.similarity_threshold } ")
            context += doc.page_content + f"\n\n----------end source {i}\n"

        return context
    

    def _run(self, question: str, sources: List[str], db_type: str, root_path: str, critique: Optional[str] = None, improved_question: Optional[str] = None) -> str:
        context = ""
        available_collections = collection_names.keys()
        logger.info(f"Checking sources {sources} against {available_collections}")
        if question:
            if critique and critique.strip():
                logger.info(f"Using critique to refresh context: {critique} using sources {sources}")
                question = f"{question}"
            all_retrieved_docs = []
            for src in sources:
                if src in available_collections:
                    collection_name= collection_names[src]
                    retriever, temp_ptr = self.get_retriever(collection_name =collection_name , dbtype=db_type, root_path=root_path)
                    if retriever:
                        retrieved_docs = retriever.invoke(question)
                        all_retrieved_docs.extend(retrieved_docs)
                        logger.info(f"Retrieved {len(retrieved_docs)} documents for context")
            else:
                logger.info( f"Could not vector db source {src} in \nknowledge map:\n{collection_names}  ")

            context = self.format_list_documents_as_string(results=all_retrieved_docs)
        return {"question":question, "context": context, "critique": critique or ""}
    
