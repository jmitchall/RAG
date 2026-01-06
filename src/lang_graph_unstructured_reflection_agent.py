from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from abc import ABC
from typing import List,  Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings
import os
import logging
import traceback
from inference.vllm_srv.minstral_langchain import get_langchain_vllm_mistral_quantized, convert_chat_prompt_to_minstral_prompt_value

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReflectionAgentState(TypedDict):
            messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], "add_messages"]
            agent_instance: "ReflectionAgent" 
            continue_refining: bool = True
            question: str = ""
            context:str = ""

class ReflectionAgent(ABC):

    def __init__(self, brain, embedding_model = "BAAI/bge-large-en-v1.5", root_path = "/home/jmitchall/vllm-srv" ,  similarity_threshold:float=0.65, **kwargs):
        self.llm = brain
        self.similarity_threshold =similarity_threshold
        # Create Generation Prompt for answer
        self.generation_chain = self.get_generation_chain()
        # Create Reflection Prompt for Answer Evaluation
        self.reflection_chain= self.get_reflection_chain()
        self.count = 0
        self.embeddings = HuggingFaceOfflineEmbeddings(model_name=embedding_model)
        self.collection_name = kwargs.get("collection_name")
        self.DATABASE_TYPE = kwargs.get("DATABASE_TYPE")
        self.root_path = root_path

    def get_initial_state(self, question: str = "") -> ReflectionAgentState:
        context = ""
        if question:
                retriever, _ = self.get_retriever()
                if retriever:
                    retrieved_docs = retriever.invoke(question)
                    context = self.format_list_documents_as_string(results=retrieved_docs)
                    logger.info(f"Retrieved {len(retrieved_docs)} documents for context")
        
        return ReflectionAgentState(
            agent_instance=self,
            question=question,
            context=context,
            messages=[HumanMessage(content=question)] if question else []
        )
    
    def get_retriever_and_vector_stores(self, vdb_type:str, vector_db_persisted_path:str, 
                                    collection_ref:str, retriever_embeddings) :
        from vectordatabases.qdrant_vector_db_commands import QdrantClientSmartPointer, quadrant_does_collection_exist ,get_quadrant_client, get_qdrant_retriever
        from vectordatabases.fais_vector_db_commands import create_faiss_vectorstore, get_faiss_retriever
        from vectordatabases.chroma_vector_db_commands import get_chroma_vectorstore, get_chroma_retriever
        langchain_retriever = None
        qdrant_client: QdrantClientSmartPointer = None
        # test persisted vector store loading and retriever creation
        print(
            f"\n🔍 Testing loading of persisted vector store for collection '{collection_ref}' from path: {vector_db_persisted_path} ...")
        match vdb_type:
            case "qdrant":
                # Reconnect to the persisted Qdrant database
                qdrant_client: QdrantClientSmartPointer = get_quadrant_client(vector_db_persisted_path)
                if quadrant_does_collection_exist(qdrant_client, collection_ref):
                    langchain_retriever = get_qdrant_retriever(qdrant_client, collection_ref, embeddings=retriever_embeddings, k=5)
                    print(f"✅ Created Qdrant retriever wrapper for collection '{collection_ref}'")
                else:
                    print(f"⚠️  Collection '{collection_ref}' does not exist yet. Skipping retrieval test.")
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
    

    def get_retriever (self):
         if self.DATABASE_TYPE and self.collection_name and self.root_path:
            self.DATABASE_TYPE=  self.DATABASE_TYPE.lower()
            vector_db_persisted_path =f"{self.root_path}/langchain_{self.collection_name}_{self.DATABASE_TYPE}"
            #check if directory exists
            if os.listdir(vector_db_persisted_path):
                return self. get_retriever_and_vector_stores(self.DATABASE_TYPE, vector_db_persisted_path , self.collection_name, self.embeddings)
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
                print(f" Skipping :{ doc.metadata.get("source")}, page:{doc.metadata.get("page")} based on Threshold {self.similarity_threshold } ")
                continue
            context += doc.page_content + "\n\n"

        return context

    def get_generation_chain(self): 
              
        answer_generation_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""You are an expert Dungeuns and dragon Game Master. 
    You recall all the rules in the following sources:
    1) Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond
    2) Monster Manual - Dungeons & Dragons - Sources - D&D Beyond
    3) Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond
    4) Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond

    Your TASK:
    Is to generate the best possible RESPONSE or ANSWER to the 
    USER'S QUERY:
    ====================
    {question}
    
    by first taking into consideration the CONTEXT 
                                                          
    CONTEXT START:=======================================
    {context} 
    ===================:CONTEXT END
    and siting what sources were used in providing the RESPONSE or ANSWER to a USER'S QUERY.
                                               
    If the user provides feedback or critique on the RESPONSE or ANSWER to a USER'S QUERY, 
    RESPOND with a refined version of your previous attempts, improving 
                                                        
    ANSWER RUBRICS:
    a) clarity
    b) succinctness
    c) readability 

    as needed.
        """),
                # This is used to inject the actual content or message that the post will be based on.  
                # The placeholder will be populated with the user’s request at runtime.
                MessagesPlaceholder(variable_name="messages")  
            ])
        return answer_generation_prompt | self.llm
    
    def get_reflection_chain(self):
        generated_reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""You are an Dungeons and Dragon Player. You recall all the rules in the following sources:
    1) Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond
    2) Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond
    
    USER'S QUERY:
    ====================
    {question}
    ====================
                                                      
    Your TASK:
    1. Assess the RESPONSE or ANSWER to a USERS QUERY 
    2. Evaluate the clarity, succinctness, and readability of the RESPONSE or ANSWER to a USERS QUERY
    3. Consider RESPONSE's relevance to the USERS QUERY
    4. Analyze RESPONSE's effectiveness in ANSWERing to a USERS QUERY credibility by considering the source it originated from.
    5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
                                                    
    Provide a detailed critique that includes:
            - A brief explanation of the RESPONSE's or ANSWER's strengths and weaknesses.
            - Specific areas that could be improved.
            - Actionable suggestions for enhancing credibility, clarity, succinctness, and readability.

    Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful,                                                   
    constructive, and practical.
                                                      
    Encapulate Every Critique with
    CRITIQUE START: ======================================
    [Place Critique Here]
    ==============================: CRITIQUE END                                  
    
    """),
            # This is used to inject the actual content or message that the post will be based on.  
            # The placeholder will be populated with the user’s request at runtime.
            MessagesPlaceholder(variable_name="messages")  
        ])
        return generated_reflection_prompt | self.llm


def agent_generation_node( state: ReflectionAgentState) -> ReflectionAgentState:
    """
    In LangGraph Agent Flows are represented via Agent WorkFolws where 
    States accumulate and change base on their travels from Node to Node

    This Node Cooresponds to a Reflection Agent's Generated Reponse

    Args:
        state: 
    """
    try:
        agent_components = state["agent_instance"]
        question = state.get("question")
        context = state.get("context")
        ai_messages = state.get("messages")
        
        logger.info(f"Generation node processing question: {question[:50]}...")
        
        # Create Generation Prompt with context and question
        logger.info("Invoking generation chain...")
        last_message =  [ ai_messages[-1] ]
        generated_response_and_answer = agent_components.generation_chain.invoke({
                "question": question,
                "context": context,
                "messages": last_message
            }
        )
        logger.info("Generation successful")
        
    except Exception as e:
        logger.error(f"❌ Generation node failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return error message in state
        error_msg = f"Generation failed: {str(e)}"
        return {**state, "messages": [AIMessage(content=error_msg)]}
    
    if generated_response_and_answer and generated_response_and_answer.strip():
        answer_provided = True
    else:
        answer_provided = False
    if not ai_messages:
        ai_messages = [AIMessage(content=generated_response_and_answer)]
    else:
        ai_messages.append(AIMessage(content=generated_response_and_answer))
    return {**state, "continue_refining": answer_provided, "messages": ai_messages }
        

def agent_reflection_node( state: ReflectionAgentState) -> ReflectionAgentState:
    """
    In LangGraph Agent Flows are represented via Agent WorkFolws where 
    States accumulate and change base on their travels from Node to Node

    This Node Cooresponds to a Reflection Agent's Generated Reponse

    Args:
        state: 
    """
    try:
        agent_components = state["agent_instance"]
        ai_messages = state.get("messages")
        question = state.get("question")
        logger.info("Invoking reflection chain...")
        last_response =  [ ai_messages[-1] ]
        # Create Reflection on current meesages
        response_reflection = agent_components.reflection_chain.invoke(
            {
                "question": question,
                "messages": last_response
            }
        )
        logger.info("Reflection successful")
        
    except Exception as e:
        logger.error(f"❌ Reflection node failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return with no refinement needed on error
        return {**state, "continue_refining": False}
    ai_messages.append(HumanMessage(content=response_reflection))
    if response_reflection and response_reflection.strip():
        critique_provided = True
    else:
        critique_provided = False
    return {**state, "continue_refining": critique_provided , "messages": ai_messages , "context": response_reflection} # pretend human feedback


def should_agent_reflect(state: ReflectionAgentState):  
    agent_components = state["agent_instance"]
    if agent_components.count <2 and state.get("continue_refining", True):
        agent_components.count +=1
    else:
        state["continue_refining"] = False
        return END
    return "agent_reflection_node"

def should_agent_retry(state: ReflectionAgentState):
    if  state.get("continue_refining", True):
        return "agent_generation_node"
    return END

if __name__ == "__main__":
    from inference.vllm_srv.cleaner import cleanup_vllm_engine , force_gpu_memory_cleanup   
    force_gpu_memory_cleanup()
    llm = None
    try:
        logger.info("🚀 Starting Reflection Agent...")
        
        # Create Agent Brain
        logger.info("Initializing vLLM engine...")
        llm = get_langchain_vllm_mistral_quantized(download_dir="./models")
        logger.info("✅ vLLM engine initialized successfully")
        
        reflection_agent = ReflectionAgent(
            brain=llm,embedding_model = "BAAI/bge-large-en-v1.5", 
            root_path = "/home/jmitchall/vllm-srv" ,
            collection_name= "dnd_player",
            DATABASE_TYPE = "qdrant"
        )
        logger.info("✅ Reflection agent created")

        # Initialize a StateGraph
        graph = StateGraph(ReflectionAgentState)
        
        graph.add_node("agent_generation_node",agent_generation_node)
        graph.add_node("agent_reflection_node",agent_reflection_node)
        graph.add_conditional_edges( "agent_reflection_node", should_agent_retry )   
        graph.add_conditional_edges("agent_generation_node",should_agent_reflect)
        graph.set_entry_point("agent_generation_node")
        workflow_graph = graph.compile()
        logger.info("✅ Workflow graph compiled")
        
        query = "What is a Rogue"
        logger.info(f"Processing query: {query}")

        # Invoke workflow with initial state
        result = workflow_graph.invoke(reflection_agent.get_initial_state (question=query))
        logger.info("✅ Workflow completed successfully")
        logger.info(f"Final result: {result["messages"][-1].content}")
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Try to get GPU info if available
        try:
            import subprocess
            gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv'], 
                                              stderr=subprocess.STDOUT).decode()
            logger.error(f"GPU Status:\n{gpu_info}")
        except:
            logger.error("Could not retrieve GPU status")
        
        raise
    
    finally:
        # Suppress vLLM's benign exit message
        if llm is not None:
            logger.info("✅ Workflow complete, exiting...")
            import sys
            import os
            cleanup_vllm_engine(llm)
            # Suppress the benign "Engine core died" message by redirecting stderr
            # This message appears during normal shutdown and is not an error
            sys.stderr = open(os.devnull, 'w')
            # Engine will cleanup automatically when Python exits

    


