from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langgraph.graph import END, StateGraph
from abc import ABC
from typing import List,  Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from pydantic import BaseModel, Field
from embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings
import os
from inference.vllm_srv.cleaner import check_gpu_memory_status ,force_gpu_memory_cleanup
import traceback
from langchain.output_parsers import PydanticOutputParser
import json
import re
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from inference.vllm_srv.minstral_langchain import create_vllm_chat_model

# Configure logging
import logging
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

class QuestionResponseSchema(BaseModel):
    answer: str = Field( description="expert's RESPONSE or ANSWER the USER'S QUERY" )
    question: str = Field( description="the original question asked" )
    source: str = Field( description="the original question asked" ) 
    context_summary: str = Field( description="less than 500 character summary of context used to ANSWER the USER'S QUERY" )
    
    @classmethod
    def has_required_fields(cls, data: dict) -> bool:
        """Check if dict has all required fields for this model."""
        required_fields = set(cls.model_fields.keys())
        dict_fields = set(data.keys())
        return required_fields.issubset(dict_fields)
    
    @classmethod
    def validate_dict_safe(cls, data: dict) -> tuple[bool, str]:
        """Safely validate dict and return success status with error message."""
        try:
            cls.model_validate(data)
            return True, "Valid"
        except Exception as e:
            return False, str(e)

class CritiqueOfAnswerSchema(BaseModel):
    critique: str = Field( description="the evaluation of the RESPONSE or ANSWER to a USER'S QUERY  based on it's clarity, succinctness, and readability" )
    clarity: float = Field( description="Single float rating from 0.0 - 1.0 where 0.0 means the RESPONSE is incoherrent and hard to understand where as 1.0 means the explaination is really easy to understand")
    succinct: float = Field( description="Single float rating from 0.0 - 1.0 where 0.0 means the amonut of text used to answer is very large where as 1.0 means the explaination is as consise as possible without sacrificing ability to understand")
    readabilty: float = Field( description="Single float rating from 0.0 - 1.0 where 0.0 means the explaination requires a graduate degress to truly comprehend where as 1.0 means reading level is that of a 5th grader  ")
    revision_needed: bool = Field (description="Single boolean returning True if the answr needs to be improved based on the evaluation, and False if it is sufficently clear, succinct, and readable")
    response: QuestionResponseSchema = Field( description="the original RESPONSE or ANSWER to USER'S QUERY being critiqued" )

    @classmethod
    def has_required_fields(cls, data: dict) -> bool:
        """Check if dict has all required fields for this model."""
        required_fields = set(cls.model_fields.keys())
        dict_fields = set(data.keys())
        return required_fields.issubset(dict_fields)
    
    @classmethod
    def validate_dict_safe(cls, data: dict) -> tuple[bool, str]:
        """Safely validate dict and return success status with error message."""
        try:
            cls.model_validate(data)
            return True, "Valid"
        except Exception as e:
            return False, str(e)

class VectorDBContextResponse(BaseModel):
      context: str = Field( description="Context of the USER'S QUERY" )
      

class ReflectionAgent(ABC):
    # Class-level embeddings - shared across all instances
    embeddings = None
    similarity_threshold =0.7

    def __init__(self, brain, root_path = "/home/jmitchall/vllm-srv" ,  similarity_threshold:float=0.65, **kwargs):
        self.llm =brain
        ReflectionAgent.similarity_threshold =similarity_threshold
        ReflectionAgent.embeddings =HuggingFaceOfflineEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        # Create Generation Prompt for answer
        self.generation_chain = self.get_generation_chain()
        # Create Reflection Prompt for Answer Evaluation
        self.reflection_chain= self.get_reflection_chain()
        self.count = 0
        self.DATABASE_TYPE = kwargs.get("DATABASE_TYPE")
        self.root_path = root_path
        self.question ="" 
        self.context = ""

    @staticmethod
    def get_response_llm_results( json_str :str ):

        answer_generation_parser = PydanticOutputParser(pydantic_object=QuestionResponseSchema)
        try:
            return_value = answer_generation_parser.parse(json_str)
            return return_value
        except Exception as answer_generation_excpt:
            print(f"answer_generation_excpt : {answer_generation_excpt}")

        
        # Try direct JSON parsing
        try:
            return_value = json.loads(json_str)
        except Exception as json_parse_excpt:
            print(f"json.loads error: {json_parse_excpt}")
            return json_str  # Return original string if all parsing fails

        # Validate against schemas
        is_valid, message = QuestionResponseSchema.validate_dict_safe(return_value)
        if is_valid:
            # Create the model instance
            return QuestionResponseSchema.model_validate(return_value) 
        
        # If validation fails, return the raw data
        print(f"Validation failed for both schemas. Returning raw data: {return_value}")
        return return_value  

    @staticmethod
    def get_critique_llm_results( json_str :str ):

        reflection_critique_parser = PydanticOutputParser(pydantic_object=CritiqueOfAnswerSchema)
        try:
            return_value= reflection_critique_parser.parse(json_str)
            return return_value
        except Exception as reflection_critique_excpt:
            print(f"reflection_critique_parser : {reflection_critique_excpt}")

        # Try direct JSON parsing
        try:
            return_value = json.loads(json_str)
        except Exception as json_parse_excpt:
            print(f"json.loads error: {json_parse_excpt}")
            return json_str  # Return original string if all parsing fails

        is_valid, message = CritiqueOfAnswerSchema.validate_dict_safe(return_value)
        if is_valid:
            # Create the model instance
            return CritiqueOfAnswerSchema.model_validate(return_value)  
        
        # If validation fails, return the raw data
        print(f"Validation failed for both schemas. Returning raw data: {return_value}")
        return return_value  

    @staticmethod
    def extract_json_response_output(raw_output: str):
        """
        Clean and extract JSON from LLM output that may contain extra text.
        """
        # Find text between ```json and ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1).strip()
            try:
                return_value = ReflectionAgent.get_response_llm_results(json_str)
                return return_value       
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found in code block: {e}")

        # Find text between 1st { and last }
        first_brace = raw_output.find('{')
        last_brace = raw_output.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_output[first_brace:last_brace + 1]

            try:
                return_value = ReflectionAgent.get_response_llm_results(json_str)
                return return_value 
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found between braces: {e}")
        
        return raw_output

    @staticmethod
    def extract_json_reflection_output(raw_output: str):
        """
        Clean and extract JSON from LLM output that may contain extra text.
        """
        # Find text between ```json and ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        
        if match:
            json_str = match.group(1).strip()
            try:
                return_value = ReflectionAgent.get_critique_llm_results(json_str)
                return return_value       
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found in code block: {e}")

        # Find text between 1st { and last }
        first_brace = raw_output.find('{')
        last_brace = raw_output.rfind('}')
        
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_output[first_brace:last_brace + 1]

            try:
                return_value = ReflectionAgent.get_critique_llm_results(json_str)
                return return_value 
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found between braces: {e}")
        
        return raw_output
          

    def get_initial_state(self, question: str = "") -> ReflectionAgentState:
        return ReflectionAgentState(
            agent_instance=self,
            question=question,
            messages=[HumanMessage(content=question)] if question else []
        )
    
    @staticmethod
    def get_retriever_and_vector_stores( vdb_type:str, vector_db_persisted_path:str, 
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
    
    @staticmethod
    def get_retriever( collection_name:str, dbtype:str , root_path:str  ):
         if dbtype and collection_name and root_path:
            dbtype=  dbtype.lower()
            vector_db_persisted_path =f"{root_path}/langchain_{collection_name}_{dbtype}"
            #check if directory exists
            if os.listdir(vector_db_persisted_path):
                return ReflectionAgent.get_retriever_and_vector_stores(dbtype, vector_db_persisted_path , collection_name, ReflectionAgent.embeddings)
            else:
                raise ValueError(f" No Directory {vector_db_persisted_path}")

    @staticmethod
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
            if similarity < ReflectionAgent.similarity_threshold:
                print(f" Skipping :{ doc.metadata.get("source")}, page:{doc.metadata.get("page")} based on Threshold {ReflectionAgent.similarity_threshold } ")
                continue
            context += doc.page_content + f"\n\n----------end source {i}"

        return context


    @tool
    def refresh_question_context ( question, sources:List[str], db_type:str , root_path:str):
        """ Takes in a {question} and converts the following context {sources}
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
        Returns: a String indicating the USER'S QUERY along with the CONTEXT that should be considered 
        """
        collection_names = {
            "Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond" : "dnd_dm",
            "Vampire-the-Masquerade":"vtm",
            "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond": "dnd_mm",
            "Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond": "dnd_raven",
            "Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond": "dnd_player",
        }

        if question:
            context = ""
            all_retrieved_docs = []
            for src in sources:
                collection_name= collection_names[src]
                retriever, temp_ptr = ReflectionAgent.get_retriever(collection_name =collection_name , dbtype=db_type, root_path=root_path)
                if retriever:
                    retrieved_docs = retriever.invoke(question)
                    all_retrieved_docs.extend(retrieved_docs)
                    logger.info(f"Retrieved {len(retrieved_docs)} documents for context")

            context = ReflectionAgent.format_list_documents_as_string(results=all_retrieved_docs)
        return context
        


    def execute_tool_and_populate_context(self, inputs):
            """Execute tool through LLM and extract result to populate context.
            will receive in

            Args:
                inputs: {
                        "question": question,
                        "messages": last_message,
                        "sources":""
                        }
            """
            question = inputs.get('question', '')
            db_type = inputs.get('db_type', '')
            messages = inputs.get('messages', [])
            sources = inputs.get('sources', [])
            
            # Create a prompt that encourages the LLM to use the tool
            tool_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "You need to get context for this question: {question}. "
                    "Use the refresh_question_context tool with appropriate root_path: {root_path},"
                    "db_type: {db_type}, and sources: {sources}"
                ),
                MessagesPlaceholder(variable_name="messages")
            ])
            
            # Use manual tool execution instead of bind_tools to prevent memory issues
            # tools = [self.refresh_question_context]
            # llm_with_tools = self.llm.bind_tools(tools)  # This creates new instances
            
            # Get LLM response without tool binding to prevent memory leaks
            manual_chain = tool_prompt | self.llm
            
            # Execute tool directly instead of through LLM to avoid memory issues
            logger.info("Executing context retrieval manually to prevent memory leaks")
            self.context = ""
            try:
                # Direct tool execution with provided parameters
                tool_args = {
                    'question': question,
                    'sources': sources,
                    'db_type': db_type,
                    'root_path': self.root_path
                }
                self.context = self.refresh_question_context.invoke(tool_args)
                logger.info(f"Successfully retrieved context: {len(self.context)} characters")
            except Exception as tool_error:
                logger.warning(f"Tool execution failed: {tool_error}")
                self.context = "Context retrieval failed due to memory constraints."
            
            # Return inputs with populated context
            return {
                'question': question,
                'context': self.context,
                'messages': messages,
            }
      
    def get_generation_chain(self): 
        answer_generation_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("""You are an expert Dungeuns and Dragon Game Master. 
    
    Your TASK:
    Is to generate the best possible RESPONSE or ANSWER to
    USER'S QUERY: 
    ============
    {question}
    ============
                                                          
    by first taking into consideration the following 
    ============
    {context}
    ============
    .  
    Then deliver an expert's RESPONSE or ANSWER the USER'S QUERY, 
    citing of what sources were used to provide the RESPONSE, and 
    summarize the context used to ANSWER the USER'S QUERY.
                                               
    If the user provides feedback or CRITIQUE on the RESPONSE or ANSWER to a USER'S QUERY, 
    respond with a refined version of your previous attempts, 
    improving the ANSWER using the following  
    
    RUBRICS:
    a) clarity
    b) succinctness
    c) readability 

    as needed. USe the follwing Format for Both types of responses:

    IMPORTANT: Return only a JSON object with actual data values, NOT a schema definition. 
    Do not include "properties" or "required" fields - just return the actual answer data.
    
    Example of CORRECT format:
    {{
        "answer": "Your detailed answer here",
        "question": "The original question", 
        "source": "D&D Player's Handbook",
        "context_summary": "Summary of context used"
    }}

    {format_instructions}
        """),
                # This is used to inject the actual content or message that the post will be based on.  
                # The placeholder will be populated with the user's request at runtime.
                MessagesPlaceholder(variable_name="messages")  
            ])
        answer_generation_parser = PydanticOutputParser(pydantic_object=QuestionResponseSchema)
        
        # Add format instructions to the prompt
        answer_generation_prompt = answer_generation_prompt.partial(
            format_instructions=answer_generation_parser.get_format_instructions()
        )
        tool_executor = RunnableLambda(self.execute_tool_and_populate_context)
        return tool_executor |answer_generation_prompt | self.llm 
    
    def get_reflection_chain(self):
        generated_reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""You are an Dungeons and Dragon Player. You recall all the rules in the following sources:
    {sources}
    
    USER'S QUERY:
    ====================
    {question}
    ====================
                                                      
    Your TASK:
    1. Assess the RESPONSE or ANSWER to a USERS QUERY 
    2. Analyze RESPONSE's effectiveness and relevance in ANSWERing to a USERS QUERY credibility by considering the source it originated from.
    3. Evaluate the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
    4. Determine if the RESPONSE or ANSWER needs revision based on your evaluation. 
    5. Assign clarity ranges from 0.0 - 1.0 where 0.0 means the RESPONSE is incoherrent and 1.0 means the explaination is really easy to understand
    6. Assign succinct ranges from 0.0 - 1.0 where 0.0 means text length is too large and 1.0 means the explaination is as concise
    7. Assign readabilty ranges from 0.0 - 1.0 where 0.0 means understanding requires a graduate degress and 1.0 means requires at least a 5th grade reading level 
    8. Assign "revision_needed": true if there is no citation or source indicated in Response's answer
    9. Assign "revision_needed": true if any  "clarity", "succinct", or "readabilty" RUBRIC is less than or equal to 0.5
                                                    
    Provide a detailed critique that includes:
            - A brief explanation of the RESPONSE's or ANSWER's strengths and weaknesses.
            - Specific areas that could be improved.
            - Actionable suggestions for enhancing credibility, clarity, succinctness, and readability.

    Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful,                                                   
    constructive, and practical.
                                                      
    IMPORTANT: Return only a JSON object with actual data values, NOT a schema definition. 
    Do not include "properties" or "required" fields - just return the actual critique data.
    
    Example of CORRECT format:
    {{
        "critique": "Your actual critique text here",
        "clarity": 0.85,
        "succinct": 0.75,
        "readabilty": 0.90,
        "revision_needed": true,
        "response": {{
                    "answer": "The provided detailed answer here",
                    "question": "The original question", 
                    "source": "D&D Player's Handbook",
                    "context_summary": "Summary of context used"
                }}
    }}
                                                      
    {format_instructions}
    """),
            # This is used to inject the actual content or message that the post will be based on.  
            # The placeholder will be populated with the user's request at runtime.
            MessagesPlaceholder(variable_name="messages")  
        ])
        reflection_critique_parser = PydanticOutputParser(pydantic_object=CritiqueOfAnswerSchema)
        
        # Add format instructions to the prompt
        generated_reflection_prompt = generated_reflection_prompt.partial(
            format_instructions=reflection_critique_parser.get_format_instructions()
        )
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
        ai_messages = state.get("messages")
        
        logger.info(f"Generation node processing question: {question[:50]}...")
        
        # Create Generation Prompt with context and question
        logger.info("Invoking generation chain...")
        last_message =  [ ai_messages[-1] ]
        
        generated_response_output = agent_components.generation_chain.invoke({
                "question": question,
                "messages": last_message,
                "sources":"""
1) Dungeon Master’s Guide - Dungeons & Dragons - Sources - D&D Beyond
2) Monster Manual - Dungeons & Dragons - Sources - D&D Beyond
3) Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond
4) Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond
"""
            }
        )
        logger.info("Generation successful")
        generated_response_obj = agent_components.extract_json_response_output(generated_response_output.content)
        # Extract answer from the parsed Pydantic object
        if isinstance(generated_response_obj, QuestionResponseSchema):
            generated_response_and_answer = f"""ANSWER: {{
        "answer": "{generated_response_obj.answer}",
        "question": "{question}", 
        "source": "{generated_response_obj.source}",
        "context_summary": "{generated_response_obj.context_summary}"
    }}"""     
        else:
            generated_response_and_answer = str(generated_response_obj)
                
        
        
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
    return {**state, "continue_refining": answer_provided, "messages": ai_messages , "context": agent_components.context}
        

def agent_reflection_node( state: ReflectionAgentState) -> ReflectionAgentState:
    """
    In LangGraph Agent Flows are represented via Agent WorkFolws where 
    States accumulate and change base on their travels from Node to Node

    This Node Corresponds to a Reflection Agent's Generated Reponse

    Args:
        state: 
    """
    try:
        agent_components = state["agent_instance"]
        ai_messages = state.get("messages")
        question = state.get("question")
        logger.info("Invoking reflection chain...")
        last_response =  [ ai_messages[-1] ]

        # Create Reflection on current messages
        response_reflection_output = agent_components.reflection_chain.invoke(
            {
                "question": question,
                "messages": last_response,
                "sources":"""    
1) Player’s Handbook - Dungeons & Dragons - Sources - D&D Beyond
2) Horror Adventures - Van Richten’s Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond
"""
            }
        )
        logger.info("Reflection successful")
        response_reflection_obj= agent_components.extract_json_reflection_output(response_reflection_output.content)
        # Extract critique from the parsed Pydantic object
        critique_provided = True
        if isinstance(response_reflection_obj, CritiqueOfAnswerSchema):
            response_reflection = f"""
            Evaluated Text: {response_reflection_obj.response}
            Critique: {response_reflection_obj.critique} 
            """
            critique_provided = response_reflection_obj.revision_needed
            if response_reflection_obj.clarity <=.5  or response_reflection_obj.succinct <=.5  or response_reflection_obj.readabilty <=.5 and not critique_provided:
                print(f""" 
INCORRECT EVALUATION:
                    clarity: {response_reflection_obj.clarity}
                    succinct:{response_reflection_obj.succinct}
                    readable:{response_reflection_obj.readabilty}
BUT RETRY is {critique_provided}
                      """)
        else:
            response_reflection = str(response_reflection_obj) 
            if not (response_reflection and response_reflection.strip()):
                critique_provided = False     
        
    except Exception as e:
        logger.error(f"❌ Reflection node failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return with no refinement needed on error
        return {**state, "continue_refining": False}
    ai_messages.append(HumanMessage(content=response_reflection))
    return {**state, "continue_refining": critique_provided , "messages": ai_messages , "context": response_reflection} # pretend human feedback


def should_agent_reflect(state: ReflectionAgentState):  
    agent_components = state["agent_instance"]
    if agent_components.count <5 and state.get("continue_refining", True):
        agent_components.count +=1
    else:
        state["continue_refining"] = False
        return END
    return "agent_reflection_node"

def should_agent_retry(state: ReflectionAgentState):
    if  state.get("continue_refining", True):
        return "agent_generation_node"
    return END


def get_gpu_usage():
     # Check GPU memory
    memory_stat = check_gpu_memory_status()
    print(f"📊BEFORE GPU Memory Status:")
    print(f"   - Total: {memory_stat['total_mb']} MB")
    print(f"   - Used: {memory_stat['used_mb']} MB ({memory_stat['used_ratio']:.1%})")
    print(f"   - Free: {memory_stat['free_mb']} MB ({memory_stat['free_ratio']:.1%})")
    return memory_stat

def resolve_vllm_memory_issues():
    """
    Utility function to diagnose and resolve VLLM memory issues.
    
    This function can be called independently to:
    1. Check GPU memory status
    2. Clean up GPU memory
    3. Test VLLM with process isolation
    4. Provide specific recommendations
    """
    print("🔍 VLLM Memory Issue Resolver")
    print("=" * 50)
    
   
    memory_status =get_gpu_usage()
    # Analyze memory situation
    if memory_status['free_ratio'] < 0.3:
        print("\n⚠️  Low GPU memory detected!")
        print("\n🧹 Attempting memory cleanup...")
        force_gpu_memory_cleanup()
        
        # Check again after cleanup
        memory_status = check_gpu_memory_status()
        print(f"   After cleanup: {memory_status['free_mb']} MB free")
    
        # Check GPU memory
    memory_status =get_gpu_usage()
    if memory_status['free_ratio'] < 0.6:  # Need at least 60% free memory
        raise RuntimeError(f"Insufficient GPU memory: only {memory_status['free_mb']} MB free")



if __name__ == "__main__":
    llm = None
    force_gpu_memory_cleanup()
    try:
        logger.info("🚀 Starting Reflection Agent...")
        
        # Create Agent Brain
        logger.info("Initializing vLLM engine...")
        # get_langchain_vllm_mistral_quantized(download_dir="./models")
        llm = create_vllm_chat_model( download_dir="./models")  # Change this to your model directory
        logger.info("✅ vLLM engine initialized successfully")
        
        reflection_agent = ReflectionAgent(
            brain=llm,  #"TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
            embedding_model = "BAAI/bge-large-en-v1.5", # good for RAG Searching 
            root_path = "/home/jmitchall/vllm-srv" ,
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
        
        query = "Who is Strahd von Zarovich in Dungeons and Dragons lore?"
        logger.info(f"Processing query: {query}")

        # Invoke workflow with initial state
        result = workflow_graph.invoke(reflection_agent.get_initial_state (question=query))
        logger.info("✅ Workflow completed successfully")
        logger.info(f"\n\n{"=" * 80}\nTRACE result: {result}")
        logger.info(f"\n\nFinal result: {result["messages"][-1].content}")
        
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
            
            print(check_gpu_memory_status())
            logger.info("✅ Workflow complete, exiting...")
            import sys
            import os
            llm.cleanup_llm_memory()
            # Suppress the benign "Engine core died" message by redirecting stderr
            # This message appears during normal shutdown and is not an error
            sys.stderr = open(os.devnull, 'w')
            # Engine will cleanup automatically when Python exits
            force_gpu_memory_cleanup()

    



    


