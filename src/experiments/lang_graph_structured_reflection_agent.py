#!/usr/bin/env python3
"""
LangGraph Structured Reflection Agent - Experimental with Pydantic Schemas

Author: Jonathan A. Mitchall
Version: 1.0
Last Updated: January 10, 2026

License: MIT License

Copyright (c) 2026 Jonathan A. Mitchall

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Revision History:
    2026-01-10 (v1.0): Initial comprehensive documentation
"""

import json
import os
import re
import sys
import traceback
from abc import ABC
from pathlib import Path

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from refection_logger import logger
from typing import List, Annotated, TypedDict

from vectordatabases.qdrant_vector_db_commands import QdrantClientSmartPointer, quadrant_does_collection_exist, \
    get_quadrant_client, get_qdrant_retriever
from vectordatabases.fais_vector_db_commands import create_faiss_vectorstore, get_faiss_retriever
from vectordatabases.chroma_vector_db_commands import get_chroma_vectorstore, get_chroma_retriever
from embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings
from inference.vllm_srv.minstral_langchain import get_langchain_vllm_mistral_quantized


class ReflectionAgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], "add_messages"]
    agent_instance: "ReflectionAgent"
    continue_refining: bool = True
    question: str = ""
    context: str = ""


class QuestionResponseSchema(BaseModel):
    answer: str = Field(description="expert's RESPONSE or ANSWER the USER'S QUERY")
    question: str = Field(description="the original question asked")
    source: str = Field(description="the original question asked")
    context_summary: str = Field(
        description="less than 500 character summary of context used to ANSWER the USER'S QUERY")

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
    critique: str = Field(
        description="the evaluation of the RESPONSE or ANSWER to a USER'S QUERY  based on it's clarity, succinctness, and readability")
    clarity: float = Field(
        description="Single float rating from 0.0 - 1.0 where 0.0 means the RESPONSE is incoherrent and hard to understand where as 1.0 means the explaination is really easy to understand")
    succinct: float = Field(
        description="Single float rating from 0.0 - 1.0 where 0.0 means the amonut of text used to answer is very large where as 1.0 means the explaination is as consise as possible without sacrificing ability to understand")
    readabilty: float = Field(
        description="Single float rating from 0.0 - 1.0 where 0.0 means the explaination requires a graduate degress to truly comprehend where as 1.0 means reading level is that of a 5th grader  ")
    revision_needed: bool = Field(
        description="Single boolean returning True if the answr needs to be improved based on the evaluation, and False if it is sufficently clear, succinct, and readable")
    response: QuestionResponseSchema = Field(
        description="the original RESPONSE or ANSWER to USER'S QUERY being critiqued")

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


class ReflectionAgent(ABC):

    def __init__(self, brain, embedding_model="BAAI/bge-large-en-v1.5", root_path="/home/jmitchall/vllm-srv",
                 similarity_threshold: float = 0.65, **kwargs):
        self.llm = brain
        self.similarity_threshold = similarity_threshold
        # Create Generation Prompt for answer
        self.generation_chain = self.get_generation_chain()
        # Create Reflection Prompt for Answer Evaluation
        self.reflection_chain = self.get_reflection_chain()
        self.count = 0
        self.embeddings = HuggingFaceOfflineEmbeddings(model_name=embedding_model)
        self.collection_name = kwargs.get("collection_name")
        self.DATABASE_TYPE = kwargs.get("DATABASE_TYPE")
        self.root_path = root_path
        self.question = ""
        self.context = ""

    @staticmethod
    def get_structure_llm_results(json_str: str):

        answer_generation_parser = PydanticOutputParser(pydantic_object=QuestionResponseSchema)
        try:
            return_value = answer_generation_parser.parse(json_str)
            return return_value
        except Exception as answer_generation_excpt:
            logger.info(f"answer_generation_excpt : {answer_generation_excpt}")

        reflection_critique_parser = PydanticOutputParser(pydantic_object=CritiqueOfAnswerSchema)
        try:
            return_value = reflection_critique_parser.parse(json_str)
            return return_value
        except Exception as reflection_critique_excpt:
            logger.info(f"reflection_critique_parser : {reflection_critique_excpt}")

        # Try direct JSON parsing
        try:
            return_value = json.loads(json_str)
        except Exception as json_parse_excpt:
            logger.info(f"json.loads error: {json_parse_excpt}")
            return json_str  # Return original string if all parsing fails

        # Validate against schemas
        is_valid, message = QuestionResponseSchema.validate_dict_safe(return_value)
        if is_valid:
            # Create the model instance
            return QuestionResponseSchema.model_validate(return_value)

        is_valid, message = CritiqueOfAnswerSchema.validate_dict_safe(return_value)
        if is_valid:
            # Create the model instance
            return CritiqueOfAnswerSchema.model_validate(return_value)

            # If validation fails, return the raw data
        logger.info(f"Validation failed for both schemas. Returning raw data: {return_value}")
        return return_value

    @staticmethod
    def extract_json_output(raw_output: str):
        """
        Clean and extract JSON from LLM output that may contain extra text.
        """
        # Find text between ```json and ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, raw_output, re.DOTALL | re.IGNORECASE)

        if match:
            json_str = match.group(1).strip()
            try:
                return_value = ReflectionAgent.get_structure_llm_results(json_str)
                return return_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found in code block: {e}")

        # Find text between 1st { and last }
        first_brace = raw_output.find('{')
        last_brace = raw_output.rfind('}')

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_output[first_brace:last_brace + 1]

            try:
                return_value = ReflectionAgent.get_structure_llm_results(json_str)
                return return_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found between braces: {e}")

        return raw_output

    def refresh_question_context(self, question):

        if not self.question or (self.question != question):
            if question:
                self.question = question
                self.context = ""
                retriever, temp_ptr = self.get_retriever()
                if retriever:
                    retrieved_docs = retriever.invoke(question)
                    self.context = self.format_list_documents_as_string(results=retrieved_docs)
                    logger.info(f"Retrieved {len(retrieved_docs)} documents for context")
        return self.question, self.context

    def get_initial_state(self, question: str = "") -> ReflectionAgentState:
        self.refresh_question_context(question)
        return ReflectionAgentState(
            agent_instance=self,
            question=self.question,
            context=self.context,
            messages=[HumanMessage(content=question)] if question else []
        )

    def get_retriever_and_vector_stores(self, vdb_type: str, vector_db_persisted_path: str,
                                        collection_ref: str, retriever_embeddings):
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
                    langchain_retriever = get_qdrant_retriever(qdrant_client, collection_ref,
                                                               embeddings=retriever_embeddings, k=5)
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
                langchain_retriever = get_chroma_retriever(loaded_vectorstore, k=5)
            case _:
                raise ValueError(
                    f"Unsupported DATABASE_TYPE: {vdb_type}. Supported types are 'qdrant', 'faiss', 'chroma'.")
        return langchain_retriever, qdrant_client

    def get_retriever(self):
        if self.DATABASE_TYPE and self.collection_name and self.root_path:
            self.DATABASE_TYPE = self.DATABASE_TYPE.lower()
            root_persist_path = f"{self.root_path}/db"
            vector_db_persisted_path = f"{root_persist_path}/langchain_{self.collection_name}_{self.DATABASE_TYPE}"
            # check if directory exists
            if os.listdir(vector_db_persisted_path):
                return self.get_retriever_and_vector_stores(self.DATABASE_TYPE, vector_db_persisted_path,
                                                            self.collection_name, self.embeddings)
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
        context = ""
        for i, doc in enumerate(results, 1):

            similarity = doc.metadata.get("similarity_score", .7)
            if similarity < self.similarity_threshold:
                logger.info(
                    f" Skipping :{doc.metadata.get("source")}, page:{doc.metadata.get("page")} based on Threshold {self.similarity_threshold} ")
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
    and then deliver an expert's RESPONSE or ANSWER the USER'S QUERY, 
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


def agent_generation_node(state: ReflectionAgentState) -> ReflectionAgentState:
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
        last_message = [ai_messages[-1]]

        generated_response_output = agent_components.generation_chain.invoke({
            "question": question,
            "context": context,
            "messages": last_message
        }
        )
        logger.info("Generation successful")
        generated_response_obj = agent_components.extract_json_output(generated_response_output)
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
    return {**state, "continue_refining": answer_provided, "messages": ai_messages}


def agent_reflection_node(state: ReflectionAgentState) -> ReflectionAgentState:
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
        last_response = [ai_messages[-1]]
        response = ai_messages[-1].content

        # Create Reflection on current messages
        response_reflection_output = agent_components.reflection_chain.invoke(
            {
                "question": question,
                "messages": last_response
            }
        )
        logger.info("Reflection successful")
        response_reflection_obj = agent_components.extract_json_output(response_reflection_output)
        # Extract critique from the parsed Pydantic object
        critique_provided = True
        if isinstance(response_reflection_obj, CritiqueOfAnswerSchema):
            response_reflection = f"""
            Evaluated Text: {response_reflection_obj.response}
            Critique: {response_reflection_obj.critique} 
            """
            critique_provided = response_reflection_obj.revision_needed
            if response_reflection_obj.clarity <= .5 or response_reflection_obj.succinct <= .5 or response_reflection_obj.readabilty <= .5 and not critique_provided:
                logger.info(f""" 
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
    return {**state, "continue_refining": critique_provided, "messages": ai_messages,
            "context": response_reflection}  # pretend human feedback


def should_agent_reflect(state: ReflectionAgentState):
    agent_components = state["agent_instance"]
    if agent_components.count < 5 and state.get("continue_refining", True):
        agent_components.count += 1
    else:
        state["continue_refining"] = False
        return END
    return "agent_reflection_node"


def should_agent_retry(state: ReflectionAgentState):
    if state.get("continue_refining", True):
        return "agent_generation_node"
    return END


if __name__ == "__main__":
    llm = None
    try:
        logger.info("🚀 Starting Reflection Agent...")

        # Create Agent Brain
        logger.info("Initializing vLLM engine...")
        llm = get_langchain_vllm_mistral_quantized(download_dir="./models")
        logger.info("✅ vLLM engine initialized successfully")

        reflection_agent = ReflectionAgent(
            brain=llm, embedding_model="BAAI/bge-large-en-v1.5",
            root_path="/home/jmitchall/vllm-srv",
            collection_name="dnd_raven",
            DATABASE_TYPE="qdrant"
        )
        logger.info("✅ Reflection agent created")

        # Initialize a StateGraph
        graph = StateGraph(ReflectionAgentState)

        graph.add_node("agent_generation_node", agent_generation_node)
        graph.add_node("agent_reflection_node", agent_reflection_node)
        graph.add_conditional_edges("agent_reflection_node", should_agent_retry)
        graph.add_conditional_edges("agent_generation_node", should_agent_reflect)
        graph.set_entry_point("agent_generation_node")
        workflow_graph = graph.compile()
        logger.info("✅ Workflow graph compiled")

        query = "Who is Strahd von Zarovich in Dungeons and Dragons lore?"
        logger.info(f"Processing query: {query}")

        # Invoke workflow with initial state
        result = workflow_graph.invoke(reflection_agent.get_initial_state(question=query))
        logger.info("✅ Workflow completed successfully")
        logger.info(f"\n\n{"=" * 80}\nTRACE result: {result}")
        logger.info(f"\n\nFinal result: {result["messages"][-1].content}")

    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Try to get GPU info if available
        try:
            import subprocess

            gpu_info = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', '--format=csv'],
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

            # Suppress the benign "Engine core died" message by redirecting stderr
            # This message appears during normal shutdown and is not an error
            sys.stderr = open(os.devnull, 'w')
            # Engine will cleanup automatically when Python exits
