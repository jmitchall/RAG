import os
import traceback
from agents.elminster.thought_process import ThoughtProcessAgent, ReflectionAgentState
from agents.elminster.knowledge import elminster_sources
from agents.elminster.prompts.context.langchain_context_prompt import get_tool_input_request
from agents.elminster.prompts.question.langchain_question_prompt import QuestionResponseSchema,  extract_json_response_output
from agents.elminster.prompts.reflection.langchain_critique_prompt import CritiqueOfAnswerSchema, extract_json_reflection_output
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from refection_logger import logger


from inference.vllm_srv.cleaner import check_gpu_memory_status, force_gpu_memory_cleanup
from inference.vllm_srv.minstral_langchain import create_vllm_chat_model
from agents.elminster.thought_process import ReflectionAgentState


def agent_context_node(state: ReflectionAgentState) -> ReflectionAgentState:
    """
    In LangGraph Agent Flows are represented via Agent WorkFolws where 
    States accumulate and change base on their travels from Node to Node

    This Node Cooresponds to a Reflection Agent's Context Retrieval

    Args:
        state: 
    """
    try:
        agent_components = state["agent_instance"]
        question = state.get("question")
        ai_messages = state.get("messages")
        critique = state.get("critique")
        logger.info(f"Generation node processing question: {question[:50]}...")

        # Create Generation Prompt with context and question
        logger.info("Invoking generation chain...")
        last_message = [ai_messages[-1]]

        generated_response_obj = agent_components.context_chain.invoke(
            {
                "input": get_tool_input_request(question=question, root_path=agent_components.root_path,
                                                db_type="qdrant", critique=critique, sources=elminster_sources),
                "messages": last_message
            }
        )
        agent_components.context = generated_response_obj.get("context", agent_components.context)
        logger.info("Generation successful")


    except Exception as e:
        logger.error(f"❌ Generation node failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return error message in state
        error_msg = f"Generation failed: {str(e)}"
        return {**state, "messages": [AIMessage(content=error_msg)]}

    return {**state, "context": agent_components.context}


def agent_improve_question_node(state: ReflectionAgentState) -> ReflectionAgentState:
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
        critique = state.get("critique")
        logger.info(f"Improve Question node processing question: {question[:50]}...")
        last_message = [ai_messages[-1]]

        generated_question_message = agent_components.revision_chain.invoke({
            "question": question,
            "critique": critique,
            "messages": last_message
        }
        )
        response_improvement = generated_question_message.content
        logger.info(f"Generation successful: {response_improvement}")
        ai_messages.append(HumanMessage(content=response_improvement))
        state["question"] = response_improvement
        state["critique"] = ""
    except Exception as e:
        logger.error(f"❌ Reflection node failed: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Return with no refinement needed on error
        return {**state, "continue_refining": False}

    return {**state, "messages": ai_messages, "question": response_improvement}


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
        ai_messages = state.get("messages")
        context = state.get("context")
        logger.info(f"Generation node processing question: {question[:50]}...")

        # Create Generation Prompt with context and question
        logger.info("Invoking generation chain...")
        last_message = [ai_messages[-1]]

        generated_response_output = agent_components.generation_chain.invoke(
            {
                "question": question,
                "context": context,
                "messages": last_message
            }
        )
        logger.info("Generation successful")
        generated_response_obj = extract_json_response_output(generated_response_output.content)
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
    return {**state, "continue_refining": answer_provided, "messages": ai_messages, "context": agent_components.context}


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
        context = state.get("context")
        # Create Reflection on current messages
        response_reflection_output = agent_components.reflection_chain.invoke(
            {
                "question": question,
                "context": context,
                "messages": last_response,

            }
        )
        logger.info("Reflection successful")
        response_reflection_obj = extract_json_reflection_output(response_reflection_output.content)
        # Extract critique from the parsed Pydantic object
        # Note: revision_needed has been validated and corrected by validate_and_correct_revision_needed()
        critique_provided = True
        if isinstance(response_reflection_obj, CritiqueOfAnswerSchema):
            response_reflection = get_reflection_assessment(response_reflection_obj.response.answer,response_reflection_obj.critique)
            state["critique"] = response_reflection_obj.critique
            critique_provided = response_reflection_obj.revision_needed
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

def do_i_have_enough_context(state: ReflectionAgentState):
    context = state.get("context", "")
    if context and len(context.strip()) > 50:
        return "agent_generation_node"
    logger.info("Not enough context retrieved, Elminster cannot proceed.")
    state["messages"].append(AIMessage(content="Not enough context retrieved, Elminster cannot proceed."))
    return END


if __name__ == "__main__":
    llm = None
    force_gpu_memory_cleanup()
    try:
        logger.info("🚀 Starting Reflection Agent...")

        # Create Agent Brain
        logger.info("Initializing vLLM engine...")
        # get_langchain_vllm_mistral_quantized(download_dir="./models")
        llm = create_vllm_chat_model(download_dir="./models")  # Change this to your model directory
        logger.info("✅ vLLM engine initialized successfully")

        reflection_agent = ThoughtProcessAgent(
            brain=llm,  # "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
            embedding_model="BAAI/bge-large-en-v1.5",  # good for RAG Searching
            root_path="/home/jmitchall/vllm-srv",
        )

        logger.info("✅ Reflection agent created")

        # Initialize a StateGraph
        graph = StateGraph(ReflectionAgentState)
        # Nodes -----------------
        graph.add_node("agent_context_node", agent_context_node)
        graph.add_node("agent_generation_node", agent_generation_node)
        graph.add_node("agent_reflection_node", agent_reflection_node)
        graph.add_node("agent_improve_question_node", agent_improve_question_node)
        # Edges ---------
        # graph.add_edge("agent_context_node", "agent_generation_node")
        graph.add_edge("agent_improve_question_node", "agent_context_node")
        # Conditional Termination Edges ------------
        graph.add_conditional_edges("agent_context_node", do_i_have_enough_context)
        graph.add_conditional_edges("agent_reflection_node", lambda state: "agent_improve_question_node" if state.get("continue_refining", True) else END) 
        graph.add_conditional_edges("agent_generation_node", should_agent_reflect)

        # Start Nnode ---------
        graph.set_entry_point("agent_context_node")
        workflow_graph = graph.compile()
        logger.info("✅ Workflow graph compiled")

        query = "Who is a Starjammer?"
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
            logger.info(check_gpu_memory_status())
            logger.info("✅ Workflow complete, exiting...")
            import sys
            import os

            llm.cleanup_llm_memory()
            # Suppress the benign "Engine core died" message by redirecting stderr
            # This message appears during normal shutdown and is not an error
            sys.stderr = open(os.devnull, 'w')
            # Engine will cleanup automatically when Python exits
            force_gpu_memory_cleanup()
