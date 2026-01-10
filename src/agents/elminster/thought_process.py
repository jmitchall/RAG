import traceback
from abc import ABC
from agents.elminster.prompts.context.langchain_context_prompt import  get_context_retrieval_prompt_tools
from agents.elminster.prompts.question.langchain_question_prompt import get_answer_prompt
from agents.elminster.prompts.reflection.langchain_critique_prompt import  get_reflection_prompt
from agents.elminster.prompts.revision.langchain_revision_prompt import get_revision_prompt
from agents.elminster.questions import dm_question_expertise, reflect_on_answer, revise_question_prompt
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from refection_logger import logger
from typing import List, Annotated, TypedDict

class ReflectionAgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], "add_messages"]
    agent_instance: "ThoughtProcessAgent"
    continue_refining: bool = True
    question: str = ""
    context: str = ""
    critique: str = ""

class ThoughtProcessAgent(ABC):
    def __init__(self, brain, root_path="/home/jmitchall/vllm-srv", **kwargs):
        self.llm = brain
        # Create Generation Prompt for answer
        self.generation_chain = self.get_generation_chain()
        # Create Reflection Prompt for Answer Evaluation
        self.reflection_chain = self.get_reflection_chain()
        # Create Context Chain
        self.context_chain = self.get_context_chain()
        # Create Improve Question Chain
        self.revision_chain = self.get_revision_chain()
        self.count = 0
        self.DATABASE_TYPE = kwargs.get("DATABASE_TYPE")
        self.root_path = root_path
        self.question = ""
        self.context = ""

    def get_initial_state(self, question: str = "") -> ReflectionAgentState:
        return ReflectionAgentState(
            agent_instance=self,
            question=question,
            messages=[HumanMessage(content=question)] if question else []
        )

    def handle_tool_calls(self, message: AIMessage, tools: List[BaseTool]) -> dict:
        """Process tool calls from the LLM and invoke matching tools"""
        if not hasattr(message, 'tool_calls') or not message.tool_calls:
            # No tool calls, return basic structure
            return {
                "question": self.question or "",
                "context": self.context or "",
                "messages": [message]
            }

        # Create a mapping of tool names to tool instances
        tool_map = {tool.name: tool for tool in tools}

        for tool_call in message.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            if tool_name in tool_map:
                try:
                    tool_instance = tool_map[tool_name]
                    # Invoke the tool with the provided arguments
                    result = tool_instance._run(**tool_args)
                    logger.info(f"✅ Tool '{tool_name}' executed successfully")

                    # Update agent's context and question from tool result
                    if isinstance(result, dict):
                        if 'context' in result:
                            self.context = result['context']
                        if 'question' in result:
                            self.question = result['question']

                except Exception as e:
                    logger.info(f"❌ Error executing tool '{tool_name}': {str(e)}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")
            else:
                logger.info(f"⚠️ Tool '{tool_name}' not found in available tools")

        # Return the input format expected by answer_generation_prompt
        return {
            "question": self.question or "",
            "context": self.context or "",
            "messages": [message]
        }

    def get_revision_chain(self):
        improve_generation_prompt = get_revision_prompt(revise_question_prompt)
        return improve_generation_prompt | RunnableLambda(
            lambda x: self.llm.unbind_tools(x)) | self.llm

    def get_context_chain(self):
        context_retrieval_prompt, tools = get_context_retrieval_prompt_tools()
        llm_with_tools = self.llm.bind_tools(tools)  # This creates new instances
        return context_retrieval_prompt | llm_with_tools | RunnableLambda(
            lambda x: self.handle_tool_calls(x, tools))

    def get_generation_chain(self):
        answer_generation_prompt = get_answer_prompt(dm_question_expertise)

        return answer_generation_prompt | RunnableLambda(
            lambda x: self.llm.unbind_tools(x)) | self.llm

    def get_reflection_chain(self):
        generated_reflection_prompt = get_reflection_prompt(reflect_on_answer)
        return generated_reflection_prompt | self.llm
