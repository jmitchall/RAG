#!/usr/bin/env python3
"""
CrewAI LLM Agent Test - Experimental CrewAI Integration

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

import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports when running directly
if __name__ == "__main__":
    src_path = Path(__file__).parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

from crewai import Agent, Task, Crew

from inference.vllm_srv.minstral_crewai import create_crewai_vllm_model
from agents.elminster.prompts.question.langchain_question_prompt import QuestionResponseSchema
from agents.elminster.tools.refresh_question_context_crewai_tool import RefreshQuestionContextToolCrewAI
from inference.vllm_srv.cleaner import check_gpu_memory_status, force_gpu_memory_cleanup
from refection_logger import logger
from agents.elminster.questions import dm_question_expertise, question_response_output
from langchain.output_parsers import PydanticOutputParser
from crewai.project import CrewBase, agent, task

@CrewBase
class CrewAIReflectionAgent():
    """
    CrewAI-based reflection agent system for D&D question answering.
    
    This class uses CrewAI's @CrewBase decorator which provides automatic
    agent and task management. The decorator enables the use of @agent and @task
    decorators on methods, making them CrewAI-compatible components.
    """

    def __init__(self, llm):
        """
        Initialize the CrewAI Reflection Agent.
        
        This is a STANDARD PYTHON CONSTRUCTOR (not from CrewAI base class).
        CrewAI doesn't require specific initialization patterns, so this is
        a custom constructor for our implementation.
        
        Args:
            llm: Large Language Model instance used by all agents in the crew.
                This should be a CrewAI-compatible LLM wrapper (e.g., from
                create_crewai_vllm_model()). The same model is used for both:
                - General reasoning and text generation
                - Function calling / tool invocation
        """
        self.llm = llm

    @agent
    def context_agent(self):
        """
        Create a context retriever agent that uses tools to fetch D&D rulebook context.
        
        **THIS METHOD USES CrewAI's @agent DECORATOR**
        
        The @agent decorator is a CrewAI feature that marks this method as an agent factory.
        CrewAI will call this method to create an Agent instance that can be assigned to tasks.
        
        Why CrewAI needs the @agent decorator:
        - Allows CrewAI to automatically discover and manage agents in the crew
        - Enables proper agent lifecycle management and initialization
        - Integrates agents with CrewAI's task execution pipeline
        - Provides automatic dependency injection for crew components
        
        Returns:
            Agent: A CrewAI Agent instance configured to retrieve context from D&D rulebooks.
                  The agent has access to RefreshQuestionContextToolCrewAI for searching
                  vector databases containing rulebook content.
        
        Agent Configuration Explained:
            role (str): The agent's job title/identity. CrewAI uses this to help the LLM
                       understand what role it should play during task execution.
            
            goal (str): What the agent is trying to achieve. Guides the agent's decision-making
                       and helps it understand success criteria.
            
            backstory (str): Detailed background/instructions for the agent. This is crucial
                            for defining behavior, constraints, and operating procedures.
                            Think of it as the agent's system prompt.
            
            verbose (bool): If True, prints detailed logs during agent execution showing
                           thought process, tool calls, and reasoning steps.
            
            llm: The language model used for general reasoning tasks including:
                 * Reading and understanding task descriptions
                 * Analyzing tool results and incorporating them into responses
                 * Generating the final formatted output
                 * General planning and reasoning
            
            function_calling_llm: The model used specifically for tool calling operations:
                                 * Deciding which tool to call based on task requirements
                                 * Generating correct parameters for tool invocation
                                 * Parsing tool schemas to understand required arguments
                                 * Interpreting tool results in the function calling flow
                                 
                                 By setting both llm and function_calling_llm to self.llm,
                                 we use the same vLLM model for both. You could use different
                                 models (e.g., a cheaper one for function_calling_llm).
            
            allow_delegation (bool): If True, this agent can delegate subtasks to other agents
                                    in the crew. Set to False here because this agent has a
                                    specific, self-contained job.
            
            tools (List[BaseTool]): List of tools this agent can use. Tools extend the agent's
                                   capabilities beyond just text generation (e.g., searching
                                   databases, making API calls, etc.)
        """
        context_backstory = """You are a D&D Context Retrieval Specialist. 

CRITICAL: You CANNOT answer questions from memory. You MUST ALWAYS use the refresh_question_context tool to retrieve information from the official D&D rulebooks before responding.

Your ONLY job is to:
1. Receive a question about D&D
2. Use the refresh_question_context tool to search the rulebooks
3. Return the retrieved context

You do NOT have knowledge stored in your training. All D&D information MUST come from the tool."""

        self.context_agent= Agent(
            role="D&D Context Retrieval Specialist",
            goal="Retrieve accurate D&D rulebook context using the refresh_question_context tool",
            backstory=context_backstory,
            verbose=True,
            llm=self.llm,  # General reasoning and output generation
            function_calling_llm=self.llm,  # Tool selection and parameter generation
            allow_delegation=False,
            tools = [RefreshQuestionContextToolCrewAI()]
        )
        return self.context_agent

    @task
    def context_retriever_task(self , question: str = "What is a Rogue?" , vector_db_type: str = "qdrant",
                               root_path: str = "/home/jmitchall/vllm-srv"):
        """
        Create a task for retrieving D&D rulebook context using the context agent.
        
        **THIS METHOD USES CrewAI's @task DECORATOR**
        
        The @task decorator marks this method as a task factory. CrewAI calls this method
        to create Task instances that define work for agents to perform.
        
        Why CrewAI needs the @task decorator:
        - Enables automatic task discovery and registration in the crew
        - Allows CrewAI to manage task dependencies and execution order
        - Integrates tasks with the crew's workflow engine
        - Provides proper task lifecycle management
        
        Args:
            question (str): The user's question about D&D rules, mechanics, monsters, etc.
                           This is passed to the tool as the search query.
                           Default: "What is a Rogue?"
            
            vector_db_type (str): Which vector database system to use for retrieval.
                                 Options: 'qdrant', 'faiss', or 'chroma'
                                 Each has different performance characteristics.
                                 Default: "qdrant"
            
            root_path (str): Absolute path to project root where vector databases are stored.
                           The method will look for databases in {root_path}/db/
                           Default: "/home/jmitchall/vllm-srv"
        
        Returns:
            Task: A CrewAI Task instance that instructs the context_agent to retrieve
                 relevant context from D&D rulebooks using the refresh_question_context tool.
        
        Task Configuration Explained:
            description (str): Detailed instructions for the agent about what to do.
                              This is like a work order that tells the agent:
                              - What the goal is
                              - What parameters to use
                              - Which tool to call
                              - Any important constraints or requirements
            
            expected_output (str): Description of what the task result should look like.
                                  CrewAI uses this to help the agent format its output
                                  correctly and to validate that the task completed properly.
                                  This acts as a template/specification for the agent.
            
            agent (Agent): The agent instance that will execute this task.
                          Must be an agent created with the @agent decorator.
        """
        # Create proper source list (elminster_sources is a string with instructions, not a list)
        actual_sources = [
            "Dungeon Master's Guide - Dungeons & Dragons - Sources - D&D Beyond",
            "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond",
            "Player's Handbook - Dungeons & Dragons - Sources - D&D Beyond",
            "Horror Adventures - Van Richten's Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond"
        ]
        
        # Format sources as a simple list for the description
        sources_list = "\n".join([f"- {src}" for src in actual_sources])
        
        context_task_description = f"""Your task is to retrieve relevant context from D&D rulebooks to answer the user's question.

USER'S QUESTION: {question}

YOU MUST use the refresh_question_context tool with these exact parameters:
- question: "{question}"
- sources: {actual_sources}
- db_type: "{vector_db_type}"
- root_path: "{root_path}"

Available sources to search:
{sources_list}

IMPORTANT: Use the refresh_question_context tool to retrieve context. Do not try to answer the question without using the tool first."""
        
        return Task(
            description=context_task_description,
            expected_output="""Structured output containing the question, retrieved context, and optional critique.
            
            Format the output as:
            QUESTION: [The original question from USER QUERY]
            
            CONTEXT: [Retrieved text from relevant D&D rulebook sourcebooks that provide information to answer the question.
            This should be comprehensive and include specific details, rules, abilities, and mechanics.]
            
            CRITIQUE: [Any critique or refinement suggestions, or leave empty if not applicable]
            
            The context must be detailed and comprehensive enough to fully answer the question.""",
            agent=self.context_agent()
        )

    @agent
    def question_response_agent(self):
        """
        Create an agent that generates expert answers to D&D questions using provided context.
        
        **THIS METHOD USES CrewAI's @agent DECORATOR**
        
        The @agent decorator marks this as an agent factory for CrewAI's automatic
        agent management system (see context_agent() for full @agent explanation).
        
        Returns:
            Agent: A CrewAI Agent instance configured as a D&D Game Master that can
                  read context and generate comprehensive answers to questions.
        
        Agent Configuration Explained:
            role (str): Agent's identity as a "Dungeons and Dragons Game Master"
                       with question-answering expertise.
            
            goal (str): What this agent aims to accomplish - providing compelling
                       and accurate responses to D&D questions.
            
            backstory (str): Imported from dm_question_expertise, contains detailed
                            instructions about how to behave as a knowledgeable DM
                            and answer questions accurately.
            
            verbose (bool): If True, shows detailed execution logs.
            
            llm: Language model for reasoning and text generation. Same model used
                for both general reasoning and function calling since this agent
                doesn't need separate models.
            
            allow_delegation (bool): Set to False because this agent has a specific
                                    terminal task (generate answer) and shouldn't
                                    delegate to other agents.
        """
        self.question_agent = Agent(
            role="Questions Answering Dungeons and Dragons Game Master",
            goal="provide compelling and accurate response to it",
            backstory=dm_question_expertise,
            verbose=True,
            llm=self.llm,
            allow_delegation=False # Disable delegation for this agent meaning it cannot call other agents
        )
        return self.question_agent

    @task
    def question_response_task(self):
        """
        Create a task for generating expert D&D answers using context from the previous task.
        
        **THIS METHOD USES CrewAI's @task DECORATOR**
        
        The @task decorator marks this as a task factory for CrewAI's task management
        (see context_retriever_task() for full @task explanation).
        
        This task is designed to receive output from context_retriever_task via task
        dependencies (set using task.context). When executed, this task will have access
        to the QUESTION and CONTEXT retrieved by the previous task.
        
        Args:
            None - This task receives all its input from the previous task's output
                  via CrewAI's context mechanism.
        
        Returns:
            Task: A CrewAI Task instance that instructs the question_response_agent to
                 read the context from the previous task and generate a comprehensive
                 answer with proper JSON formatting.
        
        Task Configuration Explained:
            description (str): Instructions for the agent explaining that it will receive
                              QUESTION and CONTEXT from the previous task, and should
                              generate an expert response answering the question using
                              the provided context.
            
            expected_output (str): Specification of the output format. Includes Pydantic
                                  format instructions which tell the agent exactly how to
                                  structure its JSON response according to QuestionResponseSchema.
            
            agent (Agent): The question_response_agent that will execute this task.
            
            output_pydantic (Type[BaseModel]): **CrewAI-specific parameter**
                                               Tells CrewAI to parse the agent's output
                                               into a Pydantic model (QuestionResponseSchema).
                                               This ensures structured, validated output
                                               rather than just plain text.
            
            output_file (str): **CrewAI-specific parameter**
                              File path where CrewAI should save the task output.
                              CrewAI will automatically write the result to this JSON file
                              after the task completes.
        """
        answer_generation_parser = PydanticOutputParser(pydantic_object=QuestionResponseSchema)

        question_output = question_response_output
        question_output += f"""
        
{answer_generation_parser.get_format_instructions()}"""

        # Build the task description
        # When task.context is set, CrewAI will provide the output from context_task
        # The agent will have access to the QUESTION and CONTEXT from the previous task
        # The LLM can read and understand the formatted text with QUESTION: and CONTEXT: sections
        task_description = """Your TASK:
Generate the best possible RESPONSE or ANSWER to the USER'S QUERY.

You will receive CONTEXT from the previous task that includes:
- QUESTION: The original question to answer
- CONTEXT: Retrieved text from D&D rulebooks relevant to the question
- CRITIQUE: Optional critique or refinement suggestions

IMPORTANT: Read the QUESTION from the previous task's output and answer it directly.

Use this information to deliver an expert RESPONSE that:
1. Directly answers the USER'S QUERY found in the QUESTION section above
2. Cites the D&D sources used from the CONTEXT section
3. Summarizes the key information from the context used to answer

The context from the previous task will be provided below."""

        return Task(
            description=task_description,
            expected_output=question_output,
            agent=self.question_response_agent(),
            output_pydantic=QuestionResponseSchema,
            output_file="response.json"
        )


if __name__ == "__main__":
    llm = None
    force_gpu_memory_cleanup()
    try:
        logger.info("🚀 Starting Reflection Agent...")

        # NOTE: OPENAI_API_KEY is NOT required when using custom LLMs
        # CrewAI only requires this environment variable when:
        # 1. Using string model names like llm="gpt-4"
        # 2. Not providing a custom LLM instance
        # Since we're using create_crewai_vllm_model() which returns a custom
        # BaseLLM implementation, the OPENAI_API_KEY is unnecessary.
        # 
        # The confusion comes from older CrewAI versions (< 0.28.0) that had
        # stricter validation. Modern versions are more flexible with custom LLMs.

        # Create Agent Brain with CrewAI-compatible wrapper
        logger.info("Initializing vLLM engine with CrewAI wrapper...")
        llm = create_crewai_vllm_model(download_dir="./models")  # Returns CrewAI-compatible wrapper
        logger.info("✅ vLLM engine initialized successfully")

        # Initialize CrewAI Reflection Agent
        reflection_agent = CrewAIReflectionAgent(llm=llm)

        # Define the user question
        user_question = "What is a Warlock in D&D?"

        # Create tasks with proper dependencies
        # Task 1: Context retrieval using the tool
        context_task = reflection_agent.context_retriever_task(question=user_question)
        
        # Task 2: Answer generation using context from Task 1
        question_task = reflection_agent.question_response_task()
        
        # Set up task dependency - question_task receives output from context_task
        question_task.context = [context_task]

        # Create Crew with both tasks in sequence
        # Note: CrewAI automatically extracts agents from tasks, so we don't need to specify them
        crew = Crew(
            tasks=[context_task, question_task],
            verbose=True
        )

        # Kickoff the CrewAI process
        result = crew.kickoff()

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
