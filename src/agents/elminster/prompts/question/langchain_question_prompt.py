
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from refection_logger import logger
import re
import json

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

    @classmethod
    def validate_dict_safe(cls, data: dict) -> tuple[bool, str]:
        """Safely validate dict and return success status with error message."""
        try:
            cls.model_validate(data)
            return True, "Valid"
        except Exception as e:
            return False, str(e)

def get_response_llm_results(json_str: str):

        answer_generation_parser = PydanticOutputParser(pydantic_object=QuestionResponseSchema)
        try:
            return_value = answer_generation_parser.parse(json_str)
            return return_value
        except Exception as answer_generation_excpt:
            logger.info(f"answer_generation_excpt : {answer_generation_excpt}")

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

            # If validation fails, return the raw data
        logger.info(f"Validation failed for both schemas. Returning raw data: {return_value}")
        return return_value



def get_answer_prompt(template:str):
     template += """
USE the follwing Format for response:

CRITICAL REQUIREMENTS - ALL 4 FIELDS ARE MANDATORY:
1. "answer" - The complete expert response to the user's query
2. "question" - The exact original question being answered
3. "source" - The specific D&D source material cited (e.g., Player's Handbook, Monster Manual)
4. "context_summary" - A concise summary (under 500 characters) of the context used

IMPORTANT: Return only a JSON object with actual data values, NOT a schema definition. 
Do not include "properties" or "required" fields - just return the actual answer data.

Example of CORRECT format - ALL 4 FIELDS REQUIRED:
{{
    "answer": "Rogues gain several key class features: Expertise (proficiency bonus doubled for two skills), Sneak Attack (extra damage when attacking with advantage), Thieves' Cant (secret language), and Cunning Action (bonus action for Dash, Disengage, or Hide).",
    "question": "What are the class features available to Rogues?", 
    "source": "Player's Handbook - Dungeons & Dragons - Sources - D&D Beyond",
    "context_summary": "Rogue class features from levels 1-20 including base abilities, Cunning Action at 2nd level, and subclass-specific features"
}}

MANDATORY FIELDS CHECKLIST:
✓ answer (string - detailed response)
✓ question (string - original question)
✓ source (string - D&D source material)
✓ context_summary (string - under 500 chars)

{format_instructions}"""
     answer_generation_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(template),
            # This is used to inject the actual content or message that the post will be based on.
            # The placeholder will be populated with the user's request at runtime.
            MessagesPlaceholder(variable_name="messages")
        ])
     answer_generation_parser = PydanticOutputParser(pydantic_object=QuestionResponseSchema)

        # Add format instructions to the prompt
     return answer_generation_prompt.partial(
        format_instructions=answer_generation_parser.get_format_instructions()
    )
     
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
                return_value = get_response_llm_results(json_str)
                return return_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found in code block: {e}")

        # Find text between 1st { and last }
        first_brace = raw_output.find('{')
        last_brace = raw_output.rfind('}')

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_output[first_brace:last_brace + 1]

            try:
                return_value = get_response_llm_results(json_str)
                return return_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found between braces: {e}")

        return raw_output