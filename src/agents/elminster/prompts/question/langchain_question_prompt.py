#!/usr/bin/env python3
"""
Elminster Agent - Question Answering Prompts and Schema

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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from refection_logger import logger
from agents.elminster.questions import question_response_output
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
     template += question_response_output
     template += """
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