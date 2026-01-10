from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from agents.elminster.prompts.question.langchain_question_prompt import QuestionResponseSchema
from refection_logger import logger
from refection_logger import logger
import json
import re

class CritiqueOfAnswerSchema(BaseModel):
    critique: str = Field(
        description="the evaluation of the RESPONSE or ANSWER to a USER'S QUERY  based on it's clarity, succinctness, and readability")
    clarity: bool = Field(
        description="Single boolean returning False when the RESPONSE is incoherrent and true means the explaination is really easy to understand")
    succinct: bool = Field(
        description="Single boolean returning False when the text length is too large and true means the explaination is as concise enough to still answer comprehensively")
    readabilty: bool = Field(
        description="Single boolean returning False when answer interpretation requires a graduate degress and true when understanding requires at least a 5th grade reading level")
    revision_needed: bool = Field(
        description="Single boolean returning True if the answer needs to be improved based on the evaluation, and False if it is sufficently clear, succinct, and readable")
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


def validate_and_correct_revision_needed(critique_obj: CritiqueOfAnswerSchema) -> CritiqueOfAnswerSchema:
    """
    Post-process validation to enforce revision_needed logic based on rubrics and critique content.
    
    This function enforces the rules that LLMs often fail to follow:
    1. Any failing rubric (clarity, succinct, readabilty = False) → revision_needed must be True
    2. Critique containing suggestion words → revision_needed must be True
    
    Args:
        critique_obj: The CritiqueOfAnswerSchema object from LLM
        
    Returns:
        CritiqueOfAnswerSchema with corrected revision_needed value
    """
    original_revision_needed = critique_obj.revision_needed
    corrected = False
    reason = []
    
    # RULE 1: Check if any rubric failed
    if not critique_obj.clarity or not critique_obj.succinct or not critique_obj.readabilty:
        if not critique_obj.revision_needed:
            critique_obj.revision_needed = True
            corrected = True
            failed_rubrics = []
            if not critique_obj.clarity:
                failed_rubrics.append("clarity")
            if not critique_obj.succinct:
                failed_rubrics.append("succinct")
            if not critique_obj.readabilty:
                failed_rubrics.append("readabilty")
            reason.append(f"Failed rubrics: {', '.join(failed_rubrics)}")
    
    # RULE 2: Check for trigger words in critique (context-aware)
    trigger_words = [
        "benefit", "could", "should", "would", "might", "however",
        "consider", "recommend", "suggest", "better", "enhance", 
        "improve", "add", "include", "more", "lacks", "missing",
        "needs", "require", "ought"
    ]
    
    critique_lower = critique_obj.critique.lower()
    
    # Check for negation patterns that indicate NO revision is needed
    negation_patterns = [
        r'\bno\s+\w*\s*improvements?\b',
        r'\bno\s+\w*\s*revision\b',
        r'\bno\s+\w*\s*changes?\b',
        r'\bno\s+\w*\s*modifications?\b',
        r'\bnot?\s+\w*\s*need',
        r'\bdon\'?t\s+need',
        r'\bdoes\s?n\'?t\s+need',
        r'\bwithout\s+need',
        r'\bcould\s+not\s+be\s+better',
        r'\bnothing\s+\w*\s*needs?\b',
    ]
    
    has_negation = any(re.search(pattern, critique_lower) for pattern in negation_patterns)
    
    # Only check for trigger words if no negation pattern is found
    if not has_negation:
        found_triggers = [word for word in trigger_words if word in critique_lower]
        
        if found_triggers and not critique_obj.revision_needed:
            critique_obj.revision_needed = True
            corrected = True
            reason.append(f"Trigger words found: {', '.join(found_triggers)}")
    
    # Log correction if made
    if corrected:
        logger.warning(f"""
🔧 CORRECTED revision_needed: {original_revision_needed} → {critique_obj.revision_needed}
   Reason: {' | '.join(reason)}
   Rubrics: clarity={critique_obj.clarity}, succinct={critique_obj.succinct}, readabilty={critique_obj.readabilty}
   Critique excerpt: {critique_obj.critique[:100]}...
        """)
    
    return critique_obj


def get_critique_llm_results(json_str: str):

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

        is_valid, message = CritiqueOfAnswerSchema.validate_dict_safe(return_value)
        if is_valid:
            # Create the model instance
            return CritiqueOfAnswerSchema.model_validate(return_value)

            # If validation fails, return the raw data
        logger.info(f"Validation failed for both schemas. Returning raw data: {return_value}")
        return return_value

def extract_json_reflection_output(raw_output: str):
        """
        Clean and extract JSON from LLM output that may contain extra text.
        Applies post-processing validation to ensure revision_needed is logically consistent.
        """
        # Find text between ```json and ```
        json_pattern = r'```json\s*(.*?)\s*```'
        match = re.search(json_pattern, raw_output, re.DOTALL | re.IGNORECASE)

        if match:
            json_str = match.group(1).strip()
            try:
                return_value = get_critique_llm_results(json_str)
                # Apply validation if we got a CritiqueOfAnswerSchema object
                if isinstance(return_value, CritiqueOfAnswerSchema):
                    return_value = validate_and_correct_revision_needed(return_value)
                return return_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found in code block: {e}")

        # Find text between 1st { and last }
        first_brace = raw_output.find('{')
        last_brace = raw_output.rfind('}')

        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_str = raw_output[first_brace:last_brace + 1]

            try:
                return_value = get_critique_llm_results(json_str)
                # Apply validation if we got a CritiqueOfAnswerSchema object
                if isinstance(return_value, CritiqueOfAnswerSchema):
                    return_value = validate_and_correct_revision_needed(return_value)
                return return_value
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON found between braces: {e}")

        return raw_output

def get_reflection_prompt(template:str):
    template +="""
    IMPORTANT: Return only a JSON object with actual data values, NOT a schema definition. 
    Do not include "properties" or "required" fields - just return the actual critique data.
    
    Example of CORRECT format:
    {{
        "critique": "Your actual critique text here",
        "clarity": true,
        "succinct": false,
        "readabilty": true,
        "revision_needed": true,
        "response": {{
                    "answer": "The provided detailed answer here",
                    "question": "The original question", 
                    "source": "D&D Player's Handbook",
                    "context_summary": "Summary of context used"
                }}
    }}
                                                      
    {format_instructions}
    """
    generated_reflection_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(template),
            # This is used to inject the actual content or message that the post will be based on.  
            # The placeholder will be populated with the user's request at runtime.
            MessagesPlaceholder(variable_name="messages")
        ])
    reflection_critique_parser = PydanticOutputParser(pydantic_object=CritiqueOfAnswerSchema)

    # Add format instructions to the prompt
    return generated_reflection_prompt.partial(
        format_instructions=reflection_critique_parser.get_format_instructions()
    )
