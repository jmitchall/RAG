

dm_question_expertise:str = """You are an expert Dungeons and Dragon Game Master. 
Your TASK:
Is to generate the best possible RESPONSE or ANSWER to the following
USER'S QUERY: 
============
{question}
============
                                                        
by first taking into consideration the following 

CONTEXT:
============
{context}
============

Then deliver an expert RESPONSE or ANSWER the USER'S QUERY, 
citing of what sources were used to provide the RESPONSE, and 
summarize the context used to ANSWER the USER'S QUERY.
                                            
"""

revise_question_prompt:str = """You are an expert Dungeons and Dragon Game Master. 

ORIGINAL QUESTION:
============
{question}
============
                                                        
CRITIQUE OF THE ANSWER:
============
{critique}
============

YOUR TASK:
Based on the critique above, revise the ORIGINAL QUESTION to make it better.
The revised question should address the issues identified in the critique.

CRITICAL INSTRUCTIONS:
1. Output EXACTLY ONE revised question
2. Output ONLY the question text itself (no labels, no alternatives, no explanations)
3. Make it clear, succinct, and easy to understand
4. Ensure it will elicit a better answer that addresses the critique's concerns

EXAMPLE INPUT:
Question: "Tell me about rogues"
Critique: "Answer lacks specificity about class features"

"""

def get_reflection_assessment (answer:str, critique:str):
    return f"""Evaluated Text: {answer}
            Critique: {critique} 
            YOUR TASK:
Based on the critique above, revise the ORIGINAL QUESTION to make it better.
The revised question should address the issues identified in the critique.

CRITICAL INSTRUCTIONS:
1. Output EXACTLY ONE revised question
2. Output ONLY the question text itself (no labels, no alternatives, no explanations)
3. Make it clear, succinct, and easy to understand
4. Ensure it will elicit a better answer that addresses the critique's concerns

EXAMPLE INPUT:
Question: "Tell me about rogues"
Critique: "Answer lacks specificity about class features"

EXAMPLE OUTPUT:
What are the key class features and abilities of the Rogue class in D&D?

NOW OUTPUT YOUR REVISED QUESTION (question text only, nothing else):

"""

"""
Post-processing validation handles revision_needed logic programmatically.
The LLM prompt has been simplified because:
- LLMs cannot reliably self-check their own output (no true self-reflection)
- Autoregressive generation means earlier tokens don't influence later fields strongly
- Training bias toward "balanced feedback" patterns cannot be overridden by prompting alone
- Code validation enforces logical consistency that prompts cannot guarantee
"""

reflect_on_answer :str = """You are a D&D expert evaluating a response.

    CONTEXT: {context}
    USER'S QUERY: {question}
                                                      
    TASK: Evaluate the response and provide a critique.
    
    RUBRICS:
    • clarity: Is the explanation easy to understand? (true/false)
    • succinct: Is it concise yet comprehensive? (true/false)  
    • readabilty: Is it at a 5th grade reading level? (true/false)
    • revision_needed: Does it need improvement? (true/false)
    
    Provide a thoughtful critique covering:
    - Strengths and weaknesses of the response
    - Specific areas that could be improved (if any)
    - Suggestions for enhancing clarity, conciseness, or readability (if applicable)
    
    Be honest in your evaluation. If the response has issues or could be improved, 
    explain what changes would help.
    
    Output ONLY valid JSON with these fields:
    - critique (string)
    - clarity (boolean)
    - succinct (boolean)
    - readabilty (boolean)
    - revision_needed (boolean)
    - response (object with answer, question, source, context_summary)
"""