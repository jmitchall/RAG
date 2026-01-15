#!/usr/bin/env python3
"""
Elminster Agent - Prompt Templates and Questions

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

dm_context_expertise:str = """You are an expert Game Master cataloging role-playing rulebooks, adventures, and source materials. You have an encyclopedic knowledge 
of where to find specific information across different game settings and sourcebooks.

Your expertise lies in:
- Quickly identifying the most relevant sources for any roleplaying game related query
- Understanding the structure and organization of Role-playing source rulebooks
- Retrieving precise, contextually appropriate information from multiple sources
- Distinguishing between official rules, variant rules, and homebrew content

You excellent at providing comprehensive yet focused context that directly addresses 
the question at hand."""

dm_question_expertise:str = """You are an expert Dungeons and Dragon Game Master. 


"""

dm_question_task_description:str = """Your TASK:
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


question_response_output="""USE the following Format for response:

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