#!/usr/bin/env python3
"""
Elminster Agent - Context Retrieval Prompts

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

from agents.elminster.tools.refresh_question_context_tool import RefreshQuestionContextTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional


def get_context_retrieval_prompt_tools():
    return ChatPromptTemplate.from_messages([
        ("human", "{input} "), MessagesPlaceholder(variable_name="messages")]), [RefreshQuestionContextTool()]


def get_tool_input_request(question: str, sources: str, db_type: str, root_path: str, critique: Optional[str] = None):
    return f"""Call refresh_question_context tool with these parameters:

🔴 MANDATORY: You MUST include ALL sources from the numbered list below - regardless of how many there are (1, 2, 3, 4, or more).
🔴 DO NOT select a subset. DO NOT omit any numbered items. USE EVERY SINGLE ONE.

REQUIRED PARAMETERS:
- question: "{question}"
- root_path: "{root_path}"
- db_type: "{db_type}"
- sources: ⚠️ COPY ALL items from this numbered list into your sources array:
{sources}
- critique: "{critique}"
- improved_question: <An improved version of the original question based on critique>

EXAMPLE 1 - Single source:
TOOL_CALL: {{
            "name": "refresh_question_context",
            "arguments": 
                    {{
                        "question": "What is a Warlock"", 
                        "root_path": "/home/jmitchall/vllm-srv", 
                        "db_type": "qdrant", 
                        "sources": ["Player's Handbook - Dungeons & Dragons - Sources - D&D Beyond"], 
                        "critique": "The question could be improved by providing a source and explict details on what aspects of the Warlock class are of interest.", 
                        "improved_question": "What are all the detailed class features, abilities, and progression options available to Warlocks?"
                    }}
             }}

EXAMPLE 2 - Two sources (BOTH must be included):
TOOL_CALL: {{
            "name": "refresh_question_context",
            "arguments": 
                    {{
                        "question": "What creatures inhabit Ravenloft?",
                        "root_path": "/home/jmitchall/vllm-srv", 
                        "db_type": "qdrant", 
                        "sources": [
                            "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond",
                            "Horror Adventures - Van Richten's Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond"
                        ], 
                        "critique": "", 
                        "improved_question": ""
                    }}
             }}
            
EXAMPLE 3 - Four sources (ALL 4 must be included - NOTE CORRECT SPELLING: Van Richten's, Ravenloft):
TOOL_CALL: {{
            "name": "refresh_question_context",
            "arguments": 
                    {{
                        "question": "Who is Strahd von Zarovich?",
                        "root_path": "/home/jmitchall/vllm-srv", 
                        "db_type": "qdrant", 
                        "sources": [
                            "Dungeon Master's Guide - Dungeons & Dragons - Sources - D&D Beyond",
                            "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond",
                            "Player's Handbook - Dungeons & Dragons - Sources - D&D Beyond",
                            "Horror Adventures - Van Richten's Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond"
                        ], 
                        "critique": "", 
                        "improved_question": ""
                    }}
             }}

✓ VALIDATION CHECKLIST before submitting your tool call:
  1. Count the numbered items in the sources list above - your sources array MUST have EXACTLY that many items
  2. Each source string must be an EXACT character-by-character copy (strip only the number and closing parenthesis)
  3. Check spelling: "Van Richten's" (NOT "Van Richen's") and "Ravenloft" (NOT "Ravenloot")
  4. Confirm all apostrophes, hyphens, and spaces are preserved exactly
  5. Do not abbreviate, modify, or "fix" any text
  6. If the list has 2 items, your array has 2. If it has 4 items, your array has 4. Always match the count.

"""
