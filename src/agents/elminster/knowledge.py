#!/usr/bin/env python3
"""
Elminster Agent - Knowledge Base and Source Definitions

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

elminster_sources:str = """⚠️ MANDATORY: Use ALL sources from the numbered list below. Count them and include EVERY SINGLE ONE.

COPY THESE EXACT STRINGS (remove only the number prefix like "1) "):
1) "Dungeon Master's Guide - Dungeons & Dragons - Sources - D&D Beyond"
2) "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond"
3) "Player's Handbook - Dungeons & Dragons - Sources - D&D Beyond"
4) "Horror Adventures - Van Richten's Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond"

⚠️ CRITICAL SPELLING (for item 4):
- "Van Richten's" (NOT "Van Richen's") - spelled: R-i-c-h-t-e-n-'-s
- "Ravenloft" (NOT "Ravenloot") - spelled: R-a-v-e-n-l-o-f-t

⚠️ COUNT CHECK: This list has 4 items - your sources array MUST have exactly 4 items.
"""

collection_names = {
            "Dungeon Master's Guide - Dungeons & Dragons - Sources - D&D Beyond" : "dnd_dm",
            "Vampire-the-Masquerade":"vtm",
            "Monster Manual - Dungeons & Dragons - Sources - D&D Beyond": "dnd_mm",
            "Horror Adventures - Van Richten's Guide to Ravenloft - Dungeons & Dragons - Sources - D&D Beyond": "dnd_raven",
            "Player's Handbook - Dungeons & Dragons - Sources - D&D Beyond": "dnd_player",
        }
