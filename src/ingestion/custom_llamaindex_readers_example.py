"""
Examples: Creating Custom LlamaIndex File Readers

This demonstrates how to create custom readers that can be used with
SimpleDirectoryReader's file_extractor parameter.

Custom readers must implement load_data() method and return List[Document].
"""

import os
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document


# ============================================================================
# Example 1: Custom Text Reader with Preprocessing
# ============================================================================

class CustomTextReader(BaseReader):
    """
    Custom text reader that preprocesses text (removes extra whitespace,
    extracts metadata from headers, etc.)
    """
    
    def __init__(
        self,
        remove_extra_whitespace: bool = True,
        extract_metadata_from_header: bool = True,
        encoding: str = "utf-8"
    ):
        """
        Initialize the custom text reader.
        
        Args:
            remove_extra_whitespace: Clean up extra spaces/newlines
            extract_metadata_from_header: Extract metadata from file header
            encoding: File encoding
        """
        self.remove_extra_whitespace = remove_extra_whitespace
        self.extract_metadata_from_header = extract_metadata_from_header
        self.encoding = encoding
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load data from a text file.
        
        Args:
            file: Path to the file (provided by SimpleDirectoryReader)
            extra_info: Additional metadata (provided by SimpleDirectoryReader)
            
        Returns:
            List[Document]: List containing one preprocessed document
        """
        with open(file, encoding=self.encoding) as f:
            text = f.read()
        
        # Preprocess text
        if self.remove_extra_whitespace:
            # Remove multiple consecutive newlines
            import re
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
            # Remove trailing whitespace from lines
            text = '\n'.join(line.rstrip() for line in text.splitlines())
        
        # Extract metadata
        metadata = extra_info or {}
        metadata.update({
            'file_path': str(file),
            'file_name': file.name,
            'file_size': file.stat().st_size,
        })
        
        if self.extract_metadata_from_header:
            header_metadata = self._extract_header_metadata(text)
            metadata.update(header_metadata)
        
        return [Document(text=text, metadata=metadata)]
    
    def _extract_header_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from header lines (e.g., # KEY: value)."""
        metadata = {}
        lines = text.splitlines()
        
        for line in lines[:20]:  # Check first 20 lines
            if line.strip().startswith('#'):
                line = line.strip('#').strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip().lower().replace(' ', '_')] = value.strip()
        
        return metadata


# ============================================================================
# Example 2: Custom CSV Reader (Creates Multiple Documents)
# ============================================================================

class CustomCSVReader(BaseReader):
    """
    Custom CSV reader that creates one Document per row.
    Each row becomes a separate document in the index.
    """
    
    def __init__(
        self,
        content_columns: List[str],
        concat_rows: bool = False,
        delimiter: str = ',',
        encoding: str = 'utf-8'
    ):
        """
        Initialize CSV reader.
        
        Args:
            content_columns: Columns to combine into document text
            concat_rows: If True, concatenate all rows into one doc; if False, one doc per row
            delimiter: CSV delimiter
            encoding: File encoding
        """
        self.content_columns = content_columns
        self.concat_rows = concat_rows
        self.delimiter = delimiter
        self.encoding = encoding
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load data from CSV file.
        
        Returns:
            List[Document]: One document per row (or one combined document if concat_rows=True)
        """
        documents = []
        
        with open(file, encoding=self.encoding) as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            
            all_text_parts = []
            
            for row_idx, row in enumerate(reader):
                # Combine content columns
                content_parts = []
                for col in self.content_columns:
                    if col in row and row[col]:
                        content_parts.append(f"{col}: {row[col]}")
                
                text = "\n".join(content_parts)
                
                if self.concat_rows:
                    all_text_parts.append(text)
                else:
                    # Create one document per row
                    metadata = extra_info.copy() if extra_info else {}
                    metadata.update({
                        'file_path': str(file),
                        'file_name': file.name,
                        'row_index': row_idx,
                        **{k: v for k, v in row.items() if k not in self.content_columns}
                    })
                    
                    documents.append(Document(text=text, metadata=metadata))
            
            # If concatenating all rows into one document
            if self.concat_rows:
                metadata = extra_info.copy() if extra_info else {}
                metadata.update({
                    'file_path': str(file),
                    'file_name': file.name,
                    'row_count': len(all_text_parts),
                })
                
                combined_text = "\n\n".join(all_text_parts)
                documents.append(Document(text=combined_text, metadata=metadata))
        
        return documents


# ============================================================================
# Example 3: Custom JSON Reader with Nested Structure Support
# ============================================================================

class CustomJSONReader(BaseReader):
    """
    Custom JSON reader that can extract specific fields from JSON structure.
    Useful for structured JSON data.
    """
    
    def __init__(
        self,
        json_path: str = ".",
        text_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
        is_jsonl: bool = False
    ):
        """
        Initialize JSON reader.
        
        Args:
            json_path: Path within JSON to extract (e.g., "data.items" or "." for root)
            text_field: Field name containing text content
            metadata_fields: Fields to extract as metadata
            is_jsonl: Whether file is JSONL (one JSON object per line)
        """
        self.json_path = json_path
        self.text_field = text_field
        self.metadata_fields = metadata_fields or []
        self.is_jsonl = is_jsonl
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from JSON file."""
        documents = []
        
        with open(file, encoding='utf-8') as f:
            if self.is_jsonl:
                # JSONL format: one JSON object per line
                for line_idx, line in enumerate(f):
                    if line.strip():
                        obj = json.loads(line)
                        doc = self._create_document_from_json(obj, file, extra_info, line_idx)
                        if doc:
                            documents.append(doc)
            else:
                # Regular JSON
                data = json.load(f)
                
                # Navigate to specified path
                items = self._extract_items_from_path(data)
                
                for idx, item in enumerate(items):
                    doc = self._create_document_from_json(item, file, extra_info, idx)
                    if doc:
                        documents.append(doc)
        
        return documents
    
    def _extract_items_from_path(self, data: Any) -> List[Any]:
        """Extract items from JSON based on path."""
        if self.json_path == ".":
            # Root level
            if isinstance(data, list):
                return data
            else:
                return [data]
        
        # Navigate path (e.g., "data.items")
        path_parts = self.json_path.split('.')
        current = data
        
        for part in path_parts:
            if isinstance(current, dict):
                current = current.get(part, current)
            else:
                break
        
        if isinstance(current, list):
            return current
        else:
            return [current]
    
    def _create_document_from_json(
        self,
        item: Dict,
        file: Path,
        extra_info: Optional[Dict],
        index: int
    ) -> Optional[Document]:
        """Create document from JSON item."""
        # Extract text content
        text = item.get(self.text_field, "")
        if not text:
            # If text field not found, stringify the entire object
            text = json.dumps(item, indent=2)
        
        # Build metadata
        metadata = extra_info.copy() if extra_info else {}
        metadata.update({
            'file_path': str(file),
            'file_name': file.name,
            'item_index': index,
        })
        
        # Extract specified metadata fields
        for field in self.metadata_fields:
            if field in item:
                metadata[field] = item[field]
        
        return Document(text=str(text), metadata=metadata)


# ============================================================================
# Example 4: Custom Markdown Reader with Section Extraction
# ============================================================================

class CustomMarkdownReader(BaseReader):
    """
    Custom Markdown reader that can optionally split by headers.
    """
    
    def __init__(
        self,
        split_by_headers: bool = False,
        header_levels: Optional[List[int]] = None,
        remove_code_blocks: bool = False
    ):
        """
        Initialize Markdown reader.
        
        Args:
            split_by_headers: Split document by header sections
            header_levels: Which header levels to split on (e.g., [1, 2])
            remove_code_blocks: Remove code blocks from content
        """
        self.split_by_headers = split_by_headers
        self.header_levels = header_levels or [1, 2]
        self.remove_code_blocks = remove_code_blocks
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from Markdown file."""
        with open(file, encoding='utf-8') as f:
            content = f.read()
        
        # Optionally remove code blocks
        if self.remove_code_blocks:
            import re
            content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        
        if self.split_by_headers:
            return self._split_by_headers(content, file, extra_info)
        else:
            metadata = extra_info.copy() if extra_info else {}
            metadata.update({
                'file_path': str(file),
                'file_name': file.name,
            })
            return [Document(text=content, metadata=metadata)]
    
    def _split_by_headers(
        self,
        content: str,
        file: Path,
        extra_info: Optional[Dict]
    ) -> List[Document]:
        """Split content by markdown headers."""
        import re
        
        documents = []
        current_section = {'header': 'Introduction', 'level': 0, 'content': ''}
        
        for line in content.splitlines():
            # Check for header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                # Check if this level should split
                if level in self.header_levels:
                    # Save previous section
                    if current_section['content'].strip():
                        metadata = extra_info.copy() if extra_info else {}
                        metadata.update({
                            'file_path': str(file),
                            'file_name': file.name,
                            'section_header': current_section['header'],
                            'header_level': current_section['level'],
                        })
                        documents.append(Document(
                            text=current_section['content'],
                            metadata=metadata
                        ))
                    
                    # Start new section
                    current_section = {
                        'header': header_text,
                        'level': level,
                        'content': line + '\n'
                    }
                else:
                    current_section['content'] += line + '\n'
            else:
                current_section['content'] += line + '\n'
        
        # Add last section
        if current_section['content'].strip():
            metadata = extra_info.copy() if extra_info else {}
            metadata.update({
                'file_path': str(file),
                'file_name': file.name,
                'section_header': current_section['header'],
                'header_level': current_section['level'],
            })
            documents.append(Document(
                text=current_section['content'],
                metadata=metadata
            ))
        
        return documents


# ============================================================================
# Example 5: Custom Log File Reader
# ============================================================================

class CustomLogReader(BaseReader):
    """
    Custom reader for log files that can filter by log level,
    combine entries, and extract structured information.
    """
    
    def __init__(
        self,
        min_log_level: str = "INFO",
        group_by_timestamp: bool = False,
        max_lines: Optional[int] = None
    ):
        """
        Initialize log reader.
        
        Args:
            min_log_level: Minimum log level to include (DEBUG, INFO, WARNING, ERROR)
            group_by_timestamp: Group log entries by time windows
            max_lines: Maximum number of lines to read
        """
        self.min_log_level = min_log_level
        self.group_by_timestamp = group_by_timestamp
        self.max_lines = max_lines
        
        self.level_priority = {
            'DEBUG': 0,
            'INFO': 1,
            'WARNING': 2,
            'ERROR': 3,
            'CRITICAL': 4
        }
    
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Load data from log file."""
        import re
        
        log_entries = []
        min_priority = self.level_priority.get(self.min_log_level, 0)
        
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.max_lines and idx >= self.max_lines:
                    break
                
                # Parse log line (assuming format: LEVEL: message)
                level_match = re.match(r'^(\w+):', line)
                if level_match:
                    level = level_match.group(1)
                    if self.level_priority.get(level, 0) >= min_priority:
                        log_entries.append(line)
                else:
                    # Continuation line
                    if log_entries:
                        log_entries[-1] += line
        
        # Combine all entries into one document
        combined_text = ''.join(log_entries)
        
        metadata = extra_info.copy() if extra_info else {}
        metadata.update({
            'file_path': str(file),
            'file_name': file.name,
            'log_entries_count': len(log_entries),
            'min_log_level': self.min_log_level,
        })
        
        return [Document(text=combined_text, metadata=metadata)]


# ============================================================================
# Demonstration: Using Custom Readers with SimpleDirectoryReader
# ============================================================================

def demonstrate_custom_readers():
    """Demonstrate using custom readers with SimpleDirectoryReader."""
    
    # Create test directory and files
    test_dir = "/tmp/llamaindex_custom_readers"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create sample files
    print("📁 Creating sample files...")
    
    # 1. Custom text file
    with open(os.path.join(test_dir, "sample.txt"), "w") as f:
        f.write("""# TITLE: Machine Learning Guide
# AUTHOR: AI Assistant
# VERSION: 1.0

This is a comprehensive guide to machine learning.


It covers various topics and techniques.
""")
    
    # 2. CSV file
    with open(os.path.join(test_dir, "data.csv"), "w") as f:
        f.write("""title,content,category
Document 1,Content about machine learning,Tech
Document 2,Content about data science,Tech
Document 3,Content about AI ethics,Ethics
""")
    
    # 3. JSON file
    with open(os.path.join(test_dir, "articles.json"), "w") as f:
        json.dump({
            "articles": [
                {"id": 1, "text": "First article about AI", "author": "Alice"},
                {"id": 2, "text": "Second article about ML", "author": "Bob"},
            ]
        }, f)
    
    # 4. Markdown file
    with open(os.path.join(test_dir, "guide.md"), "w") as f:
        f.write("""# Introduction

This is the introduction.

## Overview

This is the overview section.

# Main Content

This is the main content section.
""")
    
    # 5. Log file
    with open(os.path.join(test_dir, "app.log"), "w") as f:
        f.write("""DEBUG: Starting application
INFO: Application started successfully
WARNING: High memory usage detected
ERROR: Connection timeout
INFO: Retrying connection
""")
    
    print("✅ Sample files created\n")
    
    # ========================================================================
    # Example: Using custom readers with SimpleDirectoryReader
    # ========================================================================
    
    print("=" * 80)
    print("Using Custom Readers with SimpleDirectoryReader")
    print("=" * 80)
    
    # Define file extractors (map extensions to custom readers)
    file_extractor = {
        ".txt": CustomTextReader(
            remove_extra_whitespace=True,
            extract_metadata_from_header=True
        ),
        ".csv": CustomCSVReader(
            content_columns=['title', 'content'],
            concat_rows=False  # One document per row
        ),
        ".json": CustomJSONReader(
            json_path="articles",
            text_field="text",
            metadata_fields=["id", "author"]
        ),
        ".md": CustomMarkdownReader(
            split_by_headers=True,
            header_levels=[1, 2]
        ),
        ".log": CustomLogReader(
            min_log_level="INFO",
            max_lines=100
        )
    }
    
    # Use SimpleDirectoryReader with custom readers
    print("\n🔍 Loading documents with custom readers...")
    reader = SimpleDirectoryReader(
        input_dir=test_dir,
        file_extractor=file_extractor,
        recursive=False
    )
    
    documents = reader.load_data()
    
    print(f"\n✅ Loaded {len(documents)} documents total\n")
    
    # Display results grouped by file type
    from collections import defaultdict
    docs_by_type = defaultdict(list)
    
    for doc in documents:
        file_name = doc.metadata.get('file_name', 'unknown')
        ext = Path(file_name).suffix
        docs_by_type[ext].append(doc)
    
    for ext, docs in sorted(docs_by_type.items()):
        print(f"\n{'=' * 80}")
        print(f"📄 {ext.upper()} Files: {len(docs)} document(s)")
        print(f"{'=' * 80}")
        
        for idx, doc in enumerate(docs, 1):
            print(f"\n--- Document {idx} ---")
            print(f"File: {doc.metadata.get('file_name', 'N/A')}")
            
            # Show relevant metadata
            for key, value in doc.metadata.items():
                if key not in ['file_path', 'file_name']:
                    print(f"{key}: {value}")
            
            print(f"\nText preview ({len(doc.text)} chars):")
            print(f"{doc.text[:150]}...")
    
    print("\n" + "=" * 80)
    print("✅ Demonstration Complete!")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   1. Custom readers must implement load_data(file, extra_info)")
    print("   2. Return List[Document] with text and metadata")
    print("   3. Use file_extractor dict to map extensions to readers")
    print("   4. Each reader can have its own configuration")
    print("   5. Readers can create multiple documents from one file")


# ============================================================================
# Simplified Example: How to Use in Your Code
# ============================================================================

def simple_usage_example():
    """
    Simple example showing how to use custom readers in your existing code.
    """
    print("\n" + "=" * 80)
    print("SIMPLE USAGE EXAMPLE")
    print("=" * 80)
    
    directory_path = "/tmp/llamaindex_custom_readers"
    
    # Define your custom file extractors
    file_extractor = {
        ".txt": CustomTextReader(extract_metadata_from_header=True),
        ".csv": CustomCSVReader(content_columns=['title', 'content']),
        ".json": CustomJSONReader(json_path="articles", text_field="text"),
        ".md": CustomMarkdownReader(split_by_headers=True),
        ".log": CustomLogReader(min_log_level="INFO"),
    }
    
    # Use with SimpleDirectoryReader (just like in your code)
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        required_exts=[".txt", ".csv", ".json", ".md", ".log"],
        recursive=True,
        file_extractor=file_extractor
    )
    
    documents = reader.load_data()
    
    print(f"\n✅ Loaded {len(documents)} documents")
    print(f"   File types: {set(Path(doc.metadata.get('file_name', '')).suffix for doc in documents)}")
    
    return documents


if __name__ == "__main__":
    # Run full demonstration
    demonstrate_custom_readers()
    
    # Show simple usage
    simple_usage_example()
