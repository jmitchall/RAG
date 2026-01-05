"""
Examples: Creating Custom LangChain Document Loaders

This demonstrates how to create your own langchain_community.document_loaders
by inheriting from BaseLoader and implementing the lazy_load method.

Custom loaders are useful for:
- Unsupported file formats
- Special preprocessing requirements
- Complex metadata extraction
- Custom parsing logic


https://medium.com/data-science/improved-rag-document-processing-with-markdown-426a2e0dd82b
pip install docling==2.5.2
from docling.document_converter import DocumentConverter
converter = DocumentConverter()
result = converter.convert(FILE)
docling_text = result.document.export_to_markdown()

After reading our PDF document and converting it to Markdown, we can use LangChain’s RecursiveCharacter TextSplitter to chunk according to specific Markdown syntax.

LangChain defines these default separators in Language.MARKDOWN:

separators = [
    # First, try to split along Markdown headings (starting with level 2)
    "\n#{1,6} ",
    # End of code block
    "```\n",
    # Horizontal lines
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
pip install langchain-text-splitters==0.3.2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=1000,
    chunk_overlap=100,
)
documents = text_splitter.create_documents(texts=[docling_text])


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language
from docling.document_converter import DocumentConverter


def process_file(filename: str, metadata: str = None):
    '''read file, convert to markdown, split into chunks and optionally add metadata'''

    # read file and export to markdown
    converter = DocumentConverter()
    result = converter.convert(filename)
    docling_text = result.document.export_to_markdown()

    # chunk document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=100,
    )

    docling_documents = text_splitter.create_documents(texts=[docling_text])

    if metadata:
        for doc in docling_documents:
            doc.page_content = "\n".join([metadata, doc.page_content])
    return docling_documents


    class MyCustomLoader(BaseLoader):
        def __init__(self, file_path: str, **kwargs):
            self.file_path = file_path
            # Store parameters
        
        def lazy_load(self) -> Iterator[Document]:
            # Load and process file
            # Yield Document(page_content=..., metadata=...)    



"""

import os
from typing import Iterator, List, Dict, Any, Optional
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document


# ============================================================================
# Example 1: Simple Custom Text Loader with Custom Metadata
# ============================================================================

class CustomTextLoader(BaseLoader):
    """
    Custom text loader that extracts additional metadata like word count,
    line count, and custom tags from file headers.
    """
    
    def __init__(
        self,
        file_path: str,
        encoding: str = "utf-8",
        extract_tags: bool = True
    ):
        """
        Initialize the custom text loader.
        
        Args:
            file_path: Path to the text file
            encoding: File encoding (default: utf-8)
            extract_tags: Whether to extract custom tags from file headers
        """
        self.file_path = file_path
        self.encoding = encoding
        self.extract_tags = extract_tags
    
    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load the document. This is called by LangChain when loading.
        Yields documents one at a time for memory efficiency.
        """
        with open(self.file_path, encoding=self.encoding) as f:
            text = f.read()
        
        # Extract metadata
        metadata = self._extract_metadata(text)
        
        # Create and yield document
        yield Document(
            page_content=text,
            metadata=metadata
        )
    
    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract custom metadata from the text."""
        metadata = {
            'source': self.file_path,
            'filename': os.path.basename(self.file_path),
            'word_count': len(text.split()),
            'line_count': len(text.splitlines()),
            'char_count': len(text),
        }
        
        # Extract custom tags from header (e.g., # TAG: value)
        if self.extract_tags:
            tags = {}
            for line in text.splitlines()[:10]:  # Check first 10 lines
                if line.strip().startswith('#'):
                    line = line.strip('#').strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        tags[key.strip().lower()] = value.strip()
            
            if tags:
                metadata['custom_tags'] = tags
        
        return metadata


# ============================================================================
# Example 2: Multi-File Loader (Load Multiple Documents)
# ============================================================================

class DirectoryBatchLoader(BaseLoader):
    """
    Custom loader that loads all files from a directory with a specific extension.
    Creates one document per file.
    """
    
    def __init__(
        self,
        directory_path: str,
        file_extension: str = ".txt",
        recursive: bool = False,
        max_files: Optional[int] = None
    ):
        """
        Initialize the directory batch loader.
        
        Args:
            directory_path: Path to directory
            file_extension: File extension to match (default: .txt)
            recursive: Whether to search subdirectories
            max_files: Maximum number of files to load (None = all)
        """
        self.directory_path = directory_path
        self.file_extension = file_extension
        self.recursive = recursive
        self.max_files = max_files
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents from directory."""
        file_count = 0
        pattern = "**/*" if self.recursive else "*"
        
        for file_path in Path(self.directory_path).glob(pattern + self.file_extension):
            if self.max_files and file_count >= self.max_files:
                break
            
            if file_path.is_file():
                try:
                    with open(file_path, encoding='utf-8') as f:
                        text = f.read()
                    
                    metadata = {
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_size': file_path.stat().st_size,
                        'modified_time': file_path.stat().st_mtime,
                    }
                    
                    yield Document(page_content=text, metadata=metadata)
                    file_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {file_path}: {e}")
                    continue


# ============================================================================
# Example 3: Custom CSV Loader with Row-Level Documents
# ============================================================================

class CustomCSVLoader(BaseLoader):
    """
    Custom CSV loader that creates one document per row.
    Useful for structured data where each row represents a separate entity.
    """
    
    def __init__(
        self,
        file_path: str,
        content_columns: List[str],
        metadata_columns: Optional[List[str]] = None,
        delimiter: str = ',',
        skip_header: bool = False
    ):
        """
        Initialize the custom CSV loader.
        
        Args:
            file_path: Path to CSV file
            content_columns: Columns to combine into page_content
            metadata_columns: Columns to store as metadata
            delimiter: CSV delimiter
            skip_header: Whether first row is header
        """
        self.file_path = file_path
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.delimiter = delimiter
        self.skip_header = skip_header
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load CSV rows as documents."""
        import csv
        
        with open(self.file_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            
            if self.skip_header:
                next(reader)  # Skip header row
            
            for row_idx, row in enumerate(reader):
                # Combine content columns
                content_parts = [
                    f"{col}: {row.get(col, '')}" 
                    for col in self.content_columns 
                    if col in row
                ]
                page_content = "\n".join(content_parts)
                
                # Extract metadata
                metadata = {
                    'source': self.file_path,
                    'row_number': row_idx,
                }
                
                for col in self.metadata_columns:
                    if col in row:
                        metadata[col] = row[col]
                
                yield Document(page_content=page_content, metadata=metadata)


# ============================================================================
# Example 4: Preprocessed Markdown Loader with Section Splitting
# ============================================================================

class MarkdownSectionLoader(BaseLoader):
    """
    Custom loader that splits markdown by headers and creates one document per section.
    Preserves header hierarchy in metadata.
    """
    
    def __init__(
        self,
        file_path: str,
        min_header_level: int = 1,
        max_header_level: int = 3
    ):
        """
        Initialize markdown section loader.
        
        Args:
            file_path: Path to markdown file
            min_header_level: Minimum header level to split on (1 = #)
            max_header_level: Maximum header level to split on (3 = ###)
        """
        self.file_path = file_path
        self.min_header_level = min_header_level
        self.max_header_level = max_header_level
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load markdown sections as separate documents."""
        with open(self.file_path, encoding='utf-8') as f:
            content = f.read()
        
        sections = self._split_by_headers(content)
        
        for section_idx, section in enumerate(sections):
            yield Document(
                page_content=section['content'],
                metadata={
                    'source': self.file_path,
                    'section_index': section_idx,
                    'header': section['header'],
                    'header_level': section['level'],
                }
            )
    
    def _split_by_headers(self, content: str) -> List[Dict[str, Any]]:
        """Split content by markdown headers."""
        import re
        
        sections = []
        current_section = {'header': 'Introduction', 'level': 0, 'content': ''}
        
        for line in content.splitlines():
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                # Check if this header level should trigger a split
                if self.min_header_level <= level <= self.max_header_level:
                    # Save previous section if it has content
                    if current_section['content'].strip():
                        sections.append(current_section)
                    
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
            sections.append(current_section)
        
        return sections


# ============================================================================
# Example 5: JSON Loader with Nested Structure Support
# ============================================================================

class CustomJSONLoader(BaseLoader):
    """
    Custom JSON loader that can extract nested fields and create documents
    from JSON arrays or objects.
    """
    
    def __init__(
        self,
        file_path: str,
        jq_schema: str = ".",
        content_key: str = "text",
        metadata_keys: Optional[List[str]] = None
    ):
        """
        Initialize custom JSON loader.
        
        Args:
            file_path: Path to JSON file
            jq_schema: JQ-like path to extract data (e.g., ".items[]")
            content_key: Key to use for page_content
            metadata_keys: Keys to extract as metadata
        """
        self.file_path = file_path
        self.jq_schema = jq_schema
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
    
    def lazy_load(self) -> Iterator[Document]:
        """Lazy load JSON data as documents."""
        import json
        
        with open(self.file_path, encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract items based on schema
        items = self._extract_items(data)
        
        for idx, item in enumerate(items):
            # Extract content
            content = item.get(self.content_key, str(item))
            
            # Extract metadata
            metadata = {
                'source': self.file_path,
                'item_index': idx,
            }
            
            for key in self.metadata_keys:
                if key in item:
                    metadata[key] = item[key]
            
            yield Document(page_content=content, metadata=metadata)
    
    def _extract_items(self, data: Any) -> List[Dict]:
        """Extract items from JSON based on schema."""
        if self.jq_schema == ".":
            # Root level
            if isinstance(data, list):
                return data
            else:
                return [data]
        
        # Simple path extraction (e.g., ".items", ".data.records")
        path_parts = self.jq_schema.strip('.').split('.')
        current = data
        
        for part in path_parts:
            if part.endswith('[]'):
                # Array extraction
                key = part[:-2]
                current = current.get(key, [])
            else:
                current = current.get(part, current)
        
        if isinstance(current, list):
            return current
        else:
            return [current]


# ============================================================================
# Demonstration
# ============================================================================

def demonstrate_custom_loaders():
    """Demonstrate all custom loaders."""
    
    # Create test directory
    test_dir = "/tmp/custom_loader_demo"
    os.makedirs(test_dir, exist_ok=True)
    
    # Example 1: Custom Text Loader
    print("=" * 80)
    print("Example 1: Custom Text Loader with Metadata Extraction")
    print("=" * 80)
    
    text_file = os.path.join(test_dir, "sample.txt")
    with open(text_file, "w") as f:
        f.write("""# TITLE: Machine Learning Basics
# AUTHOR: AI Assistant
# CATEGORY: Education

This is a document about machine learning fundamentals.
It includes various concepts and examples.
The custom loader will extract metadata from the header tags.
""")
    
    loader = CustomTextLoader(text_file, extract_tags=True)
    docs = list(loader.lazy_load())
    
    print(f"✅ Loaded {len(docs)} document(s)")
    for doc in docs:
        print(f"\nMetadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:100]}...")
    
    # Example 2: Directory Batch Loader
    print("\n" + "=" * 80)
    print("Example 2: Directory Batch Loader")
    print("=" * 80)
    
    # Create multiple files
    for i in range(3):
        with open(os.path.join(test_dir, f"file_{i}.txt"), "w") as f:
            f.write(f"This is content for file {i}.\nIt contains important information.")
    
    loader = DirectoryBatchLoader(test_dir, file_extension=".txt")
    docs = list(loader.lazy_load())
    
    print(f"✅ Loaded {len(docs)} document(s) from directory")
    for doc in docs:
        print(f"\n📄 {doc.metadata['filename']}: {len(doc.page_content)} chars")
    
    # Example 3: Custom CSV Loader
    print("\n" + "=" * 80)
    print("Example 3: Custom CSV Loader (Row-Level Documents)")
    print("=" * 80)
    
    csv_file = os.path.join(test_dir, "products.csv")
    with open(csv_file, "w") as f:
        f.write("""name,description,price,category
Widget,A useful widget,19.99,Tools
Gadget,An amazing gadget,29.99,Electronics
Doohickey,A mysterious doohickey,9.99,Misc
""")
    
    loader = CustomCSVLoader(
        csv_file,
        content_columns=['name', 'description'],
        metadata_columns=['price', 'category']
    )
    docs = list(loader.lazy_load())
    
    print(f"✅ Loaded {len(docs)} document(s) from CSV")
    for doc in docs[:2]:  # Show first 2
        print(f"\nContent:\n{doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Example 4: Markdown Section Loader
    print("\n" + "=" * 80)
    print("Example 4: Markdown Section Loader")
    print("=" * 80)
    
    md_file = os.path.join(test_dir, "guide.md")
    with open(md_file, "w") as f:
        f.write("""# Introduction

This is the introduction section.

## Overview

This is the overview subsection.

## Features

Key features include:
- Feature 1
- Feature 2

# Setup

This is the setup section with installation instructions.
""")
    
    loader = MarkdownSectionLoader(md_file, min_header_level=1, max_header_level=2)
    docs = list(loader.lazy_load())
    
    print(f"✅ Loaded {len(docs)} section(s) from markdown")
    for doc in docs:
        print(f"\n📑 Section: {doc.metadata['header']} (Level {doc.metadata['header_level']})")
        print(f"   Content length: {len(doc.page_content)} chars")
    
    # Example 5: JSON Loader
    print("\n" + "=" * 80)
    print("Example 5: Custom JSON Loader")
    print("=" * 80)
    
    json_file = os.path.join(test_dir, "articles.json")
    with open(json_file, "w") as f:
        import json
        json.dump({
            "articles": [
                {"id": 1, "text": "First article content", "author": "Alice", "tags": ["tech", "ai"]},
                {"id": 2, "text": "Second article content", "author": "Bob", "tags": ["science"]},
                {"id": 3, "text": "Third article content", "author": "Charlie", "tags": ["tech"]},
            ]
        }, f)
    
    loader = CustomJSONLoader(
        json_file,
        jq_schema=".articles[]",
        content_key="text",
        metadata_keys=["id", "author", "tags"]
    )
    docs = list(loader.lazy_load())
    
    print(f"✅ Loaded {len(docs)} document(s) from JSON")
    for doc in docs:
        print(f"\n📰 {doc.page_content}")
        print(f"   Metadata: {doc.metadata}")
    
    print("\n" + "=" * 80)
    print("✅ All custom loader demonstrations complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_custom_loaders()
