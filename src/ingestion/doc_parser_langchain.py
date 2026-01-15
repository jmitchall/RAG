#!/usr/bin/env python3
"""
LangChain Document Parser - Document Loading and Chunking with LangChain

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

import glob
import os
from ingestion.base_document_chunker import BaseDocumentChunker
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
)
from typing import List, Tuple
from refection_logger import logger

class DocumentChunker(BaseDocumentChunker):
    def __init__(
            self, directory_path: str, **kwargs
    ):

        self.directory_path = directory_path
        self._chunk_size = kwargs.get("chunk_size", None)  # CHARACTER LENGTH
        self._chunk_overlap = kwargs.get("chunk_overlap", None)  # CHARACTER LENGTH
        self._documents: List[Document] = []

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        if value <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self._chunk_size = value

    @property
    def chunk_overlap(self):
        return self._chunk_overlap

    @chunk_overlap.setter
    def chunk_overlap(self, value: int):
        if value < 0:
            raise ValueError("chunk_overlap must be a non-negative integer")
        if self.chunk_size and value >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self._chunk_overlap = value

    def chunk_document(self, document: Document) -> List[Document]:
        text = document.page_content
        metadata = document.metadata.copy()
        chunks = []

        # Ensure we don't have negative step size
        step_size = max(1, self.chunk_size - self.chunk_overlap)

        for i in range(0, len(text), step_size):
            chunk_text = text[i: i + self.chunk_size]
            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = len(chunks)
                chunks.append(Document(page_content=chunk_text, metadata=chunk_metadata))

        return chunks

    def directory_to_documents(self) -> List[Document]:
        """
        Load files from a directory using different loaders based on file extensions.

        Returns:
            List of Document objects from all loaded files.
        """
        # Validate directory path
        if not os.path.exists(self.directory_path):
            logger.info(f"❌ Directory does not exist: {self.directory_path}")
            return []

        # Configuration for different file types
        file_configs = {
            "txt": {"loader_cls": TextLoader, "loader_kwargs": {"encoding": "utf-8"}},
            "pdf": {
                "loader_cls": PyMuPDFLoader,
                "loader_kwargs": {},
            },
        }

        self._documents = []

        for extension, config in file_configs.items():
            pattern = os.path.join(self.directory_path, f"**/*.{extension}")
            files = glob.glob(pattern, recursive=True)

            for file_path in files:
                try:
                    loader = config["loader_cls"](file_path, **config["loader_kwargs"])
                    documents = loader.load()
                    self._documents.extend(documents)
                    logger.info(f"✅ Loaded: {os.path.basename(file_path)}")

                except Exception as e:
                    logger.info(f"❌ Failed to load {os.path.basename(file_path)}: {e}")
                    continue

        return self._documents

    def recommend_embedding_dimension(self) -> int:
        """
        Recommend embedding dimension based on chunk size.

        Returns:
            int: Recommended embedding dimension.
        """
        if self.chunk_size <= 512:
            return 384  # Suitable for small chunks
        elif self.chunk_size <= 1024:
            return 768  # Suitable for medium chunks
        else:
            return 1024  # Suitable for large chunks

    def _get_max_document_words(self) -> int:
        """
        Get the maximum word count from all documents.
        
        Returns:
            int: Maximum word count.
        """
        return max([len(doc.page_content.split()) for doc in self._documents]) if self._documents else 0

    def chunk_documents(self, documents: List[Document] = None) -> List[Document]:
        if documents is None:
            documents = self._documents
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def calculate_optimal_chunk_size(self, max_tokens: int, words_per_token: float = 0.75) -> Tuple[int, int]:
        """
        Calculate optimal chunk_size and chunk_overlap in characters based on max_tokens.
        
        Args:
            max_tokens: Maximum tokens per chunk (from embedding model limit)
            words_per_token: Average words per token (varies by language/model, default 0.75)
        
        Returns:
            Tuple[int, int]: (chunk_size_chars, chunk_overlap_chars)
        """
        # Use 85% of max_tokens to leave safety margin for tokenization variance
        safe_max_tokens = int(max_tokens * 0.85)

        # Convert tokens → words → characters
        avg_chars_per_word = 5.5  # Practical: Works well for most English text Including space after word
        self.estimated_chars_per_token = avg_chars_per_word * words_per_token  # chars/word x  words/token

        # Calculate chunk size in characters
        chunk_size_chars = int(safe_max_tokens * self.estimated_chars_per_token)

        # Overlap should be 15-20% of chunk size for context continuity
        chunk_overlap_chars = int(chunk_size_chars * 0.15)

        logger.info(f"📐 Calculated chunk parameters:")
        logger.info(f"   Max tokens (limit): {max_tokens}")
        logger.info(f"   Safe tokens (85%): {safe_max_tokens}")
        logger.info(f"   Chars per token: {self.estimated_chars_per_token :.2f}")
        logger.info(f"   Chunk size: {chunk_size_chars} characters")
        logger.info(f"   Chunk overlap: {chunk_overlap_chars} characters")

        return chunk_size_chars, chunk_overlap_chars

    def calculate_optimal_chunk_parameters_given_max_tokens(self, max_tokens: int, avg_words_per_token: float) -> Tuple[
        int, int]:
        """ Calculate optimal chunk size and overlap based on max tokens.                   
        Args:
            max_tokens: Maximum tokens allowed by the embedding model.
        Returns:
            chunk_size: Optimal chunk size in characters.
            chunk_overlap: Optimal chunk overlap in characters.
        """
        avg_words_per_sub_token = avg_words_per_token  # self.calculate_avg_words_per_token()
        worst_token_in_all_documents = self.calculate_max_char_sub_tokens_per_word(avg_words_per_sub_token)
        target_chunk_size_tokens = min(max_tokens, worst_token_in_all_documents)
        calculated_chunk_size_characters, calculated_chunk_overlap_characters = self.calculate_optimal_chunk_size(
            target_chunk_size_tokens,
            words_per_token=avg_words_per_sub_token)
        self.chunk_size = calculated_chunk_size_characters or self.chunk_size
        self.chunk_overlap = calculated_chunk_overlap_characters or self.chunk_overlap
        return self.chunk_size, self.chunk_overlap

    def validate_and_fix_chunks(self, chunk_texts: List[str], max_tokens: int) -> List[str]:
        """
        Additional validation and fixing of chunks that might still be too long.
        """
        fixed_chunks = []

        for i, chunk in enumerate(chunk_texts):
            characters = len(chunk)
            # converts characters to tokens and ensure that the tokens don't exceed max_tokens
            estimated_tokens = int(characters / self.estimated_chars_per_token) if self.estimated_chars_per_token else 0
            if estimated_tokens > max_tokens:
                logger.info(f"   ⚠️  Chunk {i + 1} still too long ({estimated_tokens} estimated tokens), re-truncating...")

                # Ultra-conservative truncation limits
                new_truncated_length = int(
                    max_tokens * self.estimated_chars_per_token) if self.estimated_chars_per_token else characters
                word_count = len(chunk.split())
                chunk = chunk[:new_truncated_length]
                new_word_count = len(chunk.split())
                logger.info(f"      ✂️  truncated {word_count} to {new_word_count} words")
            fixed_chunks.append(chunk)
        return fixed_chunks


if __name__ == "__main__":
    # Use a local Linux-compatible path instead of Windows mount
    directory_path = "/home/jmitchall/vllm-srv/test_docs"

    # Create test directory and sample files if they don't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

        # Create a sample text file
        sample_txt = os.path.join(directory_path, "sample.txt")
        with open(sample_txt, "w") as f:
            f.write("This is a sample document for testing the document chunker functionality. " * 50)

        # Create a sample PDF content as text (since we don't have actual PDFs)
        sample_pdf_as_txt = os.path.join(directory_path, "sample_content.txt")
        with open(sample_pdf_as_txt, "w") as f:
            f.write("This simulates PDF content for testing purposes. " * 100)

        logger.info(f"📁 Created test directory and sample files: {directory_path}")

    chunker = DocumentChunker(directory_path, chunk_size=1000, chunk_overlap=200)
    documents = chunker.directory_to_documents()

    logger.info(f"\n📊 Summary: Loaded {len(documents)} documents total")

    # Group by file type
    by_extension = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        ext = os.path.splitext(source)[1][1:].lower() or "unknown"
        by_extension[ext] = by_extension.get(ext, 0) + 1

    for ext, count in by_extension.items():
        logger.info(f"   {ext.upper()}: {count} documents")

    # Chunk documents
    all_chunks = chunker.chunk_documents(documents)

    logger.info(f"\n📊 Summary: Created {len(all_chunks)} chunks total")

    # Group chunks by file type
    chunk_by_extension = {}
    for chunk in all_chunks:
        source = chunk.metadata.get("source", "")
        ext = os.path.splitext(source)[1][1:].lower() or "unknown"
        chunk_by_extension[ext] = chunk_by_extension.get(ext, 0) + 1

    for ext, count in chunk_by_extension.items():
        logger.info(f"   {ext.upper()}: {count} chunks")

    # Example: logger.info first 2 chunks
    for i, chunk in enumerate(all_chunks[:2]):
        logger.info(f"\n--- Chunk {i + 1} ---")
        logger.info(f"Source: {chunk.metadata.get('source', 'unknown')}")
        logger.info(f"Content: {chunk.page_content[:200]}...")  # Print first 200 characters
