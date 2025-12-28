import glob

import os

from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyMuPDFLoader,
)
from typing import Dict, Type, Any, List, Tuple


class DocumentChunker:
    def __init__(
            self, directory_path: str
    ):
        self.directory_path = directory_path
        self._chunk_size = None
        self._chunk_overlap = None
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
            print(f"❌ Directory does not exist: {self.directory_path}")
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
                    print(f"✅ Loaded: {os.path.basename(file_path)}")

                except Exception as e:
                    print(f"❌ Failed to load {os.path.basename(file_path)}: {e}")
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

    def calculate_avg_words_per_token(self) -> int:
        """
        Estimate average words per token based on chunk size.

        Returns:
            int: Estimated average words per token.
        """
        current_documents = self._documents
        worst_case_sub_token_ratio = [len(doc.page_content.split()) / len(doc.page_content) for doc in current_documents
                                      if len(doc.page_content) > 0]
        avg_words_per_token = sum(worst_case_sub_token_ratio) / len(
            worst_case_sub_token_ratio) if worst_case_sub_token_ratio else 0
        return avg_words_per_token

    def calculate_max_char_sub_tokens_per_word(self) -> int:
        """
        Estimate maximum character sub-tokens per word based on chunk size.

        Returns:
            int: Estimated maximum character sub-tokens per word.
        """
        avg_words_per_token = self.calculate_avg_words_per_token()
        max_doc_words_in_any_page_content = max(
            [len(doc.page_content.split()) for doc in self._documents]) if self._documents else 0
        return int(max_doc_words_in_any_page_content / avg_words_per_token if avg_words_per_token > 0 else 0)

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
        Calculate optimal chunk_size and chunk_overlap based on max_tokens.
        
        Args:
            max_tokens: Maximum tokens per chunk
            words_per_token: Average words per token (varies by language/model)
        
        Returns:
            Tuple[int, int]: (chunk_size_chars, chunk_overlap_chars)
        """
        # Convert tokens to approximate word count
        max_words = int(max_tokens * words_per_token)

        # Average characters per word (including spaces) is ~5-6
        avg_chars_per_word = 5.5
        chunk_size_chars = int(max_words * avg_chars_per_word)

        # Overlap should be 10-20% of chunk size
        chunk_overlap_chars = int(chunk_size_chars * 0.20)

        chunk_size_chars = min(max_tokens, chunk_size_chars)  # Ensure chunk size does not exceed max_tokens

        print(f"📐 Calculated chunk parameters:")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Max words: {max_words}")
        print(f"   Chunk size: {chunk_size_chars} characters")
        print(f"   Chunk overlap: {chunk_overlap_chars} characters")

        return chunk_size_chars, chunk_overlap_chars

    def calculate_optimal_chunk_parameters_given_max_tokens(self, max_tokens: int) -> (int, int):
        """ Calculate optimal chunk size and overlap based on max tokens.                   
        Args:
            max_tokens: Maximum tokens allowed by the embedding model.
        Returns:
            chunk_size: Optimal chunk size in characters.
            chunk_overlap: Optimal chunk overlap in characters.
        """
        avg_words_per_sub_token = self.calculate_avg_words_per_token()
        worst_sub_words_in_all_documents = self.calculate_max_char_sub_tokens_per_word()
        target_chunk_size = min(max_tokens, worst_sub_words_in_all_documents)
        calculated_chunk_size, calculated_chunk_overlap = self.calculate_optimal_chunk_size(target_chunk_size,
                                                                                            words_per_token=avg_words_per_sub_token)
        self.chunk_size = self.chunk_size or calculated_chunk_size
        self.chunk_overlap = self.chunk_overlap or calculated_chunk_overlap
        return self.chunk_size, self.chunk_overlap


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

        print(f"📁 Created test directory and sample files: {directory_path}")

    chunker = DocumentChunker(directory_path, chunk_size=1000, chunk_overlap=200)
    documents = chunker.directory_to_documents()

    print(f"\n📊 Summary: Loaded {len(documents)} documents total")

    # Group by file type
    by_extension = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        ext = os.path.splitext(source)[1][1:].lower() or "unknown"
        by_extension[ext] = by_extension.get(ext, 0) + 1

    for ext, count in by_extension.items():
        print(f"   {ext.upper()}: {count} documents")

    # Chunk documents
    all_chunks = chunker.chunk_documents(documents)

    print(f"\n📊 Summary: Created {len(all_chunks)} chunks total")

    # Group chunks by file type
    chunk_by_extension = {}
    for chunk in all_chunks:
        source = chunk.metadata.get("source", "")
        ext = os.path.splitext(source)[1][1:].lower() or "unknown"
        chunk_by_extension[ext] = chunk_by_extension.get(ext, 0) + 1

    for ext, count in chunk_by_extension.items():
        print(f"   {ext.upper()}: {count} chunks")

    # Example: Print first 2 chunks
    for i, chunk in enumerate(all_chunks[:2]):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Content: {chunk.page_content[:200]}...")  # Print first 200 characters
