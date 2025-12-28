import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Tuple
from llama_index.core.schema import Document
from langchain.schema import Document as LangchainDocument
from llama_index.core.schema import BaseNode
from ingestion.base_document_chunker import BaseDocumentChunker

class LlamaIndexDocumentChunker(BaseDocumentChunker):
    def __init__(
            self, directory_path: str, 
            chunk_size: int = 1000, 
            chunk_overlap: int = 200
    ):
        self.directory_path = directory_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._documents: List[Document] = []
        self._langchain_documents:List[LangchainDocument] =[]
    
    @staticmethod
    def to_langchain_format(docs: List[BaseNode]) -> List[LangchainDocument]:
        """Convert llama_index BaseNode objects to LangChain Documents"""
        if not docs:
            return []
        
        return_value = []
        for doc in docs:
            metadata = (doc.metadata or {}).copy()
            # Set 'source' from 'file_path' if it exists
            if 'file_path' in metadata:
                metadata['source'] = metadata['file_path']
            
            return_value.append(LangchainDocument(
                page_content=doc.text,
                metadata=metadata
            ))
        
        return return_value

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

        self._documents = SimpleDirectoryReader(
            input_dir=self.directory_path, 
            required_exts=[".txt", ".pdf"],
            recursive=True
        ).load_data()
        
        # Remove duplicates based on file path
        seen_files = set()
        unique_documents = []
        for doc in self._documents:
            file_path = doc.metadata.get("file_path") or doc.metadata.get("source", "")
            if file_path not in seen_files:
                seen_files.add(file_path)
                unique_documents.append(doc)
        
        self._documents = unique_documents
        print(f"📄 Loaded {len(self._documents)} unique documents from {self.directory_path}")
        
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

    def calculate_avg_words_per_token(self) -> float:
        """
        Estimate average words per token based on chunk size.

        Returns:
            float: Estimated average words per token.
        """
        current_documents = self._documents
        worst_case_sub_token_ratio = [len(doc.text.split()) / len(doc.text) for doc in current_documents
                                      if len(doc.text) > 0]
        avg_words_per_token = sum(worst_case_sub_token_ratio) / len(
            worst_case_sub_token_ratio) if worst_case_sub_token_ratio else 0
        return avg_words_per_token

    def _get_max_document_words(self) -> int:
        """
        Get the maximum word count from all documents.
        
        Returns:
            int: Maximum word count.
        """
        return max([len(doc.text.split()) for doc in self._documents]) if self._documents else 0
    
    def chunk_documents(self, documents: List[Document] = None) -> List[LangchainDocument]:
        node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
        return self.to_langchain_format(nodes)

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

    def calculate_optimal_chunk_parameters_given_max_tokens(self, max_tokens: int) -> Tuple[int, int]:
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

    @staticmethod
    def validate_and_fix_chunks( chunk_texts: List[str], max_tokens: int) -> List[str]:
        """
        Additional validation and fixing of chunks that might still be too long.
        """
        fixed_chunks = []

        for i, chunk in enumerate(chunk_texts):
            # Estimate token count more accurately with pessimistic ratio
            words = len(chunk.split())
            estimated_tokens = int(words * 3.0)  # Very pessimistic - increased from 2.0

            if estimated_tokens > max_tokens:
                print(f"   ⚠️  Chunk {i + 1} still too long ({estimated_tokens} estimated tokens), re-truncating...")

                # Ultra-conservative truncation limits
                if max_tokens < 100:
                    max_words = 10  # Very strict
                elif max_tokens < 150:
                    max_words = 15  # Strict
                elif max_tokens < 200:
                    max_words = 20  # Conservative
                else:
                    max_words = int(max_tokens * 0.1)  # 10% of max_tokens - reduced from 20%

                words_list = chunk.split()
                if len(words_list) > max_words:
                    chunk = ' '.join(words_list[:max_words])
                    print(f"      ✂️  Re-truncated to {max_words} words")

            fixed_chunks.append(chunk)

        return fixed_chunks



if __name__ == "__main__":
    # Use a local Linux-compatible path instead of Windows mount
    directory_path = "/home/jmitchall/vllm-srv/data/"

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

    chunker = LlamaIndexDocumentChunker(directory_path, chunk_size=1000, chunk_overlap=200)
    documents = chunker.directory_to_documents()

    print(f"\n📊 Summary: Loaded {len(documents)} documents total")

    # Group by file type
    by_extension = {}
    lang_docs = chunker.to_langchain_format(documents)
    for doc in lang_docs:
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
