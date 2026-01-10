import os
from ingestion.base_document_chunker import BaseDocumentChunker
from langchain.schema import Document as LangchainDocument
from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.schema import Document
from llama_index.readers.file import PyMuPDFReader
from typing import List, Optional, Tuple, Callable
from refection_logger import logger

class LlamaIndexDocumentChunker(BaseDocumentChunker):
    def __init__(
            self, directory_path: str, **kwargs
    ):
        self.directory_path = directory_path
        self.chunk_size = kwargs.get("chunk_size", 1000)  # TOKEN LENGTH
        self.chunk_overlap = kwargs.get("chunk_overlap", 200)  # TOKEN lENGTH
        self._documents: List[Document] = []
        self._langchain_documents: List[LangchainDocument] = []
        self.kwargs = kwargs
        self.file_extractor = {
            ".pdf": PyMuPDFReader()  # Much better at complex PDFs
        }

    @staticmethod
    def to_langchain_format(docs: List[BaseNode]) -> List[LangchainDocument]:
        """Convert llama_index BaseNode objects to LangChain Documents"""
        if not docs:
            return []

        return_value = []
        for doc in docs:
            metadata = (doc.metadata or {}).copy()

            if 'total_pages' in metadata and 'source' in metadata:
                metadata['page'] = int(metadata['source'])
                metadata['total_pages'] = int(metadata['total_pages'])

            if 'file_type' in metadata:
                metadata['format'] = metadata['file_type']

            if 'file_path' in metadata:
                metadata['source'] = metadata['file_path']

            if 'creation_date' in metadata:
                metadata['creationdate'] = metadata['creation_date']

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
            logger.info(f"❌ Directory does not exist: {self.directory_path}")
            return []

        self._documents = SimpleDirectoryReader(
            input_dir=self.directory_path,
            required_exts=[".txt", ".pdf"],
            recursive=True,
            file_extractor=self.file_extractor
        ).load_data()
        logger.info(f"📄 Loaded {len(self._documents)} unique documents from {self.directory_path}")

        return self._documents

    def output_all_extracted_document_texts(self, output_directory: str = "/home/jmitchall/vllm-srv/data/tmp/"):
        # Group documents by file to iterate through pages
        from collections import defaultdict
        docs_by_file = defaultdict(list)
        for doc in self._documents:
            file_path = doc.metadata.get('file_path', 'unknown')
            docs_by_file[file_path].append(doc)

        # Iterate through each file and its pages
        for file_path, file_docs in docs_by_file.items():
            logger.info("=" * 20 + f" File: {file_path}" + "=" * 40)
            logger.info(f"   📄 Total pages: {len(file_docs)}")

            # Iterate through each page
            all_text = ""
            for i, doc in enumerate(file_docs, 1):
                page_num = doc.metadata.get('page_label', i)
                logger.info(f"\n   📄 Page {page_num}:")
                logger.info(f"      📚 Text length: {len(doc.text)} characters")
                logger.info(f"      📊 Word count: {len(doc.text.split())} words")
                logger.info(f"      📝 First 100 chars: {doc.text[:100]}")

                # Accumulate all text from all pages
                all_text += doc.text + "\n\n"

            # Save full document text (all pages combined) to file
            output_file = f"{output_directory}/extracted_{os.path.basename(file_path)}.txt"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(all_text)
            logger.info(f"\n   💾 Full text ({len(file_docs)} pages) saved to: {output_file}")
            logger.info(f"   📏 Total text length: {len(all_text)} characters")

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
        return max([len(doc.text.split()) for doc in self._documents]) if self._documents else 0

    def get_preferred_node_parser(self):
        """
        Get the preferred node parser based on kwargs.

        Returns: 
            SentenceSplitter or SemanticSplitterNodeParser: Preferred node parser.
        """
        node_parser_type = self.kwargs.get("node_parser_type", "sentence")
        match node_parser_type:
            case "sentence":
                return SentenceSplitter.from_defaults(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            case "hiearchial":
                top_layer = self.chunk_size
                next_layer = int(top_layer / 2)
                last_layer = int(next_layer / 2)
                chunk_size_list = [top_layer, next_layer, last_layer]
                return HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_size_list,
                                                            chunk_overlap=self.chunk_overlap)
            case "semantic":
                embed_model: Optional[BaseEmbedding] = self.kwargs.get("embed_model",
                                                                       None)  #: (BaseEmbedding): embedding model to use for semantic splitting

                if not embed_model:
                    return SentenceSplitter.from_defaults(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

                breakpoint_percentile_threshold: Optional[int] = self.kwargs.get("breakpoint_percentile_threshold",
                                                                                 95)  #: (int): dissimilarity threshold for creating semantic breakpoints, lower value will generate more nodes
                buffer_size: Optional[int] = self.kwargs.get("buffer_size",
                                                             1)  #: (int): number of sentences to group together when evaluating semantic similarity
                sentence_splitter: Optional[Callable[[str], List[str]]] = self.kwargs.get("sentence_splitter",
                                                                                          None)  #: (Callable): optional custom sentence splitter function
                include_metadata: bool = self.kwargs.get("include_metadata",
                                                         True)  #: (bool): whether to include metadata in nodes
                include_prev_next_rel: bool = self.kwargs.get("include_prev_next_rel",
                                                              True)  #: (bool): whether to include prev/next relationships
                id_func: Optional[Callable[[int, Document], str]] = self.kwargs.get("id_func",
                                                                                    None)  #: (Callable): optional function to generate unique IDs for nodes
                if sentence_splitter is None:
                    return SemanticSplitterNodeParser.from_defaults(embed_model=embed_model,
                                                                    breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                                                                    buffer_size=buffer_size,
                                                                    include_metadata=include_metadata,
                                                                    include_prev_next_rel=include_prev_next_rel,
                                                                    id_func=id_func
                                                                    )
                else:
                    return SemanticSplitterNodeParser.from_defaults(
                        buffer_size=buffer_size,
                        embed_model=embed_model,
                        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                        sentence_splitter=sentence_splitter,
                        include_metadata=include_metadata,
                        include_prev_next_rel=include_prev_next_rel,
                        id_func=id_func
                    )

    def chunk_documents(self, documents: List[Document] = None) -> List[LangchainDocument]:
        node_parser = self.get_preferred_node_parser()
        # in heirarchical this gets ALL nodes (parent + child + leaf) - most comprehensive
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=False)
        return self.to_langchain_format(nodes)

    def calculate_optimal_chunk_size(self, max_tokens: int, words_per_token: float) -> Tuple[int, int]:
        """
        Calculate optimal chunk_size and chunk_overlap in tokens based on max_tokens.
        
        Args:
            max_tokens: Maximum tokens per chunk (from embedding model limit)
            words_per_token: Average words per token (varies by language/model)
        
        Returns:
            Tuple[int, int]: (chunk_size_tokens, chunk_overlap_tokens)
        """
        # Use 80-90% of max_tokens to leave safety margin for sentence boundaries
        chunk_size_tokens = int(max_tokens * 0.85)

        # Overlap should be 10-20% of chunk size for good context continuity
        chunk_overlap_tokens = int(chunk_size_tokens * 0.15)

        # Ensure overlap doesn't exceed chunk size
        chunk_overlap_tokens = min(chunk_overlap_tokens, chunk_size_tokens - 1)

        # Calculate approximate character estimates for reference
        avg_chars_per_word = 5.5  # Practical: Works well for most English text Including space after word
        self.estimated_chars_per_token = avg_chars_per_word * words_per_token  # tokens × words/token × chars/word
        chunk_size_chars_approx = int(chunk_size_tokens * self.estimated_chars_per_token)
        chunk_overlap_chars_approx = int(chunk_overlap_tokens * self.estimated_chars_per_token)

        logger.info(f"📐 Calculated chunk parameters:")
        logger.info(f"   Max tokens (limit): {max_tokens}")
        logger.info(f"   Chunk size: {chunk_size_tokens} tokens (~{chunk_size_chars_approx} chars)")
        logger.info(f"   Chunk overlap: {chunk_overlap_tokens} tokens (~{chunk_overlap_chars_approx} chars)")
        logger.info(
            f"   Safety margin: {max_tokens - chunk_size_tokens} tokens ({((max_tokens - chunk_size_tokens) / max_tokens * 100):.1f}%)")

        return chunk_size_tokens, chunk_overlap_tokens

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
        target_chunk_size = min(max_tokens, worst_token_in_all_documents)
        calculated_chunk_size, calculated_chunk_overlap = self.calculate_optimal_chunk_size(target_chunk_size,
                                                                                            words_per_token=avg_words_per_sub_token)
        self.chunk_size = calculated_chunk_size or self.chunk_size
        self.chunk_overlap = calculated_chunk_overlap or self.chunk_overlap
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

        logger.info(f"📁 Created test directory and sample files: {directory_path}")

    chunker = LlamaIndexDocumentChunker(directory_path, chunk_size=1000, chunk_overlap=200)
    documents = chunker.directory_to_documents()

    logger.info(f"\n📊 Summary: Loaded {len(documents)} documents total")

    # Group by file type
    by_extension = {}
    lang_docs = chunker.to_langchain_format(documents)
    for doc in lang_docs:
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
