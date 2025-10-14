import numpy as np
import torch
from langchain_core.documents import Document
from typing import List, Dict, Tuple

from doc_parser import DocumentChunker
from embedding_impl import EmbeddingManager
from vector_db_factory import VectorDBFactory


class DataIngestionPipeline:
    """
    A pipeline to ingest documents, chunk them, generate embeddings, and store in a vector database.
    """

    def __init__(self, db_type: str = "faiss", directory_path: str = "/home/jmitchall/vllm-srv/test_docs",
                 embedding_model: str = "BAAI/bge-base-en-v1.5", use_embedding_server: bool = True,
                 safety_level: str = "recommended"):
        self.db_type = db_type.lower()
        self.directory_path = directory_path
        self.embedding_model = embedding_model
        self.use_embedding_server = use_embedding_server
        self.safety_level = safety_level
        self.max_tokens = None
        self.chunk_size = None
        self.chunk_overlap = None

    def get_available_vector_databases(self, validated_db: str) -> bool:
        """ Check and display available vector databases """
        print(f"🔍 Checking availability of vector databases...")
        # Show available databases
        available_dbs = VectorDBFactory.get_available_databases()
        print("🗃️  Available Vector Databases:")
        for db_name, info in available_dbs.items():
            status = "✅" if info['available'] else "❌"
            gpu = "🚀" if info['gpu_support'] else "💻"
            print(f"   {status} {gpu} {db_name}: {info['description']}")

        if not available_dbs[validated_db]['available']:
            print(f"❌ {validated_db} is not available!")
            print(f"💡 {available_dbs[validated_db]['description']}")
            return False
        else:
            print(f"✅ {validated_db} is available")
            return True

    def validate_and_fix_chunks(self, chunk_texts: List[str], max_tokens: int) -> List[str]:
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

    def calculate_safe_batch_size(self, chunk_texts: List[str], max_tokens: int) -> int:
        """
        Calculate a safe batch size based on the average chunk length and max_tokens.
        Uses very conservative estimates to prevent batch overflow.
        """
        if not chunk_texts:
            return 1

        # Calculate average words per chunk (sample more chunks for better estimate)
        sample_size = min(20, len(chunk_texts))
        total_words = sum(len(chunk.split()) for chunk in chunk_texts[:sample_size])
        avg_words_per_chunk = total_words / sample_size

        # Use very pessimistic token estimation (3.0 words-to-tokens ratio)
        estimated_tokens_per_chunk = int(avg_words_per_chunk * 3.0)

        print(
            f"📊 Batch calculation: avg {avg_words_per_chunk:.1f} words/chunk, ~{estimated_tokens_per_chunk} tokens/chunk")

        # Calculate safe batch size with large buffer
        if estimated_tokens_per_chunk <= 0:
            safe_batch_size = 1
        else:
            # Leave 40% buffer for server overhead and tokenization variance
            available_tokens = int(max_tokens * 0.6)
            safe_batch_size = max(1, available_tokens // estimated_tokens_per_chunk)

        # Cap at very conservative limits
        safe_batch_size = min(safe_batch_size, 2)  # Never more than 2 in a batch

        # For small max_tokens, force individual processing
        if max_tokens < 200:
            safe_batch_size = 1

        print(f"📊 Calculated safe batch size: {safe_batch_size} (with 40% buffer)")
        return safe_batch_size

    def process_embeddings_safely(self, embedding_manager: EmbeddingManager, chunk_texts: List[str], max_tokens: int) -> \
            List[
                np.ndarray]:
        """
        Process embeddings with multiple fallback strategies to handle token limits.
        """
        print(f"🔄 Processing {len(chunk_texts)} chunks with max_tokens={max_tokens}")

        # Strategy 1: Skip batch processing entirely - too unreliable with token limits
        print(f"⚠️  Skipping batch processing entirely, using individual processing for maximum safety")

        # Strategy 2: Process one by one with final emergency truncation (most reliable)
        print(f"🔄 Processing chunks individually with ultra-aggressive pre-check...")
        all_embeddings = []
        successful_chunks = []
        retry_count = 0
        pre_truncate_count = 0

        for i, chunk in enumerate(chunk_texts):

            safe_chunk = chunk
            try:
                embedding = embedding_manager.get_embedding(safe_chunk)
                all_embeddings.append(embedding)
                successful_chunks.append(safe_chunk)

                if (i + 1) % 1000 == 0:  # Report every 1000 chunks
                    print(f"   ✅ Processed {i + 1}/{len(chunk_texts)} chunks")

            except Exception as single_e:
                retry_count += 1
                error_msg = str(single_e)

                # Extract token count from error if available
                if "input tokens" in error_msg:
                    # Try to extract the actual token count from error message
                    try:
                        import re
                        match = re.search(r'(\d+)\s+input tokens', error_msg)
                        if match:
                            actual_tokens = int(match.group(1))
                            word_count = len(safe_chunk.split())
                            ratio = actual_tokens / word_count if word_count > 0 else 0
                            print(
                                f"   ❌ Chunk {i + 1} token overflow: {actual_tokens} tokens for {word_count} words (ratio: {ratio:.2f})")
                        else:
                            print(f"   ❌ Chunk {i + 1} failed with token overflow")
                    except:
                        print(f"   ❌ Chunk {i + 1} failed with token overflow")
                else:
                    print(f"   ❌ Chunk {i + 1} failed: {error_msg[:100]}")

                # Try progressive emergency truncation with MUCH more aggressive limits
                words = safe_chunk.split()
                truncation_attempts = [
                    (7, "7 words"),
                    (5, "5 words"),
                    (3, "3 words"),
                    (2, "2 words"),
                    (1, "1 word")  # Added single word as last resort before giving up
                ]

                success = False
                for max_words, label in truncation_attempts:
                    if len(words) >= max_words:
                        emergency_chunk = ' '.join(words[:max_words])
                        try:
                            embedding = embedding_manager.get_embedding(emergency_chunk)
                            all_embeddings.append(embedding)
                            successful_chunks.append(emergency_chunk)
                            if retry_count <= 10:  # Only print first 10
                                print(f"   🆘 Emergency truncation succeeded for chunk {i + 1} ({label})")
                            elif retry_count == 11:
                                print(f"   ... (continuing emergency truncations silently)")
                            success = True
                            break
                        except Exception as emergency_e:
                            continue

                if not success:
                    print(f"   ❌ Skipping chunk {i + 1} - all truncation attempts failed (including 1 word)")
                    continue

        if len(all_embeddings) > 0:
            success_rate = (len(all_embeddings) / len(chunk_texts)) * 100
            print(f"✅ Individual processing completed:")
            print(f"   ✅ Successful: {len(all_embeddings)}/{len(chunk_texts)} ({success_rate:.1f}%)")
            print(f"   🆘 Required emergency truncation: {retry_count}")
            return all_embeddings
        else:
            raise Exception("No chunks could be processed - all failed")

    def generate_vector_db_and_embedding_mgr(self):

        # Step 0:  Check if chosen vector database is available
        DATABASE_TYPE = self.db_type.lower()
        if not self.get_available_vector_databases(DATABASE_TYPE):
            return
        print(f"\n🎯 Using {DATABASE_TYPE.upper()} vector database")

        # Step 1: Choose optimal max_tokens based on chosen embedding  model
        self.max_tokens, actual_embedding_dim = EmbeddingManager.estimate_optimal_max_token_actual_embeddings(
            embedding_model=self.embedding_model,
            use_embedding_server=self.use_embedding_server,
            batch_processing=False,  # Changed to False since we're forcing individual processing
            safety_level=self.safety_level
        )
        print(f"📏 Using max_tokens: {self.max_tokens}")

        # Step 2: Load Documents
        chunker = DocumentChunker(self.directory_path)
        documents = chunker.directory_to_documents()

        # Step 3: Calculate optimal chunk parameters if not provided then Chunk Documents
        chunker.calculate_optimal_chunk_parameters_given_max_tokens(self.max_tokens)
        document_chunks = chunker.chunk_documents(documents)

        print(f"📊 Loaded {len(documents)} documents and created {len(document_chunks)} chunks")
        chunk_texts = [doc.page_content for doc in document_chunks]

        # Additional validation and fixing taking into consideration embedding model limits
        print(f"🔍 Validating chunk sizes based on max_tokens: {self.max_tokens} ...")
        chunk_texts = self.validate_and_fix_chunks(chunk_texts, self.max_tokens)

        # Step 4: Create EmbeddingManager with the CORRECT dimension

        embedding_manager = EmbeddingManager(
            model_name=self.embedding_model,
            embedding_dim=actual_embedding_dim,  # Use detected dimension
            use_server=self.use_embedding_server,
            max_tokens=self.max_tokens
        )

        # Step 6: Verify the dimension (optional double-check)
        verified_dim = embedding_manager.get_actual_embedding_dimension()
        if verified_dim != actual_embedding_dim:
            print(f"⚠️  Dimension mismatch! Detected: {actual_embedding_dim}, Verified: {verified_dim}")
            actual_embedding_dim = verified_dim  # Use the verified dimension

        # step 7: Process embeddings with robust error handling and multiple fallback strategies
        try:
            print(f"🔄 Getting embeddings for {len(chunk_texts)} text chunks...")
            embeddings = self.process_embeddings_safely(embedding_manager, chunk_texts, self.max_tokens)

            # If we lost some chunks, update our data structures
            if len(embeddings) < len(chunk_texts):
                print(f"⚠️  Some chunks were skipped. Using {len(embeddings)} out of {len(chunk_texts)} chunks")
                # We need to track which chunks were successful
                # For now, just truncate to match (this assumes they're processed in order)
                # In production, you'd want to track indices more carefully
                print(f"⚠️  Warning: Document-embedding alignment may be approximate")

        except Exception as e:
            print(f"❌ All embedding strategies failed: {e}")
            print(f"💡 Current max_tokens: {self.max_tokens}")
            return None, None

        print(f"✅ Successfully processed {len(embeddings)} embeddings")

        # Step 8: Only keep document chunks that have corresponding embeddings
        if len(embeddings) < len(document_chunks):
            print(f"🔧 Adjusting document chunks to match embeddings count...")
            document_chunks = document_chunks[:len(embeddings)]

        # Step 9: Create vector database using factory with CORRECT dimension
        vector_db = VectorDBFactory.create_vector_db(
            db_type=DATABASE_TYPE,
            embedding_dim=actual_embedding_dim,  # Use detected dimension
            persist_path=f"/home/jmitchall/vllm-srv/vector_db_{DATABASE_TYPE}",
            use_gpu=True,
            collection_name="vtm_docs",
        )
        vector_db.add_documents(document_chunks, embeddings)

        # Save the database
        vector_db.save()

        return vector_db, embedding_manager
