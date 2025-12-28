from litellm import completion
from typing import Callable, Dict
import atexit
import signal
import sys
import torch
from retrieval.query_retrieval_pipeline import QueryRetrievalPipeline


class RAGRetrievalPipeline(QueryRetrievalPipeline):
    def __init__(self, llm_model_name: str = "OpenHermes-2.5-Mistral-7B-GPTQ", gpu_memory_utilization: float = .85,  
                 vector_db_path: str = "/home/jmitchall/vllm-srv/vector_db_chroma", 
                 embedding_model: str = "BAAI/bge-large-en-v1.5", use_embedding_server: bool = False, 
                 safety_level: str = "max"):
        super().__init__(vector_db_path, embedding_model, use_embedding_server, safety_level)

        self.llm = None  # Initialize to None
        self.llm_model_name = llm_model_name
        # Register cleanup handlers BEFORE loading model
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.load_vector_db_and_embedding_mgr()
        if self.vector_db and self.embedding_mgr:
            print("✅ Vector DB and Embedding Manager loaded successfully!")
            self.llm = self.get_supported_models(llm_model_name, 
                                                 gpu_memory_utilization=gpu_memory_utilization 
                                                )


    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
                """Cleanup resources before shutdown."""
                print("\n🧹 Cleaning up RAG pipeline...")

                try:
                    # Clear LLM config (no engine to shutdown with LiteLLM)
                    if hasattr(self, 'llm') and self.llm is not None:
                        print("   🔄 Clearing LLM configuration...")
                        del self.llm
                        self.llm = None
                        print("   ✅ LLM config cleared")

                    # Clear vector DB
                    if hasattr(self, 'vector_db') and self.vector_db is not None:
                        del self.vector_db
                        self.vector_db = None
                        print("   ✅ Vector DB unloaded")

                    # Clear embedding manager
                    if hasattr(self, 'embedding_mgr') and self.embedding_mgr is not None:
                        del self.embedding_mgr
                        self.embedding_mgr = None
                        print("   ✅ Embedding manager unloaded")

                    # Force garbage collection
                    import gc
                    gc.collect()
                    print("   ✅ Garbage collected")

                    print("✅ Cleanup complete!")

                except Exception as e:
                    print(f"⚠️  Error during cleanup: {e}")

    def get_supported_models(self, supported_model_name: str, gpu_memory_utilization: float = 0.85,
                                                  max_model_len: int = 16384) -> dict:
         """
         Get a supported LLM model configuration by name.

         Returns:
             dict: Model configuration for LiteLLM
         """
         supported_models: Dict[str, Callable] = {
             "OpenHermes-2.5-Mistral-7B-GPTQ": self.load_openhermes_mistral_7b,
             "Mistral-7B-Instruct-v0.2-GPTQ": self.load_mistral_quantized,
             "Zephyr-7B-Beta-GPTQ": self.load_zephyr_7b_beta
         }

         loader_func = supported_models.get(supported_model_name)

         if loader_func:
             model_config = loader_func(gpu_memory_utilization=gpu_memory_utilization,
                                        max_model_len=max_model_len)

             if not isinstance(model_config, dict):
                 raise TypeError(f"Loader function did not return dict config, got {type(model_config)}")

             return model_config

         available_models = ", ".join(supported_models.keys())
         raise ValueError(f"Unsupported model: '{supported_model_name}'. Available models: {available_models}")

    def load_zephyr_7b_beta(self, gpu_memory_utilization: float = 0.85, max_model_len: int = 16384) -> dict:
        """
        Configure Zephyr 7B Beta for LiteLLM (local model).

        Args:
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model sequence length

        Returns:
            dict: Model configuration for LiteLLM
        """
        print("\n💨 Loading Zephyr 7B Beta via LiteLLM...")
        print("   Aligned Mistral 7B variant")

        # LiteLLM configuration for local model
        model_config = {
            "model": "huggingface/TheBloke/zephyr-7B-beta-GPTQ",
            "api_base": None,  # Local inference
            "max_tokens": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            # Model-specific parameters
            "model_kwargs": {
                "trust_remote_code": True,
                "quantization": "gptq",
                "dtype": "half"
            }
        }

        print("✅ Zephyr 7B Beta configured for LiteLLM!")
        return model_config

    def load_mistral_quantized(self, gpu_memory_utilization: float = 0.85,
                                       max_model_len: int = 16384) -> dict:
        """
        Configure Mistral 7B Instruct v0.2 for LiteLLM (local model).

        Args:
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model sequence length

        Returns:
            dict: Model configuration for LiteLLM
        """
        print("🔧 Loading Mistral 7B Instruct v0.2 (GPTQ 4-bit Quantized) via LiteLLM...")
        print("   Model size: ~4GB VRAM (vs 13.5GB unquantized)")
        print("   Quantization: GPTQ 4-bit for RTX 5080 16GB compatibility")

        # LiteLLM configuration for local model
        model_config = {
            "model": "huggingface/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
            "api_base": None,  # Local inference
            "max_tokens": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            # Model-specific parameters
            "model_kwargs": {
                "trust_remote_code": True,
                "quantization": "gptq",
                "dtype": "half",
                "tensor_parallel_size": 1
            }
        }

        print("✅ Mistral 7B GPTQ configured for LiteLLM!")
        print("   Context window: 16,384 tokens")
        return model_config
        

    def load_openhermes_mistral_7b(self, gpu_memory_utilization: float = 0.85,
                                       max_model_len: int = 16384) -> dict:
        """
        Configure OpenHermes 2.5 Mistral 7B for LiteLLM (local model).

        Args:
            gpu_memory_utilization: GPU memory utilization (0.0-1.0)
            max_model_len: Maximum model sequence length

        Returns:
            dict: Model configuration for LiteLLM
        """
        print("\n🧙 Loading OpenHermes 2.5 Mistral 7B (GPTQ) via LiteLLM...")
        print("   Fine-tuned for instruction following")
        print(f"   GPU Memory: {gpu_memory_utilization*100:.0f}%")
        print(f"   Max Context: {max_model_len} tokens")

        # LiteLLM configuration for local model
        model_config = {
            "model": "huggingface/TheBloke/OpenHermes-2.5-Mistral-7B-GPTQ",
            "api_base": None,  # Local inference
            "max_tokens": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            # Model-specific parameters
            "model_kwargs": {
                "trust_remote_code": True,
                "quantization": "gptq",
                "dtype": "half",
                "tensor_parallel_size": 1
            }
        }

        print("✅ OpenHermes 2.5 configured for LiteLLM!")
        return model_config

    def get_query_context(self, query_text: str, top_k: int = 5) -> str:
        """
        Retrieve context documents for a given query.
        Args:
            query_text: The input query string
            top_k: Number of top similar documents to retrieve

        Returns:
            Concatenated context string from similar documents
        """
        results = self.retrieve(query_text, top_k=top_k)
        context = ""
        print(f"\n🏆 Top {top_k} similar documents:")
        for i, doc in enumerate(results):
            context += doc.page_content + "\n\n"
        
        # If context consists of whitespace and carriage returns, print a warning
        if context.strip() == "":
            print("⚠️ No relevant documents found for the query.")
        
        return context

    def get_query_context_advanced(self, query_text: str, top_k: int = 5, minimum_score: float = 0.4) -> str:
        """
        Retrieve context documents for a given query.
        Args:
            query_text: The input query string
            top_k: Number of top similar documents to retrieve

        Returns:
            Concatenated context string from similar documents
        """
        results = self.retrieve(query_text, top_k=top_k)
        print(f"\n🔍 Retrieved {len(results)} documents for query '{query_text}'")
        context = ""
        sources = []
        print(f"\n🏆 Top {top_k} similar documents:")
        for i, doc in enumerate(results):  
            if minimum_score > 0.0:
                score = doc.metadata.get('score', 0.0)
                if score == 0.0:
                    score = doc.metadata.get('similarity_score', 0.0)
                    doc.metadata['score'] = score
                    
                if score < minimum_score:
                    print(f"   ⏭️ Skipping document {i+1} with score {score:.4f} below minimum {minimum_score:.4f}")
                    continue
            context += doc.page_content + "\n\n"
            sources.append({
                "index": i,
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A'),
                "score": doc.metadata.get('score', 'N/A'),
                "preview": doc.page_content[:300]  # First 300 characters as preview
            })
        scores = [ score.get('score') for score in sources  if score.get('score') is not None]
        if scores:
            confidence = max( scores)
        else:
            confidence = 0.0
        print(f"Confidence score: {confidence}")
        # If context consists of whitespace and carriage returns, print a warning
        if context.strip() == "":
            print("⚠️ No relevant documents found for the query.")
      
        return context , sources, confidence

    
    def format_prompt_with_template(self, query_text: str, context: str) -> str:
        """
        Format prompt using the correct chat template for the model.
        
        Args:
            query_text: User's question
            context: Retrieved context from vector DB
            model_name: Name of the LLM model
            
        Returns:
            Properly formatted prompt string
        """
        # System message with clear instructions
        system_message = """You are a helpful AI assistant that answers questions based on the provided context. 

Instructions:
- Provide detailed, comprehensive answers
- Use information from the context to support your answer
- If the context doesn't contain enough information, say so
- Be thorough and explain your reasoning
- Use examples from the context when relevant"""

        # User message with context and question
        user_message = f"""Context information is below:
---------------------
{context}
---------------------

Based on the context above, please answer the following question in detail:

Question: {query_text}

Answer:"""

        # Format based on model type
        if self.llm_model_name == "Mistral-7B-Instruct-v0.2-GPTQ":
            # Mistral format: <s>[INST] {instruction} [/INST]
            prompt = f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
            
        elif self.llm_model_name == "OpenHermes-2.5-Mistral-7B-GPTQ":
            # ChatML format (OpenHermes uses this)
            prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
            
        elif self.llm_model_name == "Zephyr-7B-Beta-GPTQ":
            # Zephyr format
            prompt = f"""<|system|>
{system_message}</s>
<|user|>
{user_message}</s>
<|assistant|>
"""
            
        else:
            # Fallback to Mistral format
            print(f"⚠️  Unknown model format, using Mistral template")
            prompt = f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
        
        return prompt
    

    def generate_prompt(self, query_text: str, top_k: int = 5) -> str:
        """
        Generate a prompt for the LLM using retrieved context documents.
        Args:
            query_text: The input query string
            top_k: Number of top similar documents to retrieve
        Returns:
            Formatted prompt string
        """
        context, sources, confidence = self.get_query_context_advanced(query_text, top_k=top_k)
        prompt = self.format_prompt_with_template(query_text, context)
        return prompt , sources, confidence

    def answer_query(self, query_text: str, top_k: int = 5,
                     max_tokens: int = 512, temperature: float = 0.7) -> str:
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Cannot generate answer.")

        prompt, sources, confidence = self.generate_prompt(query_text, top_k=top_k)
        print("\n🧙 Generating answer from LLM...")

        try:
            response = completion(
                model=self.llm["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                stop=["</s>", "<|im_end|>", "<|user|>"]
            )

            answer = response.choices[0].message.content
            print(f"✅ Answer generated successfully!")

            return answer, sources, confidence

        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            raise

    @staticmethod
    def format_source(source: dict) -> str:
        """
        Format a source dictionary into a readable string.
        
        Args:
            source: Dictionary with 'source', 'page', 'score' keys
            
        Returns:
            Formatted string
        """
        source_name = source.get('source', 'Unknown')
        page = source.get('page', 'N/A')
        score = source.get('score', 0.0)
        
        # Ensure score is a number
        if isinstance(score, (int, float)):
            score_str = f"{score:.4f}"
        else:
            score_str = str(score)
        
        return f"• {source_name} (page {page}, score: {score_str})"
    
    def display_results(self, query: str, answer: str, sources: list, confidence: float):
        """
        Display query results in a formatted way.
        
        Args:
            query: The original query
            answer: Generated answer
            sources: List of source documents
            confidence: Confidence score
        """
        print("\n" + "="*70)
        print("🎯 FINAL ANSWER:")
        print("="*70)
        print(f"Query: {query}")
        print(f"\nAnswer: {answer}")
        
        print(f"\n📚 Sources ({len(sources)} documents):")
        for source in sources:
            print(f"   {self.format_source(source)}")
        
        print(f"\n📊 Confidence: {confidence}")
        print("="*70)

if __name__ == "__main__":  
    top_n_answers = 5
    rag_pipeline = None
    
    try:
        # Example: Search for documents similar to a sample query
        rag_pipeline = RAGRetrievalPipeline(
            llm_model_name="Mistral-7B-Instruct-v0.2-GPTQ",
            gpu_memory_utilization=0.85,
            vector_db_path="/home/jmitchall/vllm-srv/vector_db_all_docs_faiss",
            embedding_model="BAAI/bge-large-en-v1.5",
            use_embedding_server=False,
            safety_level="max"  # Using safe mode
        )
        
        # Define test queries
        test_queries = [
            """Given the Archetypes based on playstyles 
Actor: Enjoys role-playing and embodying their character's personality and motivations.
Explorer: Focused on discovering the game world, its secrets, and hidden locations.
Instigator: Seeks out action, challenges, and opportunities to be confrontational.
Power Gamer: Aims to optimize character abilities to be as effective and powerful as possible.
Slayer: Enjoys the combat aspect of the game, relishing the challenge of defeating enemies.
Storyteller:  Prioritizes collaborative storytelling and creating a compelling narrative with the group.
Thinker: Prefers strategic thinking, puzzle-solving, and planning the party's actions.
Watcher: Acts as a supportive and observant member of the party, often enjoying the group dynamic without being the focus of attention. 
Come up with a D&D campaign  idea that would appeal to a mix of players whose styles are Actor, Storyteller, Watcher, Power Gamer, Explorer, and Watcher
"""
        ]
        
        # Process each query
        for query in test_queries:
            print(f"\n🔍 Processing query: '{query}'")
            max_allowed_tokens_for_response = 512
            # .7 represents a good balance between creativity and coherence 
            # .1 is very focused and deterministic 
            # 1.0 is very creative but may lose coherence
            temperature_for_response = 0.75 
            answer, sources, confidence = rag_pipeline.answer_query(query, top_k=top_n_answers,
             max_tokens=max_allowed_tokens_for_response, temperature=temperature_for_response)
            rag_pipeline.display_results(query, answer, sources, confidence)


    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if rag_pipeline is not None:
            print("\n🧹 Performing final cleanup...")
            rag_pipeline.cleanup()
        print("\n👋 Program exiting cleanly")