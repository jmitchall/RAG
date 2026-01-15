#!/usr/bin/env python3
"""
vLLM LangChain Embeddings - LangChain Compatible vLLM Wrapper

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

import torch
from langchain_core.embeddings import Embeddings
from typing import List
from vllm import LLM
from refection_logger import logger

# This class creates an "embeddings" system using vLLM (a fast AI model library)
# Think of embeddings as converting text into numbers that computers can understand and compare
# For example, "cat" and "kitten" would get similar numbers because they mean similar things
class VLLMOfflineEmbeddings(Embeddings):
    def __init__(
            self,
            model_name: str,
            tensor_parallel_size: int = 1,
            gpu_memory_utilization: float = 0.9,
            device: str = "cuda"
    ):
        """
        Set up the embedding system with GPU (graphics card) support for faster processing.
        
        Simple Explanation:
        - This function prepares the AI model to convert text into numbers
        - It can use your computer's GPU (graphics card) to work much faster than the CPU
        - GPU is like having a specialized calculator for AI tasks
        
        Args:
            model_name: The name or location of the AI model to use
                       (like choosing which translator app to use)
                       
            tensor_parallel_size: How many GPUs to split the work across (default: 1)
                                 Think of this like using multiple calculators at once
                                 If you have 2 GPUs, you can set this to 2 for faster processing
                                 
            gpu_memory_utilization: How much of your GPU's memory to use (0.0 to 1.0, default: 0.9)
                                   0.9 means use 90% of available GPU memory
                                   Using less (like 0.7) leaves room for other programs
                                   
            device: Which processor to use - "cuda" for GPU or "cpu" for regular processor
                   GPU (cuda) is much faster but requires a compatible graphics card
                   CPU works on any computer but is slower
        """

        # STEP 1: Check if a GPU is actually available on this computer
        # CUDA is the technology that lets us use NVIDIA graphics cards for AI
        if device == "cuda" and not torch.cuda.is_available():
            # If user asked for GPU but it's not available, warn them and use CPU instead
            logger.info("Warning: CUDA not available, falling back to CPU")
            device = "cpu"

        # STEP 2: Create the actual AI model (called a "client" here)
        # This is like opening the translator app and loading the dictionary
        self.client = LLM(
            model=model_name,  # Which AI model to load (like which language dictionary to use)
            task="embed",  # Tell it we want to convert text to numbers (embeddings), not generate text

            # Only use multiple GPUs if we're actually using GPU mode
            # If on CPU, we set this to 1 since CPU doesn't support parallel processing the same way
            tensor_parallel_size=tensor_parallel_size if device == "cuda" else 1,

            # Only use GPU memory if we're in GPU mode
            # Set to 0.0 for CPU since CPU uses regular RAM, not GPU memory
            gpu_memory_utilization=gpu_memory_utilization if device == "cuda" else 0.0,
        )

        # STEP 3: Remember which device we're using so we can check it later if needed
        self.device = device

    @property
    def max_tokens(self) -> int:
        """
        Return the maximum number of tokens the model can handle for embeddings.
        This is important because models have limits on how much text they can process at once.
        """
        # For many embedding models, the max token limit is around 512 or 1024
        # Here we return a safe default; in a real implementation, you might query the model's config
        # Example Vllm Models for text Emabeddings:
        model_dimensions = {
            "intfloat/e5-mistral-7b-instruct": 512,
            "BAAI/bge-base-en-v1.5": 512,
            "BAAI/bge-large-en-v1.5": 1024,
            "intfloat/e5-large-v2": 512,
        }

        # Dynamically get from the model config if not in our predefined list
        try:
            return self.client.llm_engine.model_config.max_model_len
        except AttributeError:
            # Fallback to tokenizer if model config not accessible
            try:
                tokenizer = self.client.get_tokenizer()
                return tokenizer.model_max_length
            except:
                # Safe default fallback
                if self.client.model in model_dimensions:
                    return model_dimensions[self.client.model]
        return 512  # Default safe limit    

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple text documents into numerical embeddings.
        
        Simple Explanation:
        - Takes a list of text strings (like ["hello", "goodbye", "thanks"])
        - Converts each one into a list of numbers (embeddings)
        - Returns a list of lists - one number list for each text
        
        Example:
            Input: ["cat", "dog", "car"]
            Output: [[0.2, 0.5, ...], [0.3, 0.4, ...], [0.8, 0.1, ...]]
            (Each inner list might have hundreds or thousands of numbers)
        
        Why this is useful:
        - You can compare these numbers to find similar texts
        - Texts with similar meanings will have similar numbers
        """
        # Send all the texts to the AI model to be converted
        outputs = self.client.embed(texts)

        # Extract just the embedding numbers from the model's output
        # "out.outputs.embedding" means: for each output, get the embedding part
        return [out.outputs.embedding for out in outputs]

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a single piece of text into a numerical embedding.
        
        Simple Explanation:
        - Takes one text string (like "hello world")
        - Converts it into a list of numbers
        - This is a shortcut for when you only have one text to convert
        
        Example:
            Input: "hello world"
            Output: [0.2, 0.5, 0.1, 0.8, ...] (might be hundreds of numbers)
        
        How it works:
        - It wraps your single text in a list: [text]
        - Calls embed_documents to do the conversion
        - Takes the first (and only) result with [0]
        """
        # Put the single text in a list, convert it, then take the first result
        return self.embed_documents([text])[0]

# HERE IS AN EXAMPLE OF HOW TO USE THE ABOVE CLASS IN A LANGCHAIN LCEL PIPELINE
# Uncomment and run in your application environment
# from langchain_core.vectorstores import FAISS
# from langchain_core.chat_models import ChatOpenAI
# from langchain_core.prompts.chat import ChatPromptTemplate
# from langchain_core.chains import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser 
#
# # 1. Initialize your custom vLLM Embedding class
# embeddings = VLLMOfflineEmbeddings("intfloat/e5-mistral-7b-instruct")

# # 2. Create a Vector Store (e.g., using FAISS)
# # In a real app, you'd load actual documents here
# vectorstore = FAISS.from_texts(
#     ["vLLM is a high-throughput library for LLM inference.", 
#      "LCEL is a declarative way to compose LangChain components."],
#     embeddings
# )
# retriever = vectorstore.as_retriever()

# # 3. Define the LCEL Chain
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI(model="gpt-4o") # Example model

# # This is the LCEL pipeline
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | model
#     | StrOutputParser()
# )

# # 4. Invoke the chain
# response = chain.invoke("What is vLLM?")
# logger.info(response)


#  Setup Embeddings & Vector Store
# from langchain_community.vectorstores import Qdrant
# from qdrant_client import QdrantClient
#
# embeddings = VLLMOfflineEmbeddings("intfloat/e5-mistral-7b-instruct")
# # Example using Qdrant Vector Store to store List a list of LangChain Document objects
# from langchain_core.schema import Document
# from typing import List
# # documents: List[Document] 
#
# # Initialize Qdrant client
# qdrant_client = QdrantClient(path="./qdrant_db")  # Local persistent storage
# 
# # Assuming 'documents' is a list of LangChain Document objects
# vectorstore = Qdrant (
#     client=qdrant_client,
#     collection_name="my_collection",
#     embedding_function=embeddings,
# )
# retriever = vectorstore.as_retriever()

#
