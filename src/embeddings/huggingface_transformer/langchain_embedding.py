import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
from langchain_core.embeddings import Embeddings

# This class creates an "embeddings" system using HuggingFace models
# WHAT ARE EMBEDDINGS? Think of them as translating words into a secret code of numbers
# Words with similar meanings get similar numbers, so computers can understand relationships
# For example: "happy" and "joyful" would get very similar number patterns
# But "happy" and "sad" would get very different number patterns
class HuggingFaceOfflineEmbeddings(Embeddings):
    def __init__(self, model_name: str ,
                 torch_dtype: str = "float16",):
        """
        Set up the embedding system using a HuggingFace model.
        
        Simple Explanation:
        - HuggingFace is like a library where thousands of AI models are stored
        - We download and use one of these models to convert text to numbers
        - This all happens "offline" meaning it runs on your computer (not the internet)
        
        Args:
            model_name: The name of the AI model from HuggingFace
                       Example: "sentence-transformers/all-MiniLM-L6-v2"
                       (like choosing which dictionary to use)
                       
            torch_dtype: How precisely to store numbers (default: "float16")
                        - "float32" = very precise but uses lots of memory (like measuring with a ruler to 1/32 inch)
                        - "float16" = less precise but uses half the memory (measuring to 1/16 inch - good enough!)
                        - "bfloat16" = special format that's good for AI, uses half memory
                        Think of it like photo quality: higher quality = bigger file size
        """
        
        # STEP 1: Set up the number precision system
        # Create a dictionary (like a phonebook) that converts text names to actual torch settings
        self.dtype_map = {
            "float16": torch.float16,  # Half precision - uses less memory, slightly less accurate
            "bfloat16": torch.bfloat16,  # Brain floating point - designed specifically for AI
            "float32": torch.float32  # Full precision - uses more memory, more accurate
        }
        # Get the actual torch setting from our map, default to float16 if not found
        self.torch_dtype = self.dtype_map.get(torch_dtype, torch.float16)

        # STEP 2: Load the tokenizer (the text chopper-upper)
        # WHAT IS A TOKENIZER? It breaks sentences into pieces the AI can understand
        # Example: "I love cats" might become ["I", "love", "cats"] or ["I", "lov", "e", "cat", "s"]
        # Different models break text differently, so we need the right tokenizer for our model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # STEP 3: Load the actual AI model (the brain that does the work)
        # This downloads the model from HuggingFace if you don't have it yet
        # Once downloaded, it saves it on your computer for next time (like installing an app)
        self.model = AutoModel.from_pretrained(
            model_name,  # Fixed: was self.model_name (typo), should be model_name
            torch_dtype=self.torch_dtype  # Use the precision setting we chose above
            # , cache_dir="./model_cache"  # Optional: where to save downloaded models on your computer
        )
        
        # STEP 4: Put model in evaluation mode
        # WHAT DOES THIS MEAN? Models have two modes:
        # - Training mode: the model is learning and updating itself (like a student studying)
        # - Evaluation mode: the model is just making predictions (like taking a test)
        # We use eval() because we're not teaching it, just using it
        self.model.eval()
        
        # STEP 5: Check if we have a GPU and set up device
        # GPU (Graphics Processing Unit) = super fast calculator built into graphics cards
        # Originally made for video games, but perfect for AI because both need lots of math
        if torch.cuda.is_available():
            # Clear out any old data from GPU memory (like emptying your backpack before school)
            torch.cuda.empty_cache()
            
            # Move the entire AI model to GPU memory
            # This is like moving your textbooks from your backpack to your desk - makes work faster!
            # GPU can do thousands of calculations at the same time (parallel processing)
            # CPU does calculations one at a time (serial processing)
            self.model.to('cuda')
            print(f"⚡ Model loaded on GPU with dtype={self.torch_dtype}")
        else:
            # No GPU found, so we'll use the regular CPU (slower but works everywhere)
            print(f"💻 Model loaded on CPU")
        print(f"✅ Local model is ready to get embeddings")

    @property
    def get_tokenizer(self):
        """
        Return the tokenizer used by the embedding model.
        """
        return self.tokenizer

    @property
    def max_tokens(self) -> int:
        """
        Return the maximum number of tokens the model can handle.
        
        Simple Explanation:
        - Different AI models can only process a certain amount of text at once
        - This limit is measured in "tokens" (pieces of words)
        - This function tells us what that limit is for our chosen model
        
        Example:
            If the model can handle 512 tokens, and we give it 600 tokens,
            it will have to cut off or ignore the extra 88 tokens.
            
        Why is this important?
        - When preparing text for the model, we need to make sure we don't exceed this limit
        - If we do, the model might give errors or not work properly
        
        How we find the max tokens:
        1. First, try to get it from the tokenizer (most reliable source)
        2. If that doesn't work, try the model's configuration
        3. If both fail, use a safe default of 512 tokens
        """
        
        # STEP 1: Try to get max tokens from the tokenizer
        # The tokenizer knows how much text it can handle
        # model_max_length is a built-in property that most tokenizers have
        if self.tokenizer and hasattr(self.tokenizer, 'model_max_length'):
            max_length = self.tokenizer.model_max_length
            
            # Sometimes tokenizers return a very large number (like 1000000000) 
            # which means "unlimited" but isn't practical
            # If we see this, we'll check the model config instead
            if max_length and max_length < 1000000:  # Reasonable max length
                return max_length
        
        # STEP 2: If tokenizer didn't give us a good answer, check the model's config
        # The model's configuration also stores max position embeddings
        # (max_position_embeddings = maximum number of tokens the model can handle at once)
        if self.model and hasattr(self.model, 'config'):
            # Try to get max_position_embeddings from the model config
            if hasattr(self.model.config, 'max_position_embeddings'):
                return self.model.config.max_position_embeddings
            
            # Some models use 'n_positions' instead of 'max_position_embeddings'
            # (different models use different names for the same thing)
            elif hasattr(self.model.config, 'n_positions'):
                return self.model.config.n_positions
        
        # STEP 3: If both methods above failed, use a safe default
        # 512 tokens is a common limit for many embedding models
        # Better to be cautious than to cause errors
        print("Warning: Could not determine max tokens from tokenizer or model config. Using default of 512.")
        return 512

    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple text documents into numerical embeddings.
        
        Simple Explanation:
        - Takes a list of sentences/documents (like ["hello", "goodbye", "thank you"])
        - Converts each one into a list of numbers (embeddings)
        - Returns all the number lists together
        
        Why process one at a time?
        - Each document might be different lengths
        - This approach is simpler and more reliable
        - For production systems, you'd batch them for speed
        
        Example:
            Input: ["I like cats", "Dogs are fun", "Pizza is great"]
            Output: [
                [0.2, 0.5, -0.1, ...],  # Numbers for "I like cats"
                [0.3, 0.4, -0.2, ...],  # Numbers for "Dogs are fun"
                [0.1, 0.6, 0.3, ...]    # Numbers for "Pizza is great"
            ]
            Each inner list might have 384, 768, or even more numbers!
        """
        # Create an empty list to store all our embeddings
        embeddings = []
        
        # Process each text one at a time (loop through the list)
        for text in texts:
            # STEP 1: Tokenize the text
            # This breaks the text into pieces and converts to numbers the model understands
            # Think of it like translating English to a code the AI can read
            # 
            # Parameters explained:
            # - return_tensors='pt': Return PyTorch tensors (PyTorch's special number containers)
            # - truncation=True: If text is too long, cut it off (models have max lengths)
            # - padding=True: If text is too short, add blank spaces to make it standard length
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            
            # STEP 2: Move the tokenized text to GPU if available
            # If we have a GPU, we need to move our data there too (model and data must be in same place)
            # It's like: if your calculator is on your desk, you need to bring your homework to the desk too!
            if torch.cuda.is_available():
                # This moves each part of the input to GPU memory
                # The curly braces {} create a dictionary with the same keys but GPU values
                inputs = {key: val.to('cuda') for key, val in inputs.items()}

            # STEP 3: Run the model to get embeddings
            # with torch.no_grad() means: "don't track history, we're not training"
            # This saves memory and makes things faster (we don't need to remember steps for learning)
            with torch.no_grad():
                # The ** syntax unpacks the dictionary - it's like passing multiple arguments at once
                # The model processes our text and returns complex output data
                outputs = self.model(**inputs)
            
            # STEP 4: Extract and simplify the embedding
            # The model returns lots of data; we just want the embedding part
            # 
            # WHAT IS MEAN POOLING? Imagine you have a paragraph with 10 words
            # The model gives you 10 sets of numbers (one per word)
            # Mean pooling averages all 10 sets into ONE set of numbers for the whole paragraph
            # It's like getting one overall score instead of 10 individual scores
            #
            # Technical breakdown:
            # - outputs.last_hidden_state: the main embedding data from the model
            # - .mean(dim=1): average across dimension 1 (combine all word embeddings)
            # - .squeeze(): remove extra dimensions (clean up the shape)
            # - .cpu(): move from GPU memory to CPU memory (required before converting to numpy)
            # - .numpy(): convert from PyTorch tensor to regular Python numbers
            #
            # Note: If the tensor is on GPU, we MUST move it to CPU first before converting to numpy
            # NumPy arrays can only exist in CPU memory, not GPU memory
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # STEP 5: Convert to a regular Python list and add to our collection
            # .tolist() converts numpy array to regular Python list (easier to work with)
            embeddings.append(embedding.tolist())
        
        # Return all the embeddings we created
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Convert a single piece of text into a numerical embedding.
        
        Simple Explanation:
        - Takes just ONE sentence or document
        - Converts it to a list of numbers
        - This is a convenience shortcut when you only have one thing to convert
        
        Example:
            Input: "I love pizza"
            Output: [0.2, 0.5, -0.1, 0.8, 0.3, ...]
            
        How it works:
        - Wraps your single text in a list: ["I love pizza"]
        - Calls embed_documents (which expects a list)
        - Takes the first result with [0] (since there's only one)
        
        Why have this method?
        - Often you want to search with just one query
        - This saves you from having to write [text] and [0] every time
        - Makes the code cleaner and easier to read
        """
        # Put text in a list, convert it, then extract the first (only) result
        return self.embed_documents([text])[0]
    

