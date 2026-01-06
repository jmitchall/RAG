# Elminster Project Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Root Level Files](#root-level-files)
4. [Source Code Directory (src/)](#source-code-directory)
   - [Main Agent Files](#main-agent-files)
   - [RAG Pipeline Files](#rag-pipeline-files)
   - [Embeddings Module](#embeddings-module)
   - [Ingestion Module](#ingestion-module)
   - [Inference Module](#inference-module)
   - [Vector Databases Module](#vector-databases-module)

---

## Project Overview

**Elminster** is an AI agent framework designed to provide intelligent question-answering capabilities using **Large Language Models (LLMs)** combined with **Retrieval-Augmented Generation (RAG)**. The system processes documents from multiple sources (Dungeons & Dragons rulebooks, Vampire: The Masquerade books, etc.), converts them into searchable vector embeddings, and uses a reflection agent pattern to generate high-quality answers with iterative refinement.

### Key Technologies
- **vLLM**: Fast LLM inference engine
- **LangChain**: Framework for building LLM applications
- **LlamaIndex**: Document indexing and retrieval
- **Vector Databases**: Qdrant, FAISS, Chroma (for semantic search)
- **HuggingFace Transformers**: Local embedding models
- **LangGraph**: Agent orchestration and workflow management

---

## Project Structure

```
vllm-srv/
├── Root Configuration Files
│   ├── main.py                    # Entry point
│   ├── check_gpu.py              # GPU health checker
│   ├── pyproject.toml            # Project dependencies
│   ├── requirements.txt           # Python dependencies
│   ├── requirements-cuda.txt      # CUDA-specific dependencies
│   └── README.md                 # Project documentation (this file)
│
├── src/                          # Main source code
│   ├── Reflection Agent Files    # Core AI agent implementations
│   ├── RAG Generator Files       # Question answering pipeline
│   ├── embeddings/               # Text embedding utilities
│   ├── ingestion/                # Document loading and chunking
│   ├── inference/                # LLM inference and models
│   └── vectordatabases/          # Vector database implementations
│
├── data/                         # Document collections
│   ├── all/                      # Consolidated documents
│   ├── dnd_dm/                   # D&D Dungeon Master Guides
│   ├── dnd_mm/                   # D&D Monster Manuals
│   ├── dnd_player/               # D&D Player Handbooks
│   ├── dnd_raven/                # Van Richten's Ravenloft Guide
│   └── vtm/                      # Vampire: The Masquerade
│
└── models/                       # Downloaded AI models cache
```

---

### check_gpu.py
**Purpose**: GPU health verification script that validates GPU availability and performance before running the reflection agent.

**Functions**:
- `check_gpu_health()`: Comprehensive GPU diagnostics checking:
  - GPU detection and identification
  - Memory status (total, used, free)
  - GPU temperature monitoring
  - PyTorch CUDA compatibility
  - Minimum memory requirement validation (4GB free recommended)

**Output**: 
- ✅ Green checks for successful validation
- ⚠️ Warnings for potential issues
- ❌ Red X for critical failures

**Usage**: Run before starting the agent: `python check_gpu.py`

---

## Source Code Directory

### Main Agent Files

#### lang_graph_langchain_tools_reflection_agent.py
**Purpose**: Advanced reflection agent using LangChain tools for structured question-answering with iterative refinement.

**Key Classes**:

#### `ReflectionAgentState` (TypedDict)
A state container that holds:
- `messages`: Conversation history (Human, AI, System messages)
- `agent_instance`: Reference to the ReflectionAgent
- `continue_refining`: Boolean flag for iteration control
- `question`: User's query
- `context`: Retrieved context from vector database

#### `QuestionResponseSchema` (Pydantic Model)
Validates LLM output for answer generation:
- `answer`: Expert response to user query
- `question`: Original question (for reference)
- `source`: Source document reference
- `context_summary`: Brief (< 500 chars) summary of used context

**Methods**:
- `has_required_fields(data)`: Validates presence of required fields
- `validate_dict_safe(data)`: Safe validation returning (bool, message) tuple

#### `CritiqueOfAnswerSchema` (Pydantic Model)
Validates LLM output for answer evaluation:
- `critique`: Evaluation text
- `clarity`: Float (0.0-1.0) rating for comprehensibility
- `succinct`: Float (0.0-1.0) rating for brevity
- `readability`: Float (0.0-1.0) rating (0=graduate level, 1=5th grade)
- `revision_needed`: Boolean for refinement decision
- `response`: Nested QuestionResponseSchema

**Methods**:
- `has_required_fields(data)`: Validates required fields presence
- `validate_dict_safe(data)`: Safe validation with error handling

#### `ReflectionAgent` (ABC)
Main agent class for structured question-answering workflow.

**Constructor**:
```python
__init__(self, brain, root_path="/home/jmitchall/vllm-srv", **kwargs)
```
- `brain`: LLM instance for generation and critique
- `root_path`: Base path for vector databases
- Initializes generation and reflection chains

**Static Methods**:
- `get_response_llm_results(json_str)`: Parses JSON to QuestionResponseSchema
- `get_critique_llm_results(json_str)`: Parses JSON to CritiqueOfAnswerSchema
- `extract_json_response_output(raw_output)`: Cleans LLM JSON output for answers
- `extract_json_reflection_output(raw_output)`: Cleans LLM JSON output for critiques

**Dependencies**:
- `inference.vllm_srv.cleaner`: GPU memory management
- `vectordatabases.refresh_question_context_tool`: Context retrieval
- `langchain_core`: Message handling and prompts

---

#### lang_graph_manual_tools_reflection_agent.py
**Purpose**: Reflection agent with manually-invoked tool calls (as opposed to automatic tool binding).

**Key Classes**:

#### `ReflectionAgent` (ABC)
Agent with manual tool invocation capability.

**Constructor**:
```python
__init__(self, brain, embedding_model="BAAI/bge-large-en-v1.5", 
         root_path="/home/jmitchall/vllm-srv", similarity_threshold=0.7, **kwargs)
```

**Key Methods**:
- `manually_call_tools(tool_calls)`: Executes tool calls passed as dictionaries
  - Expected format: `[{"name": "method_name", "args": {...}}]`
  - Dynamically calls instance methods matching tool names

- `refresh_question_context(question, sources, db_type)`: Retrieves context for question
  - Supports sources: D&D materials, Vampire: The Masquerade
  - Works with vector DB types: "qdrant", "faiss", "chroma"
  - Returns context string for question answering

**Dependencies**:
- `embeddings.huggingface_transformer.langchain_embedding`: HuggingFace embeddings
- Same schema classes as langchain_tools version

---

#### lang_graph_structured_reflection_agent.py
**Purpose**: Reflection agent with structured output formatting and retriever integration.

**Key Classes**:

#### `ReflectionAgent` (ABC)
Structured version with advanced retriever capabilities.

**Constructor**:
```python
__init__(self, brain, embedding_model="BAAI/bge-large-en-v1.5",
         root_path="/home/jmitchall/vllm-srv", similarity_threshold=0.65, **kwargs)
```

**Key Methods**:
- `refresh_question_context(question)`: Updates cached context for current question
  - Checks if question changed before re-retrieving
  - Stores results in `self.question` and `self.context`

- `get_initial_state(question)`: Creates initial agent state for workflow
  - Returns: ReflectionAgentState with messages, question, context

- `get_retriever_and_vector_stores(vdb_type, vector_db_persisted_path, collection_ref, retriever_embeddings)`: 
  - Dynamically loads vector stores (Qdrant, FAISS, Chroma)
  - Creates retriever for semantic search
  - Supports multiple database backends

- `format_list_documents_as_string(results)`: Formats search results
  - Filters by similarity threshold
  - Returns formatted context string

**Dependencies**:
- All three vector database types (Qdrant, FAISS, Chroma)
- Advanced embedding support

---

#### lang_graph_unstructured_reflection_agent.py
**Purpose**: Lightweight reflection agent without structured output schemas, for simpler use cases.

**Key Classes**:

#### `ReflectionAgent` (ABC)
Minimal agent variant without Pydantic validation schemas.

**Constructor**:
```python
__init__(self, brain, embedding_model="BAAI/bge-large-en-v1.5",
         root_path="/home/jmitchall/vllm-srv", similarity_threshold=0.65, **kwargs)
```

**Key Methods**:
- `get_initial_state(question)`: Creates state with on-demand context retrieval
- `get_retriever_and_vector_stores()`: Same as structured version
- `get_retriever()`: Gets pre-configured retriever instance
- `format_list_documents_as_string(results)`: Context formatting with threshold filtering
- `get_generation_chain()`: Creates answer generation prompt chain
- `get_reflection_chain()`: Creates critique/reflection prompt chain

**Differences from Structured Version**:
- No output schemas (QuestionResponseSchema, CritiqueOfAnswerSchema)
- More flexible LLM output handling
- Better for experimental/prototyping work

---

### RAG Pipeline Files

#### langchain_rag_generator.py
**Purpose**: Utilities for building and inspecting RAG chains with detailed debugging output.

**Functions**:

#### `inspect_chat_prompt_value(prompt_value)`
Debugs ChatPromptTemplate structures before sending to LLM.
- Prints message count and types
- Shows content of each message
- Displays string representation
- **Must return** prompt_value to continue chain

#### `inspect_llm_input(llm_input, chain_name="")`
Inspects LLM input before processing.
- Shows input type and class
- Prints content and length
- Optional chain label for tracking
- Returns input unchanged

#### `inspect_llm_output(llm_output, chain_name="")`
Inspects LLM output after generation.
- Shows output type and class
- Displays generated content
- Optional chain label
- Returns output unchanged

#### `format_list_documents_as_string(results)`
Formats vector database search results.
- Takes Document objects with metadata
- Returns plain text context string
- Preserves source and page information

#### `get_retriever_and_vector_stores(vdb_type, vector_db_persisted_path, collection_ref, retriever_embeddings)`
Creates retriever from persisted vector store.
- Supports Qdrant, FAISS, Chroma
- Returns (retriever, client_reference) tuple
- Handles collection existence checking

**Usage Example**:
```python
embeddings = HuggingFaceOfflineEmbeddings(model_name="BAAI/bge-large-en-v1.5")
retriever, client = get_retriever_and_vector_stores(
    "qdrant", 
    "/path/to/vector/db", 
    "dnd_mm",
    embeddings
)
```

---

#### langchain_rag_retriever.py
**Purpose**: Utilities for retrieving and formatting documents from vector stores.

**Functions**:

#### `dict_to_str(d: dict) -> str`
Converts dictionary to formatted string.
- Used for formatting intermediate chain results
- Key-value pairs on separate lines

#### `format_list_documents_as_string(results)`
Advanced document formatting with metadata.
- Extracts source/file path
- Shows page numbers
- Displays file format
- Shows similarity scores

**Sample Output**:
```
Metadata Source/File path: Monster Manual A.pdf
page: 42
format: pdf
Total Pages in Source/File: 150
Similarity Score: 0.87
================================================================================
```

#### `get_retriever_and_vector_stores(vdb_type, vector_db_persisted_path, collection_ref, retriever_embeddings)`
Same as in langchain_rag_generator.py - creates retriever instances.

---

#### langchain_ingestion_vector_db.py
**Purpose**: End-to-end document ingestion pipeline for vector database creation and population.

**Functions**:

#### `consolidate_collections_to_all(root_data_path, document_collections)`
Merges multiple document collections into single directory.
- Creates symlinks to avoid duplication
- Supports: vtm, dnd_dm, dnd_mm, dnd_raven, dnd_player
- Target: `/data/all/` directory

**Example**:
```python
consolidate_collections_to_all(
    root_data_path="/home/jmitchall/vllm-srv/data",
    document_collections=["dnd_mm", "dnd_dm", "vtm"]
)
```

#### `load_and_chunk_texts(chunker, max_token_validator, **kwargs) -> (List[str], List[Document])`
Unified document loading and chunking interface.

**Process**:
1. Loads documents using provided chunker
2. Calculates average words-per-token ratio
3. Computes optimal chunk size/overlap for embedding model
4. Chunks documents
5. Validates chunks against token limits
6. Returns cleaned text chunks and Document objects

**Supported Chunkers**:
- `DocumentChunker` (LangChain-based)
- `LlamaIndexDocumentChunker` (LlamaIndex-based)
- Any class implementing `BaseDocumentChunker`

#### `load_langchain_texts(d_path, max_token_validator, **kwargs)`
Legacy wrapper for LangChain-based chunking.

#### `load_llamaIndex_texts(d_path, max_token_validator, **kwargs)`
Legacy wrapper for LlamaIndex-based chunking.

#### `ingest_documents(vector_db_collection_name, chunked_documents, persistence_target_path, embedding_object)`
Populates vector database with chunked documents.

**Supported Databases**:
- Qdrant: Client-based with persistent storage
- FAISS: File-based index storage
- Chroma: SQLite-backed persistence

---

### Embeddings Module

#### src/embeddings/embedding_manager.py
**Purpose**: High-level manager for text-to-vector embeddings using local transformer models.

**Class**: `EmbeddingManager`

**Constructor**:
```python
__init__(self, embedding_model_instance: EmbeddingModelInterface, embedding_dim: int = 1024)
```
- Takes configured embedding model instance
- Stores embedding dimension for validation

**Methods**:

#### `get_embedding(text: str) -> np.ndarray`
Converts single text string to embedding vector.
- Returns numpy array of numbers
- Each word's semantic meaning encoded as numbers
- Similar words = similar number patterns

#### `get_embeddings(texts: List[str]) -> List[np.ndarray]`
Batch process multiple texts (more efficient).
- Processes all texts together
- Uses GPU acceleration when available
- Returns list of embedding vectors

#### `get_actual_embedding_dimension() -> int`
Validates embedding dimension by testing with sample text.

---

#### src/embeddings/embedding_model_interace.py
**Purpose**: Abstract base class defining embedding model interface.

**Class**: `EmbeddingModelInterface` (ABC)

**Constructor**:
```python
__init__(self, max_tokens: int)
```

**Properties**:
- `max_tokens`: Maximum tokens model can process
- `safe_embedding_dim`: Safe embedding dimension after truncation

**Methods**:

#### `truncate_text(text: str) -> str`
Truncates text to fit within token limits.
- Uses 0.75 tokens-per-word approximation
- Preserves important information

#### `get_embedding(text: str)` (Abstract)
Must be implemented by subclasses.

#### `get_embeddings(texts: List[str]) -> List[np.ndarray]` (Abstract)
Must be implemented by subclasses.

#### `get_supported_model_token_limits() -> Dict[str, int]` (Abstract)
Returns token limits for different model architectures.

---

#### src/embeddings/huggingface_transformer/langchain_embedding.py
**Purpose**: HuggingFace embedding implementation compatible with LangChain.

**Class**: `HuggingFaceOfflineEmbeddings` (extends LangChain Embeddings)

**Constructor**:
```python
__init__(self, model_name: str, torch_dtype: str = "float16")
```

**Parameters**:
- `model_name`: HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
- `torch_dtype`: Number precision
  - "float16": Half precision (memory efficient)
  - "float32": Full precision (more accurate)
  - "bfloat16": Brain float (AI optimized)

**Initialization Steps**:
1. Loads tokenizer from HuggingFace
2. Loads AI model (downloads if needed, caches locally)
3. Automatically detects and configures GPU
4. Sets model to evaluation mode

**Properties**:
- `get_tokenizer`: Returns tokenizer for text splitting
- `max_tokens`: Returns model's token limit

**GPU Support**:
- Automatic GPU detection via PyTorch
- Falls back to CPU if GPU unavailable
- Prints status (⚡ GPU or 💻 CPU)

**Supported Models** (examples):
- "BAAI/bge-large-en-v1.5" (1024 dims, recommended)
- "sentence-transformers/all-MiniLM-L6-v2" (384 dims, faster)
- Any HuggingFace embedding model

---

#### src/embeddings/huggingface_transformer/local_model.py
**Purpose**: Direct HuggingFace model implementation of EmbeddingModelInterface.

**Class**: `HuggingFaceLocalModel` (extends EmbeddingModelInterface)

**Constructor**:
```python
__init__(self, model_name: str, max_tokens: int = 0, 
         torch_dtype: str = "float16", safety_level: str = "recommended")
```

**Methods**:

#### `calculate_avg_words_per_token(documents: List[str]) -> float`
Measures actual words-per-token ratio from sample documents.
- Analyzes first 10 documents for speed
- Returns average ratio (typically ~0.75 for English)
- Used for optimal chunk size calculation

#### `load_local_model()`
Loads model using transformers library (not vLLM).
- Downloads and caches model files
- Cache locations:
  - Linux/Mac: `~/.cache/huggingface/hub/`
  - Windows: `C:\Users\<username>\.cache\huggingface\hub\`
- Caches automatically reused on subsequent runs

---

### Ingestion Module

#### src/ingestion/base_document_chunker.py
**Purpose**: Abstract base class for all document chunking implementations.

**Class**: `BaseDocumentChunker` (ABC)

**Properties**:
- `chunk_size`: Characters per chunk
- `chunk_overlap`: Character overlap between chunks

**Abstract Methods** (must be implemented):

#### `directory_to_documents() -> List[Document]`
Loads all documents from directory.
- Returns LangChain Document objects
- Handles multiple file types (.pdf, .txt, etc.)

#### `chunk_documents(documents) -> List[Document]`
Splits documents into smaller chunks.
- Preserves metadata
- Maintains overlap between chunks

#### `get_chunked_texts_list(documents) -> (List[str], List[Document])`
Helper returning both text strings and Document objects.

#### `calculate_optimal_chunk_parameters_given_max_tokens(max_tokens, avg_words_per_token)`
Computes ideal chunk size/overlap for embedding model.
- Considers model's token limits
- Accounts for actual words-per-token ratio
- Returns (chunk_size, chunk_overlap) tuple

#### `validate_and_fix_chunks(chunk_texts, max_tokens) -> List[str]`
Ensures all chunks fit within model token limits.
- Splits oversized chunks
- Validates final output

---

#### src/ingestion/doc_parser_langchain.py
**Purpose**: LangChain-based document parsing and chunking.

**Class**: `DocumentChunker` (extends BaseDocumentChunker)

**Supported File Types**:
- `.pdf`: PyMuPDFLoader (handles complex layouts)
- `.txt`: TextLoader (plain text)

**Methods**:

#### `directory_to_documents() -> List[Document]`
Recursively loads all PDFs and TXTs.
- Validates directory existence
- Maintains file path in metadata
- Returns list of loaded documents

#### `chunk_document(document: Document) -> List[Document]`
Simple sliding-window chunking.
- Uses character count (not tokens)
- Preserves chunk_index in metadata
- Handles custom chunk_size and overlap

---

#### src/ingestion/doc_parser_llama_index.py
**Purpose**: LlamaIndex-based document parsing with advanced splitting strategies.

**Class**: `LlamaIndexDocumentChunker` (extends BaseDocumentChunker)

**Features**:
- **Token-based chunking**: Measures in tokens (more accurate than characters)
- **Advanced splitters supported**:
  - SemanticSplitterNodeParser: Splits on semantic boundaries
  - SentenceSplitter: Splits on sentence boundaries
  - HierarchicalNodeParser: Creates parent-child document relationships

**File Extraction**:
- PyMuPDFReader for PDFs (better complex layout handling)
- SimpleDirectoryReader for general files

**Conversion**:
- `to_langchain_format()`: Converts LlamaIndex nodes to LangChain Documents
- Preserves all metadata (source, page, total_pages, etc.)
- Maintains hierarchical relationships if present

---

#### src/ingestion/hierarchical_retriever_example.py
**Purpose**: Advanced example of hierarchical document storage and auto-merging retrieval.

**Classes**:

#### `HierarchicalNodeStore`
In-memory storage for parent-child document relationships.

**Methods**:
- `add_nodes(nodes)`: Stores all nodes with relationships
- `get_node(node_id)`: Retrieves specific node
- `get_parent(node_id)`: Gets parent of given node

#### Key Function: `convert_nodes_to_langchain_with_metadata()`
Converts LlamaIndex hierarchical nodes to LangChain format.
- Preserves parent-child metadata
- Marks leaf vs parent nodes
- Maintains all original metadata

**Use Case**: Auto-merging retrieval (fetch small chunks, merge into parent for context).

---

## Inference Module

#### src/inference/vllm_srv/vllm_process_manager.py
**Purpose**: Manages vLLM execution in separate process to avoid multiprocessing conflicts.

**Class**: `VLLMProcessManager`

**Constructor**:
```python
__init__(self, download_dir: str = "./models", 
         gpu_memory_utilization: float = 0.5, 
         max_model_len: int = 4096)
```

**Methods**:

#### `create_vllm_script() -> str`
Generates temporary Python script for vLLM initialization.
- Configures download directory
- Sets GPU memory limits
- Sets maximum context length
- Performs test generation

#### `start_vllm_process() -> bool`
Launches vLLM in separate process.
- Creates temporary script
- Spawns subprocess with timeout (10 minutes)
- Communicates stdout/stderr
- Handles cleanup on timeout

**Purpose**: Works around Python multiprocessing issues with vLLM.

---

#### src/inference/vllm_srv/cleaner.py
**Purpose**: GPU memory cleanup and shutdown utilities to prevent engine crashes.

**Functions**:

#### `cleanup_vllm_engine(llm_instance)`
Gracefully shuts down vLLM to prevent "Engine core proc died" errors.

**Process**:
1. Suppresses warnings during cleanup
2. Accesses underlying vLLM client
3. Deletes Python object references
4. Forces garbage collection
5. Catches non-critical errors

**Usage**:
```python
try:
    # Run your LLM code
finally:
    cleanup_vllm_engine(llm_instance)
```

#### `suppress_vllm_shutdown_errors()`
Redirects stderr at Python exit to hide cosmetic error messages.

**Note**: Use only as last resort if proper cleanup doesn't prevent error display.

#### `check_gpu_memory_status() -> dict`
Returns current GPU memory information.
- Total memory
- Used memory
- Free memory
- Temperature

---

### Vector Databases Module

#### src/vectordatabases/vector_db_interface.py
**Purpose**: Abstract base class defining vector database interface.

**Class**: `VectorDBInterface` (ABC)

**Abstract Methods**:

#### `add_documents(docs, embeddings)`
Adds documents and embeddings to database.

#### `search(query_embedding, top_k=5) -> List[Document]`
Searches for similar documents.

#### `save()`
Persists database to disk.

#### `load() -> bool`
Loads database from disk.

#### `get_total_documents() -> int`
Returns document count.

#### `get_embedding_dim() -> int`
Returns embedding vector dimension.

#### `get_max_document_length() -> int`
Returns maximum supported document length.

#### `delete_collection(**kwargs) -> bool`
Deletes collection and associated files.

---

#### src/vectordatabases/vector_db_factory.py
**Purpose**: Factory pattern for creating vector database instances.

**Class**: `VectorDBFactory`

**Static Methods**:

#### `create_vector_db(db_type, embedding_dim, persist_path, **kwargs) -> VectorDBInterface`
Creates appropriate database instance.
- `db_type`: "faiss", "qdrant", or "chroma"
- Handles initialization parameters

#### `get_available_databases() -> dict`
Checks what databases are installed.
- GPU support info
- Installation instructions
- Availability status

#### `get_vector_db(db_path) -> VectorDBInterface`
Loads persisted database from path.
- Auto-detects type from path naming

#### `get_available_vector_databases(validated_db) -> bool`
Validates specific database is available.
- Prints status with emoji indicators
- Returns boolean for availability

---

#### src/vectordatabases/qdrant_vector_db.py
**Purpose**: Qdrant vector database implementation.

**Class**: `QdrantVectorDB` (extends VectorDBInterface)

**Constructor**:
```python
__init__(self, embedding_dim: int = 0, persist_path: Optional[str] = None,
         collection_name: str = None, host: str = None, port: int = None,
         use_gpu: bool = True, prefer_server: bool = False, **kwargs)
```

**Features**:
- GPU optimization support
- Local or remote server mode
- Persistent storage on disk
- Collection-based organization
- Advanced filtering and metadata support

**Methods**:
- `get_list_of_collections()`: Returns available collections

---

#### src/vectordatabases/faiss_vector_db.py
**Purpose**: FAISS vector database implementation (Facebook AI Similarity Search).

**Class**: `FaissVectorDB` (extends VectorDBInterface)

**Constructor**:
```python
__init__(self, embedding_dim: int = 0, persist_path: Optional[str] = None,
         use_gpu: bool = True, gpu_memory_fraction: float = 0.8, **kwargs)
```

**Components**:
- `self.index`: FAISS index for vector storage
- `self.documents`: Parallel list of Document objects
- `self.gpu_available`: GPU acceleration flag
- `self.gpu_resources`: FAISS GPU resource manager

**Features**:
- Extremely fast similarity search
- GPU acceleration (if available)
- Low-level control over indexing
- Multiple index types (IVF, HNSW, PQ, OPQ)

**GPU Setup**:
- Automatic CUDA detection
- Memory fraction allocation (default 80%)
- Fallback to CPU if issues detected

---

#### src/vectordatabases/retriever_strategies.py
**Purpose**: Configurable retrieval strategies for better search quality.

**Class**: `RetrievalConfig` (Dataclass)

**Parameters**:
- `k`: Number of results (default: 5)
- `search_type`: "similarity" or "mmr"
- `score_threshold`: Min confidence (0-1, optional)
- `fetch_k`: Candidates for MMR (default: 20)
- `lambda_mult`: Relevance vs diversity balance (0.0-1.0)

**Pre-configured Strategies**:

#### `DEFAULT_CONFIG`
- 5 results, pure similarity, no threshold
- Use for: general purpose queries

#### `HIGH_PRECISION_CONFIG`
- 3 results, similarity only, 0.8 threshold
- Use for: high-confidence answers needed

#### `DIVERSE_CONFIG`
- 5 results, MMR with λ=0.5
- Use for: avoiding redundant results

#### `COMPREHENSIVE_CONFIG`
- 10 results, similarity, 0.6 threshold
- Use for: thorough coverage needed

**Function**: `get_retrieval_strategy(strategy_name) -> RetrievalConfig`
Retrieves pre-configured strategy by name.

---

#### src/vectordatabases/refresh_question_context_tool.py
**Purpose**: LangChain tool wrapper for refreshing question context from vector database.

**Classes**:

#### `RefreshQuestionContextInput` (Pydantic)
Input schema for context refresh tool.
- `question`: User's query
- `sources`: List of collection names
- `db_type`: "qdrant", "faiss", or "chroma"
- `root_path`: Base path for vector databases

#### `RefreshQuestionContextTool` (extends BaseTool)
LangChain-compatible tool for agent use.

**Methods**:
- `get_retriever(collection_name, dbtype, root_path)`: Creates retriever
- `get_retriever_and_vector_stores()`: Loads vector database

**Collections Supported**:
- "dnd_dm": Dungeon Master's Guide
- "dnd_mm": Monster Manual
- "dnd_player": Player's Handbook
- "dnd_raven": Van Richten's Ravenloft Guide
- "vtm": Vampire: The Masquerade

---

#### src/vectordatabases/mmr_with_scores.py
**Purpose**: Utilities for combining similarity scores with MMR diversity results.

**Functions**:

#### `compare_similarity_vs_mmr(vectorstore, query, k=5, fetch_k=20, lambda_mult=0.5)`
Compares pure similarity vs MMR results side-by-side.

**Returns**: `(similarity_results, mmr_results)`
- Both are lists of Documents
- Similarity results include distance/similarity scores
- MMR results include diversity ranking

#### `print_comparison(sim_docs, mmr_docs)`
Pretty-prints side-by-side comparison.
- Shows scoring for similarity results
- Displays MMR rankings
- Calculates overlap analysis

**Example Output**:
```
SIMILARITY SEARCH (Pure Relevance)
1. Distance: 0.1234 | Score: 0.8766
   Source: Monster Manual A.pdf (page 42)
   Content: [excerpt...]

MMR SEARCH (Relevance + Diversity)
1. Rank #1
   Source: Monster Manual B.pdf (page 15)
   Content: [different excerpt...]
```

---

## Vector Database Command Files (Advanced)

### src/vectordatabases/qdrant_vector_db_commands.py
**Purpose**: Helper functions and utilities for Qdrant database operations.

**Class**: `QdrantClientSmartPointer`
Smart connection manager that safely handles Qdrant client lifecycle.

**Constructor**:
```python
__init__(self, vector_db_persisted_path: str)
```

**Methods**:
- `get_client()`: Returns active Qdrant client (lazy initialization)
- `close()`: Safely closes connection with error handling
- `__del__()`: Destructor ensures cleanup on object deletion

**Functions**:

#### `get_quadrant_client(vector_db_persisted_path)`
Creates and returns a Qdrant client for local persistent storage.
- Returns: `QdrantClientSmartPointer` instance
- Handles connection pooling automatically

#### `quadrant_does_collection_exist(qdrant_client_ptr, collection_name) -> bool`
Checks if a named collection exists in database.

#### `qdrant_create_from_documents(client, collection_name, documents, embeddings)`
Creates Qdrant collection and populates with documents.

#### `get_qdrant_retriever(client, collection_name, embeddings, k=5)`
Returns LangChain retriever for semantic search.

---

### src/vectordatabases/chroma_vector_db_commands.py
**Purpose**: Chroma-specific vector database command wrapper.

**Functions**:
- `get_chroma_vectorstore()`: Creates ChromaDB instance
- `get_chroma_retriever()`: Creates retriever wrapper
- `chroma_create_from_documents()`: Populates ChromaDB from documents

---

### src/vectordatabases/chroma_vector_db.py
**Purpose**: Full ChromaDB implementation with GPU optimization.

**Class**: `ChromaVectorDB` (extends VectorDBInterface)

**Constructor**:
```python
__init__(self, embedding_dim: int = 0, persist_path: Optional[str] = None,
         collection_name: str = None, use_gpu: bool = True, **kwargs)
```

**Methods**:

#### `get_list_of_collections() -> List[str]`
Returns all collections in ChromaDB instance.

#### `check_gpu_availability()`
Detects GPU support and optimizes HNSW parameters.
- Increases construction_ef and search_ef on GPU
- Tunes M parameter for optimal search speed

#### `get_max_document_length() -> int`
Returns longest document in collection.

---

### src/vectordatabases/fais_vector_db_commands.py
**Purpose**: FAISS-specific command wrappers and helpers.

**Functions**:
- `create_faiss_vectorstore()`: Creates FAISS index from path
- `get_faiss_retriever()`: Returns LangChain retriever
- `faiss_create_from_documents()`: Populates FAISS from documents

---

## Inference Module (Advanced Models)

### src/inference/vllm_srv/minstral_langchain.py
**Purpose**: Mistral-7B LLM integration with LangChain using vLLM backend.

**Functions**:

#### `messages_to_mistral_prompt(messages: Sequence[BaseMessage]) -> str`
Converts LangChain message objects to Mistral prompt format.

**Input**: List of HumanMessage, AIMessage, SystemMessage objects
**Output**: Formatted Mistral instruction prompt

**Example**:
```python
[SystemMessage("You are helpful"),
 HumanMessage("What is AI?"),
 AIMessage("AI is...")] 
→ "<s>[INST] You are helpful\n\nHuman: What is AI? [/INST]"
```

#### `format_tool_calls_for_prompt(tool_calls: List[Dict]) -> str`
Formats structured tool calls for model prompt.

#### `parse_tool_calls(response_text: str) -> (List[Dict], str)`
Extracts tool calls from model output.
- Parses TOOL_CALL: JSON formatted calls
- Returns list of tool calls and cleaned response text

#### `get_langchain_vllm_mistral_quantized() -> BaseChatModel`
Creates LangChain chat model wrapper for Mistral.
- Uses GPTQ quantization for memory efficiency
- Auto-configures temperature, max_tokens

#### `create_vllm_chat_model(model_name="mistral", **kwargs) -> BaseChatModel`
Factory function to create vLLM-backed chat models.
- Supports model selection
- Configures sampling parameters
- Sets GPU memory limits

#### `convert_chat_prompt_to_minstral_prompt_value(prompt_value: ChatPromptValue) -> StringPromptValue`
Converts ChatPromptTemplate output to Mistral format.

---

### src/inference/vllm_srv/vllm_basic.py
**Purpose**: Simple example demonstrating basic vLLM text generation.

**Example Code**:
```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, my name is"], sampling_params)
```

**Key Concepts**:
- `LLM`: Main vLLM instance for inference
- `SamplingParams`: Controls generation behavior (temperature, top_p, etc.)
- `generate()`: Produces text from prompts

---

### src/inference/vllm_srv/vllm_memory_helper.py
**Purpose**: GPU memory management and optimization for vLLM.

**Functions**:
- Monitor GPU memory usage
- Optimize batch sizes based on available memory
- Warn on memory pressure

---

### src/inference/vllm_srv/vllm_process_manager.py
**Purpose**: Launches vLLM in separate subprocess to avoid multiprocessing conflicts.

**Class**: `VLLMProcessManager`

**Methods**:

#### `create_vllm_script() -> str`
Generates Python script for subprocess.
- Configures model download directory
- Sets GPU memory fraction
- Sets maximum context length

#### `start_vllm_process() -> bool`
Spawns vLLM in subprocess.
- 10-minute timeout
- Handles stdout/stderr capture
- Cleans up on completion or timeout

---

### src/inference/vllm_srv/facebook_langchain.py
**Purpose**: Facebook OPT model integration with LangChain.

**Functions**:
- Creates LangChain wrapper for OPT models
- Configures OPT-specific parameters
- Handles model loading and optimization

---

### src/inference/vllm_srv/minstral_llmlite.py
**Purpose**: Mistral model integration using llmlite library.

**Alternative to vLLM**: Uses different inference backend for comparison/compatibility.

---

### src/inference/vllm_srv/facebook_llmlite.py
**Purpose**: Facebook OPT models using llmlite backend.

**Comparison with vLLM**:
- llmlite: More control, slower setup
- vLLM: Optimized performance, easier integration

---

### src/inference/vllm_srv/microsoft.py
**Purpose**: Microsoft model integrations (if applicable).

**Supported Models**: Models from Microsoft's model zoo.

---

## Embeddings Module (Advanced)

### src/embeddings/vllm/langchain_embedding.py
**Purpose**: vLLM-based embeddings (alternative to HuggingFace).

**Class**: `VLLMOfflineEmbeddings` (extends LangChain Embeddings)

**Constructor**:
```python
__init__(self, model_name: str, 
         tensor_parallel_size: int = 1,
         gpu_memory_utilization: float = 0.9,
         device: str = "cuda")
```

**Parameters**:
- `tensor_parallel_size`: GPU count for model parallelization
  - Default: 1 (single GPU)
  - Set to 2+ to split large models across GPUs
  
- `gpu_memory_utilization`: Fraction of GPU memory to use (0.0-1.0)
  - 0.9: Use 90% of available GPU memory
  - 0.7: Conservative, leaves room for other tasks
  - 0.5: Very conservative, slower but stable

- `device`: "cuda" for GPU or "cpu" for CPU
  - Automatically falls back to CPU if CUDA unavailable

**Supported Models**:
- "intfloat/e5-mistral-7b-instruct" (512 tokens)
- "BAAI/bge-base-en-v1.5" (512 tokens)
- "BAAI/bge-large-en-v1.5" (1024 tokens)
- "intfloat/e5-large-v2" (512 tokens)

---

### src/embeddings/huggingface_transformer/llama_index_embedding_derived.py
**Purpose**: HuggingFace embeddings wrapper for LlamaIndex.

**Class**: `LlamaIndexHuggingFaceOfflineEmbeddings` (extends LlamaIndex BaseEmbedding)

**Use Case**: When using LlamaIndex-based document chunking and retrieval.

---

## Project Dependencies Overview

### Core AI/ML Libraries

#### **vLLM** (>= 0.6.0)
- **Purpose**: Fast LLM inference engine
- **Key Features**: 
  - GPU-optimized inference (10-20x faster than standard)
  - Token batching and memory management
  - Multi-GPU support
- **Usage**: Text generation and chat completions
- **GPU Memory**: ~8-16GB for Mistral-7B with context length 4096

#### **LangChain** (0.3.27)
- **Purpose**: Framework for building LLM applications
- **Key Features**:
  - Prompt engineering utilities
  - Chain/workflow orchestration
  - RAG pipeline builders
  - Tool/agent integration
- **Usage**: Building generation chains, retrieval pipelines, agents
- **Main Components**:
  - `langchain-core`: Base abstractions (0.3.78)
  - `langchain-community`: Integrations (0.3.30)
  - `langchain-text-splitters`: Document chunking (0.3.11)
  - `langchain-qdrant`: Qdrant integration (0.2.1)
  - `langchain-chroma`: Chroma integration (0.2.6)
  - `langchain-openai`: OpenAI API support (0.3.35)

#### **LlamaIndex** ([all] >= 0.14.12)
- **Purpose**: Advanced document indexing and retrieval
- **Key Features**:
  - Hierarchical document chunking
  - Semantic node parsing
  - Multi-vector retrieval
- **Sub-packages Used**:
  - `llama-index-embeddings-huggingface` (0.6.1): HuggingFace model integration
  - `llama-index-vector-stores-qdrant` (0.8.8): Qdrant backend
  - `llama-index-vector-stores-faiss` (0.5.2): FAISS backend
  - `llama-index-vector-stores-chroma` (0.5.5): Chroma backend
- **Usage**: Advanced document ingestion and chunking strategies

#### **LangGraph** (>= 1.0.1)
- **Purpose**: Agent orchestration and workflow graphs
- **Key Features**:
  - State management
  - Conditional routing
  - Loop detection and cycles
  - Agent coordination
- **Usage**: Reflection agent implementation with iterative refinement

### Vector Databases

#### **Qdrant** (qdrant-client < 1.16)
- **Purpose**: Production-grade vector database
- **Key Features**:
  - Persistent storage (local or remote)
  - Metadata filtering
  - Collection-based organization
  - HTTP/gRPC server support
- **Storage**: File-based on disk
- **Recommended for**: Production systems, advanced filtering needs

#### **FAISS** (faiss-cpu == 1.12.0)
- **Purpose**: Fast similarity search
- **Key Features**:
  - Extremely fast nearest neighbor search
  - Multiple index algorithms (IVF, HNSW, PQ, OPQ)
  - GPU acceleration support
- **Storage**: In-memory with manual persistence
- **Recommended for**: Maximum performance, research

#### **ChromaDB** (== 1.1.1)
- **Purpose**: Developer-friendly vector database
- **Key Features**:
  - Built-in LangChain integration
  - Simple SQLite backend
  - On-disk persistence
  - Good defaults for small-medium projects
- **Recommended for**: Rapid prototyping, simple use cases

### Embedding Models

#### **Hugging Face Transformers** (>= 4.30.0, != 4.57.0)
- **Purpose**: Load and run transformer models locally
- **Key Features**:
  - AutoModel and AutoTokenizer for any HF model
  - GPU support via PyTorch
  - Model caching
- **Models Used**:
  - BAAI/bge-large-en-v1.5: 1024-dim embeddings (recommended)
  - sentence-transformers/all-MiniLM-L6-v2: 384-dim (faster)

#### **PyTorch** (>= 2.0.0)
- **Purpose**: Deep learning framework
- **Key Components Used**:
  - `torch`: Core tensor operations
  - `torchaudio`: Audio processing (if needed)
  - `torchvision`: Vision models (if needed)
- **GPU Support**: CUDA integration for NVIDIA GPUs
- **Memory**: Automatically manages GPU memory via PyTorch

### Document Processing

#### **PyMuPDF** (pymupdf == 1.26.7)
- **Purpose**: Extract text from PDF files
- **Features**:
  - Accurate PDF text extraction
  - Preserves layout information
  - Page-level metadata
- **Usage**: Loading D&D rulebooks and other PDFs

### API & Async Frameworks

#### **FastAPI** (>= 0.100.0)
- **Purpose**: HTTP API for serving the agent
- **Key Features**: 
  - Async support
  - Automatic API documentation
  - Request validation
- **Usage**: REST endpoint for question-answering

#### **Uvicorn** ([standard] >= 0.20.0)
- **Purpose**: ASGI web server
- **Key Features**:
  - High performance
  - Async support
  - WebSocket support
- **Usage**: Running FastAPI application

### Utilities & Dependencies

#### **Numpy** (>= 1.21.0)
- **Purpose**: Numerical computing
- **Usage**: Embedding vector operations, matrix math

#### **Pydantic** (>= 2.0.0)
- **Purpose**: Data validation and serialization
- **Usage**: Schema validation (QuestionResponseSchema, CritiqueOfAnswerSchema, etc.)
- **Key Classes**: BaseModel, Field, validator

#### **Tokenizers** (>= 0.13.0)
- **Purpose**: Fast tokenization for transformers
- **Features**:
  - High-speed tokenization
  - Special tokens handling
  - Vocabulary management

#### **Safetensors** (>= 0.3.0)
- **Purpose**: Safe loading of model weights
- **Advantages**:
  - Prevents arbitrary code execution
  - Faster loading than pickle
  - Memory safe

#### **Protobuf** (>= 3.20.0)
- **Purpose**: Serialization format
- **Usage**: gRPC communication (Qdrant, vLLM)

#### **SentencePiece** (>= 0.1.99)
- **Purpose**: Subword tokenization
- **Models Using It**: Many transformer models for non-English languages

#### **Accelerate** (>= 0.20.0)
- **Purpose**: Distributed training utilities
- **Usage**: Multi-GPU support, memory-efficient inference

#### **Ray** (>= 2.5.0)
- **Purpose**: Distributed computing framework
- **Usage**: Parallel document processing, distributed inference

#### **Typing Extensions** (>= 4.0.0)
- **Purpose**: Backported typing features
- **Usage**: Type hints in Python 3.13

### Development Tools

#### **Packaging** (>= 21.0)
- **Purpose**: Version parsing and comparison
- **Usage**: Dependency management

#### **Psutil** (>= 5.9.0)
- **Purpose**: System and process utilities
- **Usage**: Memory monitoring, process management

---

## Dependency Tree Overview

```
vllm-srv/
├── LLM Inference Layer
│   ├── vLLM (main inference engine)
│   ├── LangChain (abstraction layer)
│   └── LangGraph (agent orchestration)
│
├── Embedding Layer
│   ├── Transformers (model loading)
│   ├── PyTorch (GPU support)
│   └── HuggingFace models (BAAI, sentence-transformers)
│
├── Vector Database Layer
│   ├── Qdrant (production storage)
│   ├── FAISS (fast search)
│   └── ChromaDB (developer-friendly)
│
├── Data Processing Layer
│   ├── LlamaIndex (chunking strategies)
│   ├── PyMuPDF (PDF extraction)
│   └── Tokenizers (text splitting)
│
└── API/Infrastructure Layer
    ├── FastAPI (REST endpoints)
    ├── Uvicorn (web server)
    └── Ray (distributed processing)
```

---

## GPU Memory Requirements by Model

| Model | Context | vLLM Memory | With Embeddings |
|-------|---------|-------------|-----------------|
| Mistral-7B GPTQ | 4096 | 8-10 GB | 10-12 GB |
| Mistral-7B GPTQ | 8192 | 10-12 GB | 12-14 GB |
| BAAI-BGE-Large | N/A | N/A | 4-6 GB |
| Qdrant (100K docs) | N/A | N/A | 2-4 GB |
| **Total Recommended** | 4096 | **16 GB** | **20 GB** |

---

## Installation Quick Reference

### Basic Installation
```bash
uv sync
```

### GPU Support
```bash
# For CUDA 12.1+
uv sync --extra cuda

# Or manually:
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Optional Vector Databases
```bash
# FAISS GPU support (requires CUDA)
uv add faiss-gpu

# Qdrant server (if running separately)
uv add qdrant-server
```

---

## Dependency Installation Troubleshooting

### PyTorch/CUDA Issues
- Ensure NVIDIA driver is installed: `nvidia-smi`
- Verify CUDA compatibility: Check PyTorch wheel compatibility
- Use CPU version if GPU unavailable: `faiss-cpu` instead of `faiss-gpu`

### Memory Issues During Installation
- Use `--no-binary` flag for large packages
- Install incrementally to identify problem packages
- Check available disk space for model downloads

### Model Download Issues
- Ensure internet connection for HuggingFace model downloads
- Set `HF_HOME` environment variable if custom cache needed
- Pre-download models with: `huggingface-cli download <model_id>`

---

## Documentation Files

### src/vectordatabases/Vector_DB.md
Comparison guide for three supported vector databases:
- **Qdrant**: Production-grade with metadata filtering
- **FAISS**: Fastest, GPU-optimized, low-level control
- **Chroma**: Developer-friendly, simple defaults

---

## Configuration Files

### pyproject.toml
Project metadata and dependency management using UV package manager.

### requirements.txt
Python package dependencies for CPU environments.

### requirements-cuda.txt
Additional dependencies for CUDA GPU support.

---

## Data Directory Structure

### Collection Types
- **dnd_dm/**: D&D Dungeon Master's Guide (chapters 1-8, Magic Items, Maps, Lore)
- **dnd_mm/**: D&D Monster Manual (A-Z, all creatures)
- **dnd_player/**: D&D Player's Handbook
- **dnd_raven/**: Van Richten's Guide to Ravenloft (Horror)
- **vtm/**: Vampire: The Masquerade rules and lore
- **all/**: Consolidated symlinks to above for unified search

### Models Directory
Downloaded LLM and embedding model cache:
- Stores downloaded models to avoid re-downloading
- Uses HuggingFace cache conventions
- Includes Mistral-7B and BAAI embeddings

---

## Quick Start Guide

### 1. Check GPU Status
```bash
python check_gpu.py
```

### 2. Load Documents
```python
from src.ingestion.doc_parser_langchain import DocumentChunker
from src.langchain_ingestion_vector_db import load_and_chunk_texts

chunker = DocumentChunker("/path/to/documents")
texts, docs = load_and_chunk_texts(chunker, max_token_validator=512)
```

### 3. Create Vector Database
```python
from src.vectordatabases.vector_db_factory import VectorDBFactory
from src.embeddings.huggingface_transformer.langchain_embedding import HuggingFaceOfflineEmbeddings

embeddings = HuggingFaceOfflineEmbeddings(model_name="BAAI/bge-large-en-v1.5")
db = VectorDBFactory.create_vector_db("qdrant", embedding_dim=1024)
db.add_documents(docs, embeddings.get_embeddings([d.page_content for d in docs]))
db.save()
```

### 4. Query with Agent
```python
from src.lang_graph_structured_reflection_agent import ReflectionAgent
from src.inference.vllm_srv.minstral_langchain import create_vllm_chat_model

llm = create_vllm_chat_model()
agent = ReflectionAgent(brain=llm, DATABASE_TYPE="qdrant", collection_name="dnd_mm")
question = "What is a Rogue?"
state = agent.get_initial_state(question)
```

---

## System Requirements

- **GPU**: NVIDIA RTX 5080 (16GB VRAM recommended)
- **CPU**: Multi-core processor (8+ cores beneficial)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ for models and vector databases
- **OS**: Linux/WSL2 (Windows via WSL)
- **CUDA**: 11.8+ for GPU acceleration

---

## Troubleshooting

### "Engine core proc died unexpectedly"
**Cause**: Improper vLLM cleanup during shutdown
**Solution**: Always call `cleanup_vllm_engine(llm)` in finally block

### GPU Memory Issues
**Solution**: Run `check_gpu.py` to verify 4GB+ free memory available

### Document Loading Failures
**Solution**: Ensure PDF permissions readable, check file formats are supported

### Vector Database Errors
**Solution**: Verify database type matches persistence path naming convention

---

## Contributing

When adding new agents, document:
1. Class inheritance (ABC, ReflectionAgent, etc.)
2. Constructor parameters
3. Public methods and their signatures
4. Dependencies (imports)
5. Example usage

---

## License and Attribution

This project uses:
- **vLLM**: Served under Apache 2.0
- **LangChain**: MIT License
- **LlamaIndex**: MIT License
- **FAISS**: MIT License
- **Qdrant**: AGPL-3.0 (with commercial licensing available)

See respective projects for full terms.

---

**Last Updated**: January 2026
**Documentation Version**: 1.0
