# Architecture Documentation

## Overview
PDF Chat demo application using RAG (Retrieval Augmented Generation) that integrates with locally running LM Studio API. The system processes PDFs, stores embeddings in PostgreSQL with pgvector, and uses Llama3-DocChat-1.0-8B for chat completions.

## Components

### PDF Processing Module (src/pdf_processor.py)
- Uses unstructured library to extract text from PDFs
- Handles PDF parsing, text cleaning, and chunking
- Generates embeddings using sentence-transformers

### Vector Store Module (src/vector_store.py)
- Manages PostgreSQL with pgvector extension
- Handles document embedding storage and retrieval
- Implements similarity search for RAG

### RAG Module (src/rag.py)
- Implements RAG logic
- Retrieves relevant context from vector store
- Constructs prompts with retrieved context

### GRAG Module (src/domain/grag)
- Implements Graph-based Reranking for enhanced retrieval
- Uses Abstract Meaning Representation (AMR) for semantic parsing
- Builds document graphs based on semantic relationships
- Applies Graph Neural Networks for context-aware reranking
- Components:
  - AMR Parser: Converts text to semantic graphs
  - Graph Processor: Extracts and analyzes graph features
  - GNN Reranker: Learns and applies graph-based relationships

### Chat Module (src/chat.py)
- Implements async streaming chat completion using httpx
- Connects to local LM Studio API (http://127.0.0.1:1234)
- Uses Server-Sent Events (SSE) format for streaming responses
- Handles JSON parsing and token streaming

## Dependencies
- httpx: For async HTTP requests
- python-dotenv: For environment variable management
- unstructured: For PDF text extraction
- pgvector: For vector similarity search in PostgreSQL
- sentence-transformers: For generating text embeddings
- pdf2image: For PDF preprocessing
- python-magic: For file type detection
- torch: Deep learning framework for GRAG
- networkx: Graph operations and analysis
- penman: AMR graph parsing
- transformers: AMRBART model for semantic parsing
- torch-geometric: Graph Neural Networks

## Database Schema
```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768)  -- Dimension depends on the embedding model
);

-- Create index for similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

## API Integration
- Uses OpenAI-compatible API endpoints provided by LM Studio
- Base URL: http://127.0.0.1:1234/v1
- Endpoints:
  - POST /chat/completions: For chat completion requests

## RAG Process Flow
1. PDF Processing:
   - Extract text from PDF using unstructured
   - Clean and chunk text into manageable segments
   - Generate embeddings for each chunk

2. Storage:
   - Store text chunks and their embeddings in PostgreSQL
   - Maintain document metadata and relationships

3. Retrieval:
   - On user query, generate query embedding
   - Initial retrieval using similarity search
   - Enhanced retrieval using GRAG (optional):
     * Build semantic graphs from documents using AMR
     * Create document graph based on shared concepts
     * Apply GNN for context-aware reranking
   - Return final set of most relevant chunks

4. Generation:
   - Construct prompt with retrieved context
   - Send to LM Studio API with Llama3-DocChat model
   - Stream responses back to user