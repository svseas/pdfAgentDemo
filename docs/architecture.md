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
   - Perform similarity search to find relevant chunks
   - Retrieve top-k most relevant chunks

4. Generation:
   - Construct prompt with retrieved context
   - Send to LM Studio API with Llama3-DocChat model
   - Stream responses back to user