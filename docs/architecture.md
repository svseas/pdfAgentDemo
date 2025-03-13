# Architecture Documentation

## Overview
PDF Chat demo application using RAG (Retrieval Augmented Generation) that integrates with locally running LM Studio API. The system uses RabbitMQ for message queuing, Redis for caching, and PostgreSQL with pgvector for storage. The architecture is designed to be extensible, with clear interfaces that allow for future performance optimizations through Go service integration.

## Core Components

### Service Architecture
- Primary Python Service:
  * FastAPI web application
  * Agent-based workflow system
  * LLM integration
  * Business logic
  * Performance-critical operations (future Go migration candidates)
- Extension Points:
  * PDF processing operations
  * Vector operations
  * Cache management
  * Message queue workers

### Message Queue System (RabbitMQ)
- Queue Types:
  * Document processing queue
  * Embedding generation queue
  * Agent workflow queue
  * Graph processing queue
- Features:
  * Message persistence
  * Dead letter exchanges
  * Priority queues
  * Publisher confirms
- Monitoring:
  * Queue metrics
  * Consumer health
  * Message rates
  * Error tracking

### Caching System
- Multi-level caching strategy:
  * In-memory cache (Redis)
  * Disk-based cache
  * Database materialized views
- Key cache areas:
  * Embeddings and vector calculations
  * Query results and context
  * Agent workflow intermediates
  * Graph structures and calculations
- Cache management:
  * TTL-based expiration
  * LRU eviction policy
  * Cache warming strategies
  * Invalidation protocols

### API Layer (src/api)
- RESTful API endpoints using FastAPI
- Versioned API structure (v1)
- Document operations and chat endpoints
- Dependency injection for service management
- Async task status tracking

### Core Infrastructure (src/core)
- Configuration management
- Database connection handling
- Dependency injection container
- Message queue integration
- Cache management
- LLM Service:
  * Interfaces for LLM providers
  * Provider implementations
  * Prompt template management
  * Async streaming chat completion

### Domain Layer (src/domain)

#### Performance-Critical Processing (domain/processing)
- PDF text extraction and processing
- Vector operations and similarity search
- Batch processing utilities
- Designed for future Go migration:
  * Clean interfaces
  * Minimal dependencies
  * Stateless operations
  * Performance-focused design

#### Agent System (domain/agents)
- Base agent infrastructure
- Specialized agents:
  * Citation Agent: Extracts and validates citations
  * Context Builder: Retrieves relevant document context
  * Query Analyzer: Performs query analysis with summary context
  * Query Synthesizer: Generates final responses
  * Recursive Summarization: Creates hierarchical document summaries

#### GRAG Module (domain/grag)
- Graph-based Reranking for enhanced retrieval
- AMR parsing for semantic graph construction
- Graph processing models
- GNN-based reranking service

### Data Layer

#### Models (src/models)
- SQLAlchemy ORM models
- Document and metadata models
- Workflow tracking models
- Cache tracking models
- Task status models

#### Repositories (src/repositories)
- Document repository with vector search
- Workflow repository for context tracking
- Cache entry management
- Task status tracking

## Dependencies
- fastapi: Web framework
- sqlalchemy: ORM and database operations
- pgvector: Vector similarity in PostgreSQL
- unstructured: PDF text extraction
- sentence-transformers: Text embeddings
- pika: RabbitMQ client
- redis: Cache and pub/sub
- transformers: AMRBART model
- networkx: Graph processing

## Database Schema
```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB
);

-- Workflow tracking
CREATE TABLE workflow_summaries (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    summary_text TEXT NOT NULL,
    summary_type TEXT NOT NULL,
    embedding vector(768)
);

-- Cache tracking
CREATE TABLE cache_entries (
    id SERIAL PRIMARY KEY,
    cache_key TEXT NOT NULL,
    cache_type TEXT NOT NULL,
    last_accessed TIMESTAMP,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    metadata JSONB
);

-- Task tracking
CREATE TABLE task_status (
    id SERIAL PRIMARY KEY,
    task_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Materialized views
CREATE MATERIALIZED VIEW common_document_contexts AS
SELECT d.id, d.content, d.embedding, w.summary_text
FROM documents d
JOIN workflow_summaries w ON d.id = w.document_id
WHERE w.summary_type = 'context';

-- Vector search indices
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX ON workflow_summaries USING ivfflat (embedding vector_cosine_ops);
```

## Message Queue Architecture

1. Exchanges:
   - document.processing: PDF processing tasks
   - embedding.generation: Vector operations
   - agent.workflow: Agent coordination
   - graph.processing: GRAG operations

2. Queue Configuration:
   - Durable queues for persistence
   - Dead letter exchanges for failed messages
   - Priority queues for critical operations
   - Publisher confirms for reliability

3. Worker Types:
   - Document processing workers
   - Embedding generation workers
   - Agent workflow workers
   - Graph processing workers

4. Error Handling:
   - Automatic retries with backoff
   - Dead letter queues
   - Error logging and monitoring
   - Circuit breakers for failing services

## Workflow Process

1. Document Processing:
   - PDF text extraction
   - Intelligent chunking
   - Embedding generation
   - Storage in PostgreSQL
   - Cache population:
     * Document chunks
     * Embeddings
     * Processing metadata

2. Query Processing:
   - Cache lookup for similar queries
   - Query analysis
   - Context retrieval
   - Agent-based response generation
   - Cache updates:
     * Query patterns
     * Context retrievals
     * Response templates

3. Response Generation:
   - Context-aware prompt construction
   - Citation extraction and validation
   - Streaming response delivery
   - Workflow context maintenance
   - Cache management:
     * Response patterns
     * Citation contexts
     * Workflow states

4. Performance Optimization:
   - Efficient processing algorithms
   - Distributed caching
   - Queue-based load balancing
   - Resource monitoring
   - Extension points for future Go migration

## Future Go Migration Path

1. Performance-Critical Components:
   - PDF processing operations
   - Vector operations
   - Batch processing
   - Cache management

2. Interface Requirements:
   - Clean service boundaries
   - Language-agnostic protocols
   - Minimal cross-service dependencies
   - Clear data contracts

3. Migration Strategy:
   - Identify performance bottlenecks
   - Implement Go services incrementally
   - Maintain Python fallbacks
   - Validate performance improvements

4. Integration Points:
   - RabbitMQ for service communication
   - Redis for shared caching
   - Common data models
   - Health checks and monitoring