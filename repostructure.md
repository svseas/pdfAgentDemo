# Repository Structure

```
backend/
├── src/
│   ├── api/
│   │   └── v1/
│   │       ├── documents.py         # API endpoints for document operations
│   │       └── cache.py            # TODO: Cache management endpoints
│   ├── core/
│   │   ├── llm/                    # LLM service package
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── interfaces.py       # LLM interfaces
│   │   │   ├── providers.py        # LLM provider implementations
│   │   │   └── prompts.py          # Prompt template manager
│   │   ├── cache/                  # TODO: Cache management package
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── interfaces.py       # Cache interfaces
│   │   │   ├── providers.py        # Cache provider implementations
│   │   │   ├── policies.py         # Cache policies and strategies
│   │   │   └── monitoring.py       # Cache monitoring and statistics
│   │   ├── mq/                     # TODO: RabbitMQ integration
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── interfaces.py       # Queue interfaces (extensible for future implementations)
│   │   │   ├── consumer.py         # Message consumer implementations
│   │   │   ├── publisher.py        # Message publisher implementations
│   │   │   └── monitoring.py       # Queue monitoring and metrics
│   │   ├── config.py               # Configuration settings
│   │   ├── database.py             # Database connection and session management
│   │   ├── di.py                   # Dependency injection container
│   │   └── dependencies.py         # FastAPI dependencies
│   ├── domain/
│   │   ├── agents/                 # Agentic workflow system
│   │   │   ├── __init__.py         # Agent exports
│   │   │   ├── base_agent.py       # Base agent implementation
│   │   │   ├── citation_agent.py   # Citation extraction agent
│   │   │   ├── context_builder_agent.py  # Context retrieval agent
│   │   │   ├── query_analyzer_agent.py   # Query analysis agent
│   │   │   ├── query_synthesizer_agent.py # Answer synthesis agent
│   │   │   └── recursive_summarization_agent.py  # Document summarization agent
│   │   ├── grag/                   # Graph-based Reranking (GRAG) package
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── models.py           # AMR parsing and graph processing models
│   │   │   └── service.py          # GRAG service implementation
│   │   ├── processing/             # TODO: Performance-critical processing (future Go migration candidate)
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── pdf.py              # PDF processing operations
│   │   │   ├── vector.py           # Vector operations
│   │   │   └── batch.py            # Batch processing utilities
│   │   ├── pdf_processor.py        # PDF processing logic
│   │   ├── agentic_chunker.py      # Intelligent legal document chunking
│   │   ├── embedding_generator.py   # Async text embedding generation
│   │   ├── query_processor.py      # Query processing and response generation
│   │   ├── stepback_agent.py       # Stepback prompting agent
│   │   ├── interfaces.py           # Domain interfaces
│   │   ├── exceptions.py           # Domain exceptions
│   │   └── workflow.py             # Workflow orchestration
│   ├── models/
│   │   ├── base.py                 # Base model configuration
│   │   ├── document.py             # Document and metadata models
│   │   ├── workflow.py             # Workflow tracking models
│   │   ├── cache.py               # TODO: Cache tracking models
│   │   └── task.py                # TODO: Task tracking models
│   ├── repositories/
│   │   ├── document_repository.py  # Document and vector search
│   │   ├── workflow_repository.py  # Workflow context tracking
│   │   ├── cache_repository.py    # TODO: Cache entry management
│   │   └── task_repository.py     # TODO: Task status tracking
│   ├── schemas/
│   │   ├── document.py             # Pydantic schemas for documents
│   │   ├── cache.py               # TODO: Cache-related schemas
│   │   └── task.py                # TODO: Task-related schemas
│   ├── services/
│   │   ├── document_service.py     # Business logic layer
│   │   ├── cache_service.py       # TODO: Cache management service
│   │   └── task_service.py        # TODO: Task management service
│   ├── workers/                   # TODO: RabbitMQ workers
│   │   ├── __init__.py            # Package exports
│   │   ├── document_worker.py     # Document processing worker
│   │   ├── embedding_worker.py    # Embedding generation worker
│   │   └── agent_worker.py        # Agent workflow worker
│   └── main.py                     # FastAPI application entry point
├── migrations/                      # Alembic database migrations
├── uploads/                        # PDF file storage
├── alembic.ini                     # Alembic configuration
├── pyproject.toml                  # Project metadata and dependencies
└── requirements.txt                # Python dependencies

docs/
├── architecture.md                 # System architecture documentation
└── project_instruction.md          # Project guidelines and instructions

# Future Extension Points (Go Migration Candidates)
# - PDF processing operations (domain/processing/pdf.py)
# - Vector operations (domain/processing/vector.py)
# - Batch processing (domain/processing/batch.py)
# - Cache management (core/cache/*)
# - Message queue workers (workers/*)