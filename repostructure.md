# Repository Structure

```
backend/
├── src/
│   ├── api/
│   │   └── v1/
│   │       └── documents.py         # API endpoints for document operations
│   ├── core/
│   │   ├── llm/                    # LLM service package
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── interfaces.py       # LLM interfaces
│   │   │   ├── providers.py        # LLM provider implementations
│   │   │   └── prompts.py          # Prompt templates and manager
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
│   │   ├── pdf_processor.py        # PDF processing logic
│   │   ├── agentic_chunker.py      # Intelligent legal document chunking
│   │   ├── embedding_generator.py   # Text embedding generation
│   │   ├── query_processor.py      # Query processing and response generation
│   │   ├── stepback_agent.py       # Stepback prompting agent
│   │   ├── interfaces.py           # Domain interfaces
│   │   ├── exceptions.py           # Domain exceptions
│   │   └── workflow.py             # Workflow orchestration
│   ├── models/
│   │   ├── base.py                 # Base model configuration
│   │   ├── document.py             # Document and metadata models
│   │   └── workflow.py             # Workflow tracking models
│   ├── repositories/
│   │   ├── document_repository.py  # Document data access layer
│   │   └── workflow_repository.py  # Workflow data access layer
│   ├── schemas/
│   │   └── document.py             # Pydantic schemas for documents
│   ├── services/
│   │   └── document_service.py     # Business logic layer
│   └── main.py                     # FastAPI application entry point
├── migrations/                      # Alembic database migrations
├── uploads/                        # PDF file storage
├── alembic.ini                     # Alembic configuration
├── pyproject.toml                  # Project metadata and dependencies
└── requirements.txt                # Python dependencies

docs/
├── architecture.md                 # System architecture documentation
└── project_instruction.md          # Project guidelines and instructions