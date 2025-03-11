# Repository Structure

```
backend/
├── src/
│   ├── api/
│   │   └── v1/
│   │       └── documents.py         # API endpoints for document operations
│   ├── core/
│   │   ├── config.py               # Configuration settings
│   │   ├── database.py             # Database connection and session management
│   │   └── dependencies.py         # FastAPI dependencies
│   ├── domain/
│   │   ├── grag/                   # Graph-based Reranking (GRAG) package
│   │   │   ├── __init__.py         # Package exports
│   │   │   ├── models.py           # AMR parsing and graph processing models
│   │   │   └── service.py          # GRAG service implementation
│   │   ├── pdf_processor.py        # PDF processing logic
│   │   ├── embedding_generator.py   # Text embedding generation
│   │   └── query_processor.py      # Query processing and response generation
│   ├── models/
│   │   ├── base.py                 # Base model configuration
│   │   └── document.py             # Document and metadata models
│   ├── repositories/
│   │   └── document_repository.py   # Document data access layer
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