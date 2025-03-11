# Offline PDF-Based Q&A

A FastAPI-based chatbot that enables offline question-answering using PDF documents. Built with clean architecture principles and following SOLID, KISS, and DRY practices.

## Tech Stack

- Python 3.10+
- PostgreSQL with PGvector extension
- FastAPI
- UV package manager

## Prerequisites

- Python 3.10 or higher
- PostgreSQL 15+ with PGvector extension
- UV package manager (`pip install uv`)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/svseas/pdfAgentDemo.git
cd pdfAgentDemo
```

2. Create and activate a virtual environment using UV:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies using UV:
```bash
cd backend
uv pip sync requirements.txt
```

4. Set up PostgreSQL with PGvector:
```sql
-- Run in PostgreSQL (version 15 or higher required)
CREATE DATABASE pdf_agent;
\c pdf_agent
CREATE EXTENSION IF NOT EXISTS vector;
```

5. Configure environment:
Create a `.env` file in the backend directory:
```env
DATABASE_URL=postgresql://user:password@localhost:5432/pdf_agent
OPENAI_API_KEY=your_openai_api_key
```

6. Initialize the database:
```bash
alembic upgrade head
```

## Running the Application

Start the FastAPI server:
```bash
cd backend
uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`

## Usage

### Document Processing

1. Upload a PDF document:
```bash
curl -X POST -F "file=@path/to/your/document.pdf" http://localhost:8000/api/v1/documents/upload
```

2. Ask questions about the document:
```bash
curl -X POST http://localhost:8000/api/v1/documents/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "your question about the document",
    "top_k": 5,
    "similarity_threshold": 0.3,
    "temperature": 0.7
  }'
```

## Project Architecture

The project follows clean architecture principles:

```
backend/
├── src/
│   ├── api/           # Interface adapters
│   ├── core/          # Application core settings
│   ├── domain/        # Business logic & rules
│   ├── models/        # Database entities
│   ├── repositories/  # Data access layer
│   ├── schemas/       # Data transfer objects
│   └── services/      # Use cases
└── migrations/        # Database migrations
```


