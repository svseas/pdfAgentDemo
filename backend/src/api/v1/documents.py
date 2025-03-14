from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import logging
import shutil
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.embedding_generator import EmbeddingGenerator
from src.domain.query_processor import QueryProcessor
from src.domain.exceptions import AgentError
from src.models.workflow import (
    UserQuery,
    OriginalUserQuery,
    SubQuery
)
from src.domain.agents.context_builder_agent import ContextBuilderAgent
from src.api.dependencies import (
    get_embedding_generator,
    get_query_processor,
    get_document_repository,
    get_document_service,
    get_summarization_agent,
    get_query_analyzer_agent,
    get_citation_agent,
    get_query_synthesizer_agent,
    get_context_builder_agent,
    get_db
)
from src.repositories.document_repository import DocumentRepository
from src.repositories.workflow_repository import SQLQueryRepository, SQLWorkflowRepository
from src.services.document_service import DocumentService
from src.domain.agents.recursive_summarization_agent import RecursiveSummarizationAgent
from src.domain.agents.query_analyzer_agent import QueryAnalyzerAgent
from src.domain.agents.citation_agent import CitationAgent
from src.domain.agents.query_synthesizer_agent import QuerySynthesizerAgent
from src.schemas.rag import (
    VectorizeRequest,
    VectorResponse,
    SearchRequest,
    ContextRequest,
    QueryRequest,
    SummarizeRequest,
    QueryAnalysisRequest,
    CitationRequest,
    SynthesisRequest,
    BuildQueryContextRequest,
    ContextBuilderResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service)
) -> dict:
    """Upload and process a PDF document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
            
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Create document metadata
        doc_metadata = await document_service.create_document(
            file_path=file_path,
            original_filename=file.filename,
            file_size=file.size
        )
        
        # Process document (extract text, generate embeddings, store chunks)
        await document_service.process_document(doc_metadata.id, file_path)
        
        return {
            "message": "Document uploaded and processed successfully",
            "document_id": doc_metadata.id,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=dict)
async def query_documents(
    request: QueryRequest,
    query_service: QueryProcessor = Depends(get_query_processor),
    doc_repository: DocumentRepository = Depends(get_document_repository)
) -> dict:
    """Complete RAG workflow: search documents and generate response"""
    try:
        # Get query embedding
        query_embedding = query_service.embedding_generator.generate_embedding(request.query)
        
        # Get similar chunks from repository
        similar_chunks = await doc_repository.get_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        if not similar_chunks:
            return {
                "response": "Không tìm thấy thông tin liên quan đến câu hỏi của bạn trong tài liệu.",
                "relevant_chunks": []
            }
        
        # Sort chunks by similarity score for better context building
        sorted_chunks = sorted(similar_chunks, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # Generate response using the chunks
        response = await query_service.generate_response(
            request.query,
            sorted_chunks,
            request.temperature
        )
        
        return {
            "response": response,
            "relevant_chunks": sorted_chunks,  # Include sorted chunks for transparency
            "total_chunks": len(sorted_chunks)
        }
    except Exception as e:
        return {
            "response": f"Có lỗi xảy ra khi xử lý câu hỏi: {str(e)}",
            "relevant_chunks": [],
            "error": str(e)
        }

@router.post("/vectorize", response_model=VectorResponse)
async def vectorize_text(
    request: VectorizeRequest,
    embedding_service: EmbeddingGenerator = Depends(get_embedding_generator)
) -> VectorResponse:
    """Convert text to vector form using embeddings"""
    embedding = embedding_service.generate_embedding(request.text)
    return VectorResponse(vector=embedding.tolist())

@router.post("/search", response_model=dict)
async def semantic_search(
    request: SearchRequest,
    query_service: QueryProcessor = Depends(get_query_processor),
    doc_repository: DocumentRepository = Depends(get_document_repository)
) -> dict:
    """Find relevant document chunks using semantic search with GRAG enhancement"""
    try:
        logger.info(f"Processing search request for query: {request.query}")
        
        # Get query embedding
        query_embedding = query_service.embedding_generator.generate_embedding(request.query)
        
        # Get initial chunks from repository
        initial_chunks = await doc_repository.get_similar_chunks(
            query_embedding=query_embedding.tolist(),
            top_k=request.top_k * 2,  # Get more chunks for GRAG to work with
            similarity_threshold=request.similarity_threshold
        )
        
        if not initial_chunks:
            logger.info("No relevant chunks found")
            return {"relevant_chunks": []}
            
        # Use QueryProcessor's get_relevant_chunks for GRAG enhancement
        logger.info("Applying GRAG reranking to initial chunks")
        reranked_chunks = query_service.get_relevant_chunks(
            query=request.query,
            doc_chunks=initial_chunks,
            top_k=request.top_k,
            use_grag=request.use_grag  # Use value from request
        )
        
        logger.info(f"Reranking complete, returning {len(reranked_chunks)} chunks")
        return {"relevant_chunks": reranked_chunks}
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise

@router.post("/build-context", response_model=dict)
async def build_context(
    request: ContextRequest,
    query_service: QueryProcessor = Depends(get_query_processor)
) -> dict:
    """Build context from relevant chunks for LLM"""
    response = await query_service.generate_response(
        request.query,
        request.relevant_chunks,
        request.temperature
    )
    return {"response": response}

@router.post("/summarize", response_model=dict)
async def summarize_document(
    request: SummarizeRequest,
    summarization_agent: RecursiveSummarizationAgent = Depends(get_summarization_agent),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Generate a recursive summary of a document"""
    try:
        # Create a workflow run for this request
        query_repo = SQLQueryRepository(db)
        workflow_repo = SQLWorkflowRepository(db)
        
        # Create a system query for summarization
        query_text = f"Summarize document {request.document_id}"
        query_id = await query_repo.create_user_query(
            query_text=query_text,
            is_system_query=True  # Mark as system query so it doesn't go into original_user_queries
        )
        
        # Create workflow run
        workflow_run_id = await workflow_repo.create_workflow_run(
            user_query_id=query_id,
            status="running"
        )
        
        # Process request with workflow context
        input_data = {
            "workflow_run_id": workflow_run_id,
            "document_id": int(request.document_id),  # Convert to int
            "language": request.language,
            "max_length": request.max_length
        }
        
        result = await summarization_agent.process(input_data)
        
        # Update workflow status
        await workflow_repo.update_workflow_status(
            workflow_run_id,
            "completed"
        )
        
        return {
            "document_id": input_data["document_id"],
            "chunk_summaries": result.get("chunk_summaries", []),
            "intermediate_summaries": result.get("intermediate_summaries", []),
            "final_summary": result.get("final_summary", ""),
            "metadata": result.get("metadata", {})
        }
    except Exception as e:
        logger.error(f"Error summarizing document: {str(e)}")
        if 'workflow_run_id' in locals():
            await workflow_repo.update_workflow_status(
                workflow_run_id,
                "failed"
            )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-query", response_model=dict)
async def analyze_query(
    request: QueryAnalysisRequest,
    query_analyzer: QueryAnalyzerAgent = Depends(get_query_analyzer_agent),
    embedding_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Analyze query using stepback prompting"""
    try:
        # Create a workflow run for this query
        query_repo = SQLQueryRepository(db)
        workflow_repo = SQLWorkflowRepository(db)
        
        # Create user query first
        query_id = await query_repo.create_user_query(request.query)

        # Get the user query to access its original_query_id
        stmt = select(UserQuery).where(UserQuery.id == query_id)
        result = await db.execute(stmt)
        user_query = result.scalar_one_or_none()
        if not user_query:
            raise AgentError("Failed to retrieve created user query")

        try:
            # Generate embedding asynchronously
            logger.info("Generating embedding for query: %s", request.query)
            query_embedding = await embedding_generator.generate_embedding(request.query)
            logger.info("Generated embedding shape: %s", query_embedding.shape if query_embedding is not None else None)
            
            # Update query with embedding
            if query_embedding is not None:
                logger.info("Updating query %d with embedding", query_id)
                await query_repo.update_query_embedding(query_id, query_embedding)
                logger.info("Successfully updated query embedding")
            else:
                logger.warning("No embedding generated for query")
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {str(e)}", exc_info=True)
        
        # Create workflow run
        workflow_run_id = await workflow_repo.create_workflow_run(
            user_query_id=query_id,
            status="running"
        )
        
        # Process query with workflow context
        input_data = {
            "workflow_run_id": workflow_run_id,
            "query_id": user_query.original_query_id,  # Use original_query_id instead of user_query.id
            "query_text": request.query,
            "language": request.language
        }
        
        analysis = await query_analyzer.process(input_data)
        
        # Update workflow status
        await workflow_repo.update_workflow_status(
            workflow_run_id,
            "completed"
        )
        
        return {"analysis": analysis}
    except Exception as e:
        logger.error(f"Error analyzing query: {str(e)}")
        if 'workflow_run_id' in locals():
            await workflow_repo.update_workflow_status(
                workflow_run_id,
                "failed"
            )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-citations", response_model=dict)
async def extract_citations(
    request: CitationRequest,
    citation_agent: CitationAgent = Depends(get_citation_agent),
    db: AsyncSession = Depends(get_db)
) -> dict:
    """Extract relevant citations from document"""
    try:
        # Create a workflow run for this request
        query_repo = SQLQueryRepository(db)
        workflow_repo = SQLWorkflowRepository(db)
        
        # Create user query
        query_id = await query_repo.create_user_query(request.query)
        
        # Create workflow run
        workflow_run_id = await workflow_repo.create_workflow_run(
            user_query_id=query_id,
            status="running"
        )
        
        # Process request with workflow context
        input_data = {
            "workflow_run_id": workflow_run_id,
            "query_id": query_id,
            "document_id": int(request.document_id),  # Convert to int
            "query_text": request.query,
            "language": request.language
        }
        
        result = await citation_agent.process(input_data)
        
        # Update workflow status
        await workflow_repo.update_workflow_status(
            workflow_run_id,
            "completed"
        )
        
        return {"citations": result.get("citations", [])}
    except Exception as e:
        logger.error(f"Error extracting citations: {str(e)}")
        if 'workflow_run_id' in locals():
            await workflow_repo.update_workflow_status(
                workflow_run_id,
                "failed"
            )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/build-query-context", response_model=ContextBuilderResponse)
async def build_query_context(
    request: BuildQueryContextRequest,
    context_builder: ContextBuilderAgent = Depends(get_context_builder_agent),
    db: AsyncSession = Depends(get_db)
) -> ContextBuilderResponse:
    """Build context from original query and its sub-queries.
    
    This endpoint:
    1. Gets the original query and all related sub-queries
    2. Creates a workflow run
    3. Processes each query to build context
    4. Returns combined context results
    """
    try:
        # Get repositories
        query_repo = SQLQueryRepository(db)
        workflow_repo = SQLWorkflowRepository(db)

        # Get original query
        result = await db.execute(
            select(OriginalUserQuery).where(
                OriginalUserQuery.id == request.original_query_id
            )
        )
        original_query = result.scalar_one_or_none()
        if not original_query:
            raise HTTPException(
                status_code=404,
                detail=f"Original query {request.original_query_id} not found"
            )

        # Create a system query for context building
        query_text = f"Build context for: {original_query.query_text}"
        user_query_id = await query_repo.create_user_query(
            query_text=query_text,
            is_system_query=True  # Mark as system query
        )

        # Get existing sub-queries
        result = await db.execute(
            select(SubQuery).where(
                SubQuery.original_query_id == request.original_query_id
            )
        )
        sub_queries = result.scalars().all()
        if not sub_queries:
            raise HTTPException(
                status_code=404,
                detail=f"No sub-queries found for original query {request.original_query_id}"
            )

        # Create workflow run with the system query
        workflow_run_id = await workflow_repo.create_workflow_run(
            user_query_id=user_query_id,
            status="running"
        )

        # Process each sub-query (including original)
        all_chunks = []
        last_context_set_id = None
        
        for sub_query in sub_queries:
            result = await context_builder.process({
                "workflow_run_id": workflow_run_id,
                "sub_query_id": sub_query.id,
                "query_text": sub_query.sub_query_text,
                "is_original": sub_query.id == sub_queries[0].id,
                "top_k": request.top_k
            })
            
            # Store context set ID from first result
            if last_context_set_id is None:
                last_context_set_id = result.get("context_set_id")
            
            # Add chunks to combined list
            if "context" in result and "chunks" in result["context"]:
                all_chunks.extend(result["context"]["chunks"])
        
        # Deduplicate chunks by ID
        seen_chunk_ids = set()
        unique_chunks = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["id"])
                unique_chunks.append(chunk)
        
        # Sort by document ID and chunk index
        unique_chunks.sort(key=lambda x: (x["document_id"], x["chunk_index"]))
        
        # Update workflow status
        await workflow_repo.update_workflow_status(
            workflow_run_id,
            "completed"
        )

        # Create final response
        return {
            "status": "success",
            "workflow_run_id": workflow_run_id,
            "context_set_id": last_context_set_id,
            "original_query": sub_queries[0].sub_query_text,
            "sub_queries": [{
                "id": sq.id,
                "text": sq.sub_query_text,
                "is_original": sq.id == sub_queries[0].id
            } for sq in sub_queries],
            "context": {
                "total_chunks": len(unique_chunks),
                "total_tokens": sum(len(chunk["text"].split()) for chunk in unique_chunks),
                "chunks": unique_chunks
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building query context: {str(e)}")
        if 'workflow_run_id' in locals():
            await workflow_repo.update_workflow_status(
                workflow_run_id,
                "failed"
            )
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize", response_model=dict)
async def synthesize_answer(
    request: SynthesisRequest,
    synthesizer: QuerySynthesizerAgent = Depends(get_query_synthesizer_agent)
) -> dict:
    """Synthesize final answer using analyzed query, context and citations"""
    try:
        answer = await synthesizer.synthesize(
            query=request.query,
            analyzed_query=request.analyzed_query,
            context=request.context,
            citations=request.citations,
            language=request.language,
            temperature=request.temperature
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error synthesizing answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
