"""Common response builder utilities for API routes."""
from typing import Any, Dict, List, Optional
from datetime import datetime

class ResponseBuilder:
    """Builder for consistent API responses."""

    @staticmethod
    def success(
        data: Any,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a success response."""
        response = {
            "status": "success",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        if message:
            response["message"] = message
        if metadata:
            response["metadata"] = metadata
        return response

    @staticmethod
    def error(
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build an error response."""
        response = {
            "status": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        if error_code:
            response["error_code"] = error_code
        if details:
            response["details"] = details
        return response

    @staticmethod
    def workflow_response(
        workflow_run_id: int,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a workflow response."""
        response = {
            "status": "success",
            "workflow_run_id": workflow_run_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            response["metadata"] = metadata
        return response

    @staticmethod
    def chunk_response(
        chunks: List[Dict[str, Any]],
        total_chunks: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a response containing document chunks."""
        return {
            "status": "success",
            "chunks": chunks,
            "metadata": {
                "total_chunks": total_chunks,
                "total_tokens": sum(len(chunk["text"].split()) for chunk in chunks),
                **(metadata or {})
            },
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def query_response(
        response: str,
        relevant_chunks: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a response for query endpoints."""
        return {
            "status": "success",
            "response": response,
            "relevant_chunks": relevant_chunks,
            "metadata": {
                "total_chunks": len(relevant_chunks),
                "total_tokens": sum(len(chunk["text"].split()) for chunk in relevant_chunks),
                **(metadata or {})
            },
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def citation_response(
        citations: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a response containing citations."""
        return {
            "status": "success",
            "citations": citations,
            "metadata": {
                "total_citations": len(citations),
                **(metadata or {})
            },
            "timestamp": datetime.now().isoformat()
        }