"""Common utilities for API routes.

This package contains common utilities for:

- response_builder: Standardized API response formatting
  * Success responses
  * Error responses
  * Workflow responses
  * Specialized responses (chunks, queries, citations)

These utilities ensure consistent response formatting across
all API endpoints and provide type-safe response building.
"""

from .response_builder import ResponseBuilder

__all__ = [
    "ResponseBuilder"
]