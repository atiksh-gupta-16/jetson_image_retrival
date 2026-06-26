"""
Application-wide custom exceptions.

Each service raises its own exception type so callers can handle
failures at the right granularity.
"""


class ImagifyError(Exception):
    """Base exception for the Imagify application."""


class IngestError(ImagifyError):
    """Raised when image validation, upload, or ingestion fails."""


class EmbeddingError(ImagifyError):
    """Raised when CLIP embedding generation fails."""


class VectorStoreError(ImagifyError):
    """Raised when a ChromaDB operation fails."""


class RetrievalError(ImagifyError):
    """Raised when semantic retrieval fails."""


class IntentExtractionError(ImagifyError):
    """Raised when the intent extractor cannot extract a valid intent."""


class QueryPipelineError(ImagifyError):
    """Raised when the complete query pipeline fails."""