class ImagifyError(Exception):
    """Base exception."""

class IngestError(ImagifyError):
    """Image upload / ingestion failed."""

class EmbeddingError(ImagifyError):
    """CLIP embedding failed."""

class VectorStoreError(ImagifyError):
    """Chroma operation failed."""

class RetrievalError(ImagifyError):
    """Search query failed."""
