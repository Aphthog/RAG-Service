class RAGException(Exception):
    """Base exception for all RAG service errors."""
    pass


class IndexNotFoundError(RAGException):
    pass


class EmptyInputError(RAGException):
    pass


class EmbeddingError(RAGException):
    pass


class IndexBuildError(RAGException):
    pass


class TenantNotFoundError(RAGException):
    def __init__(self, tenant: str, available: list[str]):
        self.tenant = tenant
        self.available = available
        msg = f"Tenant '{tenant}' not found. Available tenants: {available}"
        super().__init__(msg)


class IndexLockedError(RAGException):
    def __init__(self, tenant: str):
        self.tenant = tenant
        msg = f"Index for tenant '{tenant}' is locked by another operation."
        super().__init__(msg)
