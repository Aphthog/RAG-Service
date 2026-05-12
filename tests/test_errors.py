from rag_service.errors import (
    RAGException,
    IndexNotFoundError,
    EmptyInputError,
    EmbeddingError,
    IndexBuildError,
    TenantNotFoundError,
    IndexLockedError,
)


def test_rag_exception_base():
    err = RAGException("base error")
    assert isinstance(err, Exception)
    assert str(err) == "base error"


def test_index_not_found_error():
    err = IndexNotFoundError("index path not found")
    assert isinstance(err, RAGException)
    assert "index path not found" in str(err)


def test_empty_input_error():
    err = EmptyInputError("texts cannot be empty")
    assert isinstance(err, RAGException)


def test_embedding_error():
    err = EmbeddingError("model failed to load")
    assert isinstance(err, RAGException)


def test_index_build_error():
    err = IndexBuildError("FAISS build failed")
    assert isinstance(err, RAGException)


def test_tenant_not_found_error():
    err = TenantNotFoundError("my_kb", ["kb1", "kb2"])
    assert isinstance(err, RAGException)
    assert "my_kb" in str(err)
    assert "kb1" in str(err)
    assert "kb2" in str(err)


def test_index_locked_error():
    err = IndexLockedError("my_kb")
    assert isinstance(err, RAGException)
    assert "my_kb" in str(err)
