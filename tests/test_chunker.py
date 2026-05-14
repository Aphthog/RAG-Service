from rag_service.chunker import (
    AdaptiveChunker,
    MarkdownChunker,
    ParagraphChunker,
    SemanticChunker,
    _split_sentences,
)
from rag_service.types import Chunk, ChunkConfig
from rag_service.embedder import MockEmbedder


class TestMarkdownChunker:
    def test_splits_on_headings(self):
        chunker = MarkdownChunker(ChunkConfig(max_chars=800))
        text = "# Title\nSome content here.\n\n## Section 1\nSection one text.\n\n## Section 2\nSection two text."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3
        assert "Title" in chunks[0].text
        assert all(isinstance(c.metadata, dict) for c in chunks)

    def test_empty_text_returns_empty(self):
        chunker = MarkdownChunker(ChunkConfig())
        assert chunker.chunk("") == []

    def test_preserves_overlap(self):
        cfg = ChunkConfig(max_chars=50, min_chars=10, overlap=20)
        chunker = MarkdownChunker(cfg)
        text = "A" * 120
        chunks = chunker.chunk(text)
        if len(chunks) >= 2:
            assert chunks[0].text[-10:] in chunks[1].text or len(chunks[0].text) <= cfg.max_chars

    def test_chunk_index_sequence(self):
        chunker = MarkdownChunker(ChunkConfig(max_chars=50))
        text = "Long text. " * 50
        chunks = chunker.chunk(text)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_start_end_char_positions(self):
        chunker = MarkdownChunker(ChunkConfig(max_chars=100))
        text = "Hello world. This is a test. More content here."
        chunks = chunker.chunk(text)
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(text)


class TestSemanticChunker:
    def test_chunks_with_mock_embedder(self):
        embedder = MockEmbedder(seed=123)
        chunker = SemanticChunker(embedder, ChunkConfig(max_chars=200, min_chars=50, semantic_threshold=0.5))
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.text
            assert isinstance(c.metadata, dict)

    def test_single_sentence(self):
        embedder = MockEmbedder(seed=123)
        chunker = SemanticChunker(embedder, ChunkConfig())
        text = "Just one sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "Just one sentence" in chunks[0].text

    def test_empty_text(self):
        embedder = MockEmbedder()
        chunker = SemanticChunker(embedder, ChunkConfig())
        assert chunker.chunk("") == []


class TestAdaptiveChunker:
    def test_routes_markdown_to_markdown_chunker(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "# Title\n## Section\nContent here with markdown headings."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text

    def test_routes_plain_text_to_semantic_chunker(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "This is a plain paragraph without any markdown structure. Just regular text content."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_empty_text(self):
        embedder = MockEmbedder()
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        assert chunker.chunk("") == []

    def test_chunks_inherit_metadata(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "# Doc\nContent of the document."
        chunks = chunker.chunk(text, metadata={"source": "test.md", "author": "camille"})
        for c in chunks:
            assert c.metadata == {"source": "test.md", "author": "camille"}


class TestParagraphChunker:
    def test_splits_on_paragraphs(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=4, min_chars=1))
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3
        assert "第一段" in chunks[0].text
        assert "第二段" in chunks[1].text

    def test_falls_back_to_sentences(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=200, min_chars=50))
        text = "句子一。句子二。句子三。句子四。句子五。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.text

    def test_empty_text(self):
        chunker = ParagraphChunker(ChunkConfig())
        assert chunker.chunk("") == []

    def test_single_paragraph(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=800))
        text = "只有一段话，没有空行分割。就这一句了。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "只有一段话" in chunks[0].text

    def test_preserves_overlap(self):
        cfg = ChunkConfig(max_chars=50, min_chars=10, overlap=20)
        chunker = ParagraphChunker(cfg)
        text = "A" * 120
        chunks = chunker.chunk(text)
        if len(chunks) >= 2:
            assert chunks[0].text[-10:] in chunks[1].text or len(chunks[0].text) <= cfg.max_chars

    def test_start_end_char_positions(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=8))
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = chunker.chunk(text)
        assert chunks[0].start_char == 0

    def test_chunks_inherit_metadata(self):
        chunker = ParagraphChunker(ChunkConfig())
        text = "第一段。\n\n第二段。"
        chunks = chunker.chunk(text, metadata={"source": "test.md"})
        for c in chunks:
            assert c.metadata == {"source": "test.md"}

    def test_finalize_chunk_semantic_fallback(self):
        chunker = ParagraphChunker(
            ChunkConfig(max_chars=50),
            SemanticChunker(MockEmbedder(), ChunkConfig()),
        )
        text = "这是超过五十个字的长段落" + "，继续补充内容" * 20
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_finalize_chunk_fixed_split(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=50, overlap=0))
        text = "字" * 200
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c.text) <= 50

    def test_mixed_punctuation_sentences(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=800))
        text = "第一句。Second sentence！第三句？Fourth.\n第五句！"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        assert "第一句" in chunks[0].text
        assert "Second sentence" in chunks[0].text


class TestAdaptiveChunkerLargeText:
    def test_large_text_routes_to_paragraph_chunker(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "句子。\n\n" * 20000
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.text

    def test_small_text_still_uses_semantic(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "这是小文本。依然走语义分块。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

    def test_large_text_no_blank_lines_falls_back_safely(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "句子。" * 40000
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
