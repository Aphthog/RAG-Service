import re
import logging
import numpy as np
from rag_service.types import Chunk, ChunkConfig

logger = logging.getLogger("rag_service.chunker")


def _split_sentences(text: str) -> list[str]:
    pattern = r'(?<=[。！？.!?\n])\s*'
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def _fixed_size_split(
    text: str, start: int, metadata: dict, base_idx: int, config: "ChunkConfig",
) -> list["Chunk"]:
    result = []
    step = config.max_chars - config.overlap
    for i in range(0, len(text), step):
        end = min(i + config.max_chars, len(text))
        result.append(Chunk(
            text=text[i:end], metadata=metadata,
            chunk_index=base_idx + len(result),
            start_char=start + i, end_char=start + end,
        ))
    return result


def _finalize_chunk(
    text: str, start: int, metadata: dict, base_idx: int,
    config: "ChunkConfig", semantic: "SemanticChunker | None" = None,
) -> list["Chunk"]:
    if len(text) <= config.max_chars:
        return [Chunk(text=text, metadata=metadata, chunk_index=base_idx,
                       start_char=start, end_char=start + len(text))]
    if semantic:
        sentences = _split_sentences(text)
        if len(sentences) > 2000:
            return _fixed_size_split(text, start, metadata, base_idx, config)
        sub_chunks = semantic.chunk(text, metadata)
        return [Chunk(
            text=c.text, metadata=c.metadata,
            chunk_index=base_idx + c.chunk_index,
            start_char=start + c.start_char,
            end_char=start + c.end_char,
        ) for c in sub_chunks]
    return _fixed_size_split(text, start, metadata, base_idx, config)


class SemanticChunker:
    """Chunk by embedding similarity — split at semantic boundary drops."""

    def __init__(self, embedder, config: ChunkConfig):
        self._embedder = embedder
        self._cfg = config

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        meta = metadata or {}
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text.strip(), metadata=meta, chunk_index=0, start_char=0, end_char=len(text))]

        logger.info("SemanticChunker: encoding %d sentences (%.1fK chars, threshold=%.2f)...",
                     len(sentences), len(text) / 1000, self._cfg.semantic_threshold)
        embeddings = self._embedder.encode(sentences, batch_size=self._cfg.batch_size)
        vecs = embeddings.dense

        # Compute cosine similarity between adjacent sentences
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vecs / norms
        similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)

        # Find breakpoints where similarity < threshold
        breakpoints = [0]
        for i, sim in enumerate(similarities):
            if sim < self._cfg.semantic_threshold:
                breakpoints.append(i + 1)
        breakpoints.append(len(sentences))

        # Merge sentences between breakpoints into chunks within [min_chars, max_chars]
        chunks = []
        chunk_idx = 0
        pos = 0

        for bp_start, bp_end in zip(breakpoints[:-1], breakpoints[1:]):
            segment = " ".join(sentences[bp_start:bp_end])
            if chunk_idx == 0:
                current_start = 0
            else:
                current_start = pos

            # Add segment to current chunk or start new one
            if chunk_idx == 0 or not chunks:
                current = segment
            else:
                # Check if we can append to the growing chunk
                pass  # handled below

        # Rebuild: merge sentence segments into size-bounded chunks
        chunks = []
        chunk_idx = 0
        current = ""
        current_start = 0
        current_len = 0

        for bp_start, bp_end in zip(breakpoints[:-1], breakpoints[1:]):
            segment = " ".join(sentences[bp_start:bp_end])
            seg_start = text.find(segment, current_start + current_len if chunk_idx > 0 else 0)
            if seg_start < 0:
                seg_start = current_start + current_len if chunk_idx > 0 else 0

            if not current:
                current = segment
                current_start = seg_start
                current_len = len(segment)
            elif len(current) + 1 + len(segment) <= self._cfg.max_chars:
                current += " " + segment
                current_len = len(current)
            else:
                if len(current) >= self._cfg.min_chars or not segment:
                    end_pos = current_start + len(current)
                    chunks.append(Chunk(
                        text=current.strip(),
                        metadata=meta,
                        chunk_index=chunk_idx,
                        start_char=current_start,
                        end_char=min(end_pos, len(text)),
                    ))
                    chunk_idx += 1
                    current = segment
                    current_start = seg_start
                    current_len = len(segment)

        if current.strip():
            end_pos = current_start + len(current)
            chunks.append(Chunk(
                text=current.strip(),
                metadata=meta,
                chunk_index=chunk_idx,
                start_char=current_start,
                end_char=min(end_pos, len(text)),
            ))

        # Apply overlap
        if self._cfg.overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, text)

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        return _split_sentences(text)

    def _apply_overlap(self, chunks: list[Chunk], original_text: str) -> list[Chunk]:
        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].end_char
            overlap_start = max(chunks[i].start_char - self._cfg.overlap, chunks[i - 1].start_char)
            if overlap_start < prev_end:
                chunks[i] = Chunk(
                    text=original_text[overlap_start:chunks[i].end_char],
                    metadata=chunks[i].metadata,
                    chunk_index=chunks[i].chunk_index,
                    start_char=overlap_start,
                    end_char=chunks[i].end_char,
                )
        return chunks


class ParagraphChunker:
    """按段落切分，不依赖模型推理。大文本时替代 SemanticChunker。"""

    def __init__(self, config: ChunkConfig, semantic_chunker: SemanticChunker | None = None):
        self._cfg = config
        self._semantic = semantic_chunker

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        meta = metadata or {}
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) <= 1:
            sentences = _split_sentences(text)
            if not sentences:
                return []
            logger.info("ParagraphChunker: no blank lines, splitting %d sentences (%.1fK chars)...",
                        len(sentences), len(text) / 1000)
            paragraphs = sentences
        else:
            logger.info("ParagraphChunker: splitting %d paragraphs (%.1fK chars)...",
                        len(paragraphs) if paragraphs != [text] else 1, len(text) / 1000)
        chunks = []
        current = ""
        current_start = 0
        chunk_idx = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if not current:
                current = para
                current_start = text.find(para)
            elif len(current) + 1 + len(para) <= self._cfg.max_chars:
                current += " " + para
            else:
                for c in _finalize_chunk(current, current_start, meta, chunk_idx,
                                          self._cfg, self._semantic):
                    chunks.append(c)
                    chunk_idx += 1
                current = para
                current_start = text.find(para, current_start + len(current))
        if current.strip():
            for c in _finalize_chunk(current, current_start or 0, meta, chunk_idx,
                                      self._cfg, self._semantic):
                chunks.append(c)
                chunk_idx += 1
        if self._cfg.overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].end_char
                overlap_start = max(chunks[i].start_char - self._cfg.overlap, chunks[i - 1].start_char)
                if overlap_start < prev_end:
                    chunks[i] = Chunk(
                        text=text[overlap_start:chunks[i].end_char],
                        metadata=chunks[i].metadata,
                        chunk_index=chunks[i].chunk_index,
                        start_char=overlap_start,
                        end_char=chunks[i].end_char,
                    )
        return chunks


class MarkdownChunker:
    """Structurally-aware chunker for Markdown documents."""

    def __init__(self, config: ChunkConfig, semantic_chunker: SemanticChunker | None = None):
        self._cfg = config
        self._semantic = semantic_chunker

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        meta = metadata or {}

        sections = re.split(r'(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        chunks = []
        chunk_idx = 0

        for section in sections:
            if not section.strip():
                continue
            paragraphs = re.split(r'\n\s*\n', section)
            current = ""
            current_start = None

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if not current:
                    current = para
                    current_start = text.find(para)
                elif len(current) + 2 + len(para) <= self._cfg.max_chars:
                    current += "\n\n" + para
                else:
                    result = self._finalize_chunk(current, current_start, meta, chunk_idx)
                    chunks.extend(result)
                    chunk_idx += len(result)
                    current = para
                    current_start = text.find(para, current_start + len(current))

            if current.strip():
                result = self._finalize_chunk(current, current_start or 0, meta, chunk_idx)
                chunks.extend(result)
                chunk_idx += len(result)

        # Renumber chunk indices
        for i, c in enumerate(chunks):
            c.chunk_index = i
        chunk_idx = len(chunks)

        # Apply overlap
        if self._cfg.overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev_end = chunks[i - 1].end_char
                overlap_start = max(chunks[i].start_char - self._cfg.overlap, chunks[i - 1].start_char)
                if overlap_start < prev_end:
                    chunks[i] = Chunk(
                        text=text[overlap_start:chunks[i].end_char],
                        metadata=chunks[i].metadata,
                        chunk_index=chunks[i].chunk_index,
                        start_char=overlap_start,
                        end_char=chunks[i].end_char,
                    )

        return chunks

    def _finalize_chunk(self, text: str, start: int, metadata: dict, base_idx: int) -> list[Chunk]:
        return _finalize_chunk(text, start, metadata, base_idx, self._cfg, self._semantic)


class AdaptiveChunker:
    """Routes input to MarkdownChunker or SemanticChunker based on content detection."""

    _MARKDOWN_PATTERN = re.compile(r'^#{1,6}\s|\n#{1,6}\s|```|^\|.*\|$', re.MULTILINE)
    _LARGE_TEXT_THRESHOLD = 50_000

    def __init__(self, embedder, config: ChunkConfig):
        self._embedder = embedder
        self._cfg = config
        self._semantic = SemanticChunker(embedder, config)
        self._markdown = MarkdownChunker(config, self._semantic)
        self._paragraph = ParagraphChunker(config, self._semantic)

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        if not text:
            return []
        if self._MARKDOWN_PATTERN.search(text):
            logger.debug("AdaptiveChunker: routing to MarkdownChunker")
            return self._markdown.chunk(text, metadata)
        if len(text) > self._LARGE_TEXT_THRESHOLD:
            logger.info("AdaptiveChunker: text=%.1fK chars > %d threshold, routing to ParagraphChunker",
                        len(text) / 1000, self._LARGE_TEXT_THRESHOLD)
            return self._paragraph.chunk(text, metadata)
        logger.debug("AdaptiveChunker: routing to SemanticChunker")
        return self._semantic.chunk(text, metadata)
