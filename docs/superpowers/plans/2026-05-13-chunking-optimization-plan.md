# Chunking 策略优化 实施计划

> **For agentic workers:** REQUIRED SUB-SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**目标:** 大文本（>5万字）不走语义分块，改走段落分块，避免 bge-m3 模型推理，将建索引时间从数分钟降到 ~1 分钟

**方案:** 新增 `ParagraphChunker`（按空行/句子拆分，不跑模型），`AdaptiveChunker` 路由增加大文本判断

**涉及文件:**
- 修改: `rag_service/chunker.py`
- 测试: `tests/test_chunker.py`

**评审调整:**
- 提取 `_split_sentences` + `_finalize_chunk` 为模块级函数消除重复
- `_finalize_chunk` 改为一次调用（消除重复模型编码）
- 安全网：语义降级时如果句子数 >2000，改为固定大小切分

---

### 任务 1: 重构 chunker.py（提取公共函数）

**文件:** `rag_service/chunker.py`（文件开头处）

先把 SemanticChunker 和 ParagraphChunker 共用的逻辑提取为模块级函数，消除重复。

> 注意: 这些重构不影响现有测试，Step 3 会验证。

- [ ] **Step 1: 提取 `_split_sentences` 为模块级函数**

```python
def _split_sentences(text: str) -> list[str]:
    pattern = r'(?<=[。！？.!?\n])\s*'
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]
```

放到文件顶部 `import` 之后、第一个类定义之前。然后在 `SemanticChunker._split_sentences` 中调用它：

```python
# SemanticChunker 中
def _split_sentences(self, text: str) -> list[str]:
    return _split_sentences(text)
```

- [ ] **Step 2: 提取 `_finalize_chunk` 为模块级函数**

```python
def _finalize_chunk(
    text: str, start: int, metadata: dict, base_idx: int,
    config: ChunkConfig, semantic: SemanticChunker | None = None,
) -> list[Chunk]:
    if len(text) <= config.max_chars:
        return [Chunk(text=text, metadata=metadata, chunk_index=base_idx,
                       start_char=start, end_char=start + len(text))]
    if semantic:
        # 安全网：检查语义降级的文本是否仍然过大
        sentences = _split_sentences(text)
        if len(sentences) > 2000:
            # 句子过多时走固定大小切分，避免重复 5 分钟问题
            return _fixed_size_split(text, start, metadata, base_idx, config)
        sub_chunks = semantic.chunk(text, metadata)
        return [Chunk(
            text=c.text, metadata=c.metadata,
            chunk_index=base_idx + c.chunk_index,
            start_char=start + c.start_char,
            end_char=start + c.end_char,
        ) for c in sub_chunks]
    return _fixed_size_split(text, start, metadata, base_idx, config)


def _fixed_size_split(
    text: str, start: int, metadata: dict, base_idx: int, config: ChunkConfig,
) -> list[Chunk]:
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
```

- [ ] **Step 3: 更新 MarkdownChunker 使用公共函数**

将 `MarkdownChunker._finalize_chunk` 替换为：

```python
def _finalize_chunk(self, text: str, start: int, metadata: dict, base_idx: int) -> list[Chunk]:
    return _finalize_chunk(text, start, metadata, base_idx, self._cfg, self._semantic)
```

同时修复 MarkdownChunker.chunk() 中的重复调用问题：

```python
# 旧代码（两次调用）:
chunks.extend(self._finalize_chunk(current, current_start, meta, chunk_idx))
chunk_idx += len(self._finalize_chunk(current, current_start, meta, chunk_idx))

# 改为（一次调用）:
result = self._finalize_chunk(current, current_start, meta, chunk_idx)
chunks.extend(result)
chunk_idx += len(result)
```

- [ ] **Step 4: 运行现有测试确认无回归**

```bash
cd /c/Users/Camille/Desktop/rag-service && python -m pytest tests/test_chunker.py -v
```

预期: 全部 PASS

---

### 任务 2: 新增 ParagraphChunker

**文件:** `rag_service/chunker.py`（在 SemanticChunker 之后、MarkdownChunker 之前插入）

- [ ] **Step 1: 添加 ParagraphChunker 类**

```python
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
        # 如果没有连续空行，降级到按句子切分（正则，不跑模型）
        if len(paragraphs) <= 1:
            sentences = _split_sentences(text)
            if not sentences:
                return []
            paragraphs = sentences
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
        # 加 overlap
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
```

- [ ] **Step 2: 修改 AdaptiveChunker 的路由**

```python
class AdaptiveChunker:
    _MARKDOWN_PATTERN = re.compile(r'^#{1,6}\s|\n#{1,6}\s|```|^\|.*\|$', re.MULTILINE)
    _LARGE_TEXT_THRESHOLD = 50_000  # 超过此字符数不走语义分块

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
            return self._markdown.chunk(text, metadata)
        if len(text) > self._LARGE_TEXT_THRESHOLD:
            return self._paragraph.chunk(text, metadata)
        return self._semantic.chunk(text, metadata)
```

- [ ] **Step 3: 验证现有测试仍然通过**

```bash
cd /c/Users/Camille/Desktop/rag-service && python -m pytest tests/test_chunker.py -v
```

预期: 全部 PASS

---

### 任务 3: 新增测试

**文件:** `tests/test_chunker.py`

- [ ] **Step 1: 更新 import**

```python
from rag_service.chunker import AdaptiveChunker, MarkdownChunker, SemanticChunker, ParagraphChunker
```

- [ ] **Step 2: 添加 ParagraphChunker 测试**

```python
class TestParagraphChunker:
    def test_splits_on_paragraphs(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=800))
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3
        assert "第一段内容" in chunks[0].text
        assert "第二段内容" in chunks[1].text

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
        chunker = ParagraphChunker(ChunkConfig(max_chars=100))
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = chunker.chunk(text)
        assert chunks[0].start_char == 0
        assert chunks[-1].end_char == len(text)

    def test_chunks_inherit_metadata(self):
        chunker = ParagraphChunker(ChunkConfig())
        text = "第一段。\n\n第二段。"
        chunks = chunker.chunk(text, metadata={"source": "test.md"})
        for c in chunks:
            assert c.metadata == {"source": "test.md"}

    def test_finalize_chunk_semantic_fallback(self):
        chunker = ParagraphChunker(ChunkConfig(max_chars=50), SemanticChunker(MockEmbedder(), ChunkConfig()))
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
        assert len(chunks) >= 5
```

- [ ] **Step 3: 添加 AdaptiveChunker 路由测试**

```python
class TestAdaptiveChunkerLargeText:
    def test_large_text_routes_to_paragraph_chunker(self):
        embedder = MockEmbedder(seed=123)
        chunker = AdaptiveChunker(embedder, ChunkConfig())
        text = "句子。\n\n" * 20000  # 超过 5 万字
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
        text = "句子。" * 40000  # 超过 5 万字，无空行
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1  # 不走 SemanticChunker，走固定大小切分
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /c/Users/Camille/Desktop/rag-service && python -m pytest tests/test_chunker.py -v
```

预期: 14 个测试全部 PASS

- [ ] **Step 5: 运行全部测试确认无回归**

```bash
cd /c/Users/Camille/Desktop/rag-service && python -m pytest tests/ -v
```

预期: 全部 PASS

---

### 任务 4: 端到端验证（可选）

- [ ] **Step 1: 在 server.py 环境下验证**

```bash
cd /c/Users/Camille/Desktop/rag-service && python server.py
curl -X POST http://localhost:8000/tenants/test/index \
  -H "Content-Type: application/json" \
  -d '{"texts": ["'$(python -c "print('测试句子。' * 30000)")'"]}'
```

预期: 在 1-2 分钟内返回

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | CLEAR | 6 issues, 0 critical gaps |

- **UNRESOLVED:** 0
- **VERDICT:** ENG CLEARED — ready to implement
