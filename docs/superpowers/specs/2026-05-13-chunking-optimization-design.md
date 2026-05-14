# Chunking 策略优化设计

## 问题

当前 `AdaptiveChunker` 对纯文本（非 Markdown）走 `SemanticChunker`，需要对所有句子过一遍 bge-m3 计算相邻相似度来找到分块边界。当单条文本 >5 万字时，这一步耗时数分钟甚至不可用。

## 方案

### 路由策略调整

```
AdaptiveChunker.chunk():
  ├─ Markdown 内容 → MarkdownChunker（不变）
  ├─ 纯文本，>5万字 → ParagraphChunker（新增）
  └─ 纯文本，≤5万字 → SemanticChunker（不变）
```

### ParagraphChunker（新增）

按文档结构分块，**不依赖模型推理**：

1. 优先按 `\n\n`（空行）拆分段落
2. 如果没有空行，降级为按句子正则切分（复用 `SemanticChunker._split_sentences`）
3. 段落依次合并，不超过 `max_chars`
4. 合并好的块之间按 `overlap` 切重叠
5. 单个段落/句子片段超过 `max_chars` → 退回到 `SemanticChunker`（此时段落很小，编码成本低）

### SemanticChunker 的角色变化

- 从默认策略降级为"小文本默认 + 大文本兜底"
- 代码不动，只改 `AdaptiveChunker` 的路由
- 5 万字阈值来自实测：CPU 上 bge-m3 编码 2000 句约 45s，可接受的上限

## 预期效果

| 场景 | 改前 | 改后 |
|------|------|------|
| 95 万字 JSON 建索引 | 5min+（卡在语义分块） | ~1min（正则切分 + 编码建索引） |
| 1 万字普通文档 | ~10s（语义分块） | ~10s（不变，仍走语义分块） |
| Markdown 文档 | 不变 | 不变 |
