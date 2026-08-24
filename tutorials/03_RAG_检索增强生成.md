# 第 3 课：RAG 检索增强生成

模型不知道你的私有资料。RAG（Retrieval-Augmented Generation）的思路是：**先检索相关资料，再把资料放进提示词，让模型基于资料回答**。

本课的目标：看懂 `run_rag` 的“加载 → 切分 → 向量化 → 检索 → 拼提示词 → 生成”链路，并亲手验证“答案只来自检索到的片段”。

## 先看代码

在 `src/langchain_lab/flow.py` 的 `run_rag` 中，链路是：

```python
documents = [Document(page_content=..., metadata={"source": ...}), ...]
splitter = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=15)
chunks = splitter.split_documents(documents)
vector_store = InMemoryVectorStore.from_documents(chunks, KeywordEmbeddings())
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

question = "哪里适合乘船看峡谷？"
rag_chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)
retrieved = retriever.invoke(question)
answer = rag_chain.invoke(question)
```

每个环节的角色：

| 环节 | 做了什么 | 对应代码 |
| --- | --- | --- |
| 加载 | 把资料包装成 `Document`（正文 + metadata） | `documents` |
| 切分 | 长文档切成小块，块太小检索会丢语义，块太大会超上下文 | `RecursiveCharacterTextSplitter` |
| 向量化 | 把文本变成向量，语义相近的文本向量更接近 | `KeywordEmbeddings` |
| 建索引 + 检索 | 存进向量库，按相似度找回最相关的块 | `InMemoryVectorStore` / `as_retriever` |
| 拼提示词 | 把检索结果格式化进 `Context` | `_format_docs` + `rag_prompt` |
| 生成 | 模型只看 `Context` 回答 | `model \| StrOutputParser()` |

## 观察数据流

运行：

```powershell
python run.py --provider offline --stage rag
```

输出 `切分块数: 3`、`检索来源: ['wushan.txt', 'yunyang.txt']` 和一段回答。然后在 `run_rag` 里加：

```python
print("原始文档数:", len(documents))
print("chunks 数量:", len(chunks))
for doc in retrieved:
    print("检索到:", doc.metadata["source"], "->", doc.page_content)
```

比较三件事：

- 原始 `documents` 有几篇，`chunks` 有几块 —— 感受“切分”如何改变数据粒度；
- `retrieved` 返回了哪些来源 —— 思考为什么是 `wushan.txt` 和 `yunyang.txt`，而不是 `fengjie.txt`；
- 最终 prompt 里的 `Context` 与答案的关系 —— 离线模型的回答是固定的教学文本，换成真实模型后，答案会明显“引用”Context 的内容。

> 为什么检索到的是这两篇？`KeywordEmbeddings` 用字符和二元组做哈希向量：查询“哪里适合乘船看峡谷？”包含“乘船”“峡谷”，与 `wushan.txt`（“乘船游览”“峡谷”）和 `yunyang.txt`（“峡谷”）共享信号；`fengjie.txt` 讲历史人文，与问题信号重叠最少，所以被排在第 3。

## 亲手改一改

### 1. 加入你自己的资料

在 `documents` 列表里追加一篇你熟悉的资料，例如：

```python
Document(
    page_content="万州大瀑布位于长江支流，是亚洲第一大瀑布，适合夏季避暑亲水。",
    metadata={"source": "wanzhou.txt"},
)
```

把 `question` 改成针对这篇资料的问题，打印 `retrieved`，确认它被检索回来。

### 2. 改检索数量

把 `search_kwargs={"k": 2}` 改成 `k=1`，观察 Context 变短后回答的变化。如果 Context 里没有相关信息，`rag_prompt` 的系统提示要求模型“说不知道”——真实模型会遵守，离线模型不会（它的回答是写死的）。

### 3. 给 metadata 加地区

按 README 的小挑战，给每篇 `Document` 增加 `metadata={"region": "奉节"}` 之类，再让 `_format_docs` 把地区也打出来，看答案如何“引用来源”。

## 小挑战

1. 把问题改成“哪里适合看历史文化？”，先预测最相关的来源，再打印 `retrieved` 验证。
2. 把 `chunk_size` 从 90 改成 30，观察 `chunks` 数量变多、单块信息变碎后，检索结果有什么变化。
3. 阅读 `KeywordEmbeddings`（`src/langchain_lab/models.py`），解释为什么中文查询和中文文档之间能共享信号（提示：字符二元组）。

## Q&A

这里会记录你在学习过程中提出的问题和答案。

## Quiz History

尚未测验。
