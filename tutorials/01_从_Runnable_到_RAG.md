# 第 1 课：从 Runnable 走完一条 LangChain 链路

本课的目标不是背 API，而是建立一个可调试的 mental model：LangChain 的大多数对象都可以看成“接收一种输入、返回一种输出”的 Runnable。你把它们用 `|` 串起来，就得到一条可组合的流水线。

## 先看最小形状

在 `src/langchain_lab/flow.py` 的 `run_basic_chain` 中：

```python
prompt = ChatPromptTemplate.from_messages(...)
chain = prompt | model | StrOutputParser()
answer = chain.invoke({"concept": "Runnable"})
```

数据会依次经过：

```text
{"concept": "Runnable"}
        ↓
ChatPromptValue / messages
        ↓
AIMessage
        ↓
str
```

这就是 LCEL 的价值：每一步都能单独替换和测试。真实项目里，你可以把 `model` 换成 OpenAI、Anthropic 或本地模型，而不用重写 prompt 和 parser。

## 为什么 RAG 要多几步？

模型本身不知道你的私有资料。`run_rag` 把资料变成可检索的上下文：

1. `Document` 保存正文和 metadata。
2. `RecursiveCharacterTextSplitter` 把长文档切成较小的 chunks。
3. `KeywordEmbeddings` 在本地生成可重复的向量（真实项目通常换成 embedding API 或本地 embedding 模型）。
4. `InMemoryVectorStore` 根据相似度找回 chunks。
5. `rag_prompt` 把检索结果放进 `Context`。
6. 模型只根据这个上下文生成答案。

可以把它想成“先查资料，再回答问题”，而不是让模型凭记忆猜。

会话记忆在本项目中使用 LangGraph 的 `InMemorySaver` 和 `thread_id`。这比把历史放在一个全局列表里更接近服务端应用：状态由图的 checkpointer 管理，不同 thread 相互隔离。

## 你应该观察什么

运行：

```powershell
python run.py --provider offline --stage rag
```

然后在 `run_rag` 中加入 `print(doc.page_content)`，比较：

- 原始 `documents` 有几篇；
- `chunks` 有几块；
- `retrieved` 返回了哪些来源；
- 最终 prompt 中的 `Context` 与答案之间有什么关系。

## 小挑战

1. 把问题改成“哪里适合看历史文化？”并预测最相关的来源。
2. 把 `search_kwargs={"k": 2}` 改成 `k=1`，观察上下文变短后会发生什么。
3. 给每个 `Document` 增加 `metadata={"region": ...}`，让答案打印来源和地区。

## Q&A

这里会记录你在学习过程中提出的问题和答案。

## Quiz History

尚未测验。
