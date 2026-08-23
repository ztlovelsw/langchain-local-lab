# LangChain 本地学习实验室

这是一个从零开始的、可离线运行的 LangChain 教学项目。它把一条真实应用常见的链路拆成六个小阶段：

```text
输入 → Prompt → Chat Model → Output Parser
                    ↓
文档 → 切分 → Embeddings → VectorStore → Retriever → RAG Prompt
                    ↓
             History / Tools / LangGraph
```

默认使用项目内置的 `OfflineChatModel` 和 `KeywordEmbeddings`。它们不是生产模型，而是确定性的教学替身：不需要 API key、不访问网络，也能让你观察 LangChain 的接口和数据流。学会结构后，再把模型切换为 OpenAI-compatible 服务。

## 1. 环境准备

建议使用 Python 3.13。当前机器的默认 Python 3.14 会触发 LangChain 依赖的 Pydantic V1 兼容性警告；本项目的 `pyproject.toml` 已限制在 `>=3.12,<3.14`。

在 PowerShell 中执行：

```powershell
cd D:\Desktop\langchain-learn
uv venv --python "C:\Users\20151\miniconda3\python.exe" .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

如果 PowerShell 禁止激活脚本，可以直接调用虚拟环境解释器：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

依赖已经安装到项目的 `.venv`。之后优先使用 `.\.venv\Scripts\python.exe`，避免混用系统 Python。

## 2. 第一次运行：不需要密钥

```powershell
python run.py --provider offline
```

也可以一次只看一个阶段：

```powershell
python run.py --provider offline --stage basic
python run.py --provider offline --stage structured
python run.py --provider offline --stage rag
python run.py --provider offline --stage memory
python run.py --provider offline --stage tools
python run.py --provider offline --stage graph
```

你会看到：

1. `basic`：`ChatPromptTemplate | ChatModel | StrOutputParser`，这是 LCEL（LangChain Expression Language）的基本形状。
2. `structured`：用 Pydantic schema 约束模型输出，结果可以直接当 Python 对象使用。
3. `rag`：文档切分、向量化、相似度检索、把上下文塞进提示词，再生成答案。
4. `memory`：同一个 `thread_id` 的两次调用通过 LangGraph checkpointer 共享消息历史。
5. `tools`：用 `@tool` 声明工具，并把工具接入 Runnable 流。
6. `graph`：用 LangGraph 把一个步骤显式建模成状态图节点。

## 3. 切换到真实模型

复制环境模板并填写密钥：

```powershell
Copy-Item .env.example .env
notepad .env
```

至少设置：

```dotenv
MODEL_PROVIDER=openai
OPENAI_API_KEY=你的密钥
OPENAI_MODEL=gpt-4o-mini
```

如果使用兼容 OpenAI API 的网关，再设置：

```dotenv
OPENAI_BASE_URL=https://你的网关/v1
```

然后运行：

```powershell
python run.py --provider openai --stage basic
```

确认基础调用成功后，再依次运行 `structured`、`rag`、`memory` 和 `graph`。不要把 `.env` 提交到 Git；它已经被 `.gitignore` 忽略。

### 完全本地模型

当前机器没有安装 Ollama、Docker 或其他本地模型服务，所以项目不会自动下载模型。以后你明确选择 Ollama 后，可以在 `models.py` 增加 `ChatOllama` 分支；先把 Ollama 服务和模型（例如 `llama3.2:3b`）准备好，再进行接入。CPU-only 环境运行本地模型会比较慢。

## 4. 建议的学习顺序

按下面的顺序读代码和改代码，每次只改一个变量：

### 第 1 课：Runnable 和 LCEL

阅读 `run_basic_chain`。把 system prompt 改成你自己的话，观察 `prompt.invoke(...)`、`model.invoke(...)` 和完整 `chain.invoke(...)` 的输入/输出类型差异。

### 第 2 课：结构化输出

给 `TravelPlan` 增加 `budget: int` 字段，看看离线模型和真实模型分别会怎样处理 schema。

### 第 3 课：RAG

在 `run_rag` 的 `documents` 中加入一段你熟悉的资料；改变 `question`，打印 `retrieved`，确认答案只使用检索到的片段。

### 第 4 课：历史与状态

把 `session_id` 改成两个不同值，验证不同会话互不共享；然后阅读 `run_langgraph` 中的 `GraphState`。

### 第 5 课：工具和 Agent

先理解 `@tool` 的输入 schema，再换成真实模型的 tool-calling agent。Agent 的核心不是“魔法”，而是：模型提出工具调用 → 程序执行工具 → 把结果回传模型。

## 5. 验证

安装开发依赖后运行：

```powershell
python -m pytest -q
```

测试覆盖基础 LCEL、结构化输出、RAG 检索、会话历史、工具和 LangGraph。当前 LangGraph 依赖自身可能打印一条 `LangChainPendingDeprecationWarning`，不影响运行；正式学习建议使用上面的 3.13 虚拟环境。

## 6. 文件导航

- `run.py`：最简单的入口。
- `src/langchain_lab/models.py`：离线模型、离线 embeddings，以及 OpenAI 切换逻辑。
- `src/langchain_lab/flow.py`：六个可单独运行的阶段。
- `tests/test_flow.py`：每个阶段的最小回归测试。
- `.env.example`：真实模型配置模板。
