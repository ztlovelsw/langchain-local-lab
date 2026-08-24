# 第 5 课：工具和 Agent

模型只会“说”，不会“做”。工具（Tool）让模型能够调用外部能力——查数据库、算数学、发请求、执行搜索——而 Agent 的核心循环是：

```text
模型提出工具调用 → 程序执行工具 → 把结果回传模型 → 模型再决定下一步
```

本课的目标：先理解 `@tool` 的输入 schema 和 Runnable 流，再看真实模型如何驱动 tool-calling。

## 先看代码

在 `src/langchain_lab/flow.py` 中：

```python
@tool
def lookup_weather(city: str) -> str:
    """查询教学用的示例天气数据。"""
    weather = {
        "重庆": "重庆：多云，24°C，适合步行但请带雨具。",
        "北京": "北京：晴，20°C，空气良好。",
    }
    return weather.get(city, f"{city}：暂无教学数据。")
```

以及 `run_tools`：

```python
planner = RunnableLambda(lambda question: {"city": "重庆" if "重庆" in question else "北京"})
tool_chain = planner | lookup_weather
result = tool_chain.invoke("请查询重庆天气")
return {"tool_name": lookup_weather.name, "result": result}
```

三件新东西：

1. **`@tool` 装饰器**：把一个普通函数变成 `BaseTool`。函数签名 `city: str` 自动成为工具的输入 schema（JSON Schema），docstring 成为工具的描述。
2. **`RunnableLambda`**：一段普通的 Python 逻辑，也能作为 Runnable 参与 `|` 组合。
3. **工具即 Runnable**：`lookup_weather` 可以直接被 `|` 连接，`planner` 的输出 `{"city": "重庆"}` 会被当作工具入参。

运行：

```powershell
python run.py --provider offline --stage tools
```

输出：

```text
{'tool_name': 'lookup_weather', 'result': '重庆：多云，24°C，适合步行但请带雨具。'}
```

## 观察工具契约

加几行打印：

```python
print(lookup_weather.name)                      # 工具名
print(lookup_weather.description)               # 工具的说明
print(lookup_weather.args)                      # 输入 schema（JSON Schema 字典）
print(lookup_weather.invoke({"city": "北京"}))  # 直接调用工具
```

你会发现：**工具就是“一个有名字、有描述、有输入 schema 的函数”**。Agent 的智能不在于工具本身，而在于模型能根据描述决定“该调用哪个工具、传什么参数”。

## 亲手改一改

### 1. 增加第二个工具

```python
@tool
def lookup_ticket(from_city: str, to_city: str) -> str:
    """查询两城之间的高铁票价。"""
    return f"{from_city} → {to_city}：二等座约 ¥154。"
```

用 `RunnableLambda` 或直接在 `run_tools` 里调用它，观察多参数工具如何接收入参。

### 2. 换成真实模型的 tool-calling Agent

README 说“Agent 的核心不是‘魔法’”。真实模型的 tool-calling 分三步：

1. 把工具传给模型（`model.bind_tools([lookup_weather, lookup_ticket])`）；
2. 模型返回 `AIMessage`，其中可能带 `tool_calls`（要调用的工具名 + 参数）；
3. 程序执行工具，把 `ToolMessage` 回传给模型，模型基于结果继续回答。

> 提示：LangChain 的 `create_agent`/LangGraph 的预构建 `create_react_agent` 能省去手写循环。先把上面的三步循环读懂，再上预构建封装。

### 3. 制造一次“工具调用失败”

让工具抛异常，或让 `planner` 传一个 schema 里不存在的字段，观察错误如何从 Runnable 流里冒出来。

## 小挑战

1. 给 `lookup_weather` 增加第二个参数 `unit: str = "celsius"`，看 `args` 的 schema 如何变化（默认值会变成 `default` 字段）。
2. 用 `langchain_core.tools.StructuredTool.from_function` 重写 `lookup_weather`，对比与 `@tool` 写法的差别。
3. 阅读 `run_langgraph` 的 `GraphState`，思考：如果要把“模型提调用 → 执行工具 → 回传结果”建模成状态图，需要哪些节点和状态字段？

## Q&A

这里会记录你在学习过程中提出的问题和答案。

## Quiz History

尚未测验。
