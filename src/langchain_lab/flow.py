"""End-to-end LangChain learning flow.

Each function is intentionally small.  Read them in order: prompt + model,
structured output, RAG, conversation history, tools, and finally LangGraph.
"""

from __future__ import annotations

import argparse
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.checkpoint.memory import InMemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .models import KeywordEmbeddings, build_chat_model


def run_basic_chain(model: Any) -> str:
    """PromptTemplate -> ChatModel -> StrOutputParser (LCEL)."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一位清晰、简洁的 LangChain 教练。"),
            ("human", "用一句话解释这个概念：{concept}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"concept": "Runnable"})


class TravelPlan(BaseModel):
    """The schema the structured-output step must produce."""

    destination: str = Field(description="目的地")
    days: int = Field(description="旅行天数")
    highlights: list[str] = Field(description="亮点列表")


def run_structured_output(model: Any) -> TravelPlan:
    """ChatModel -> PydanticOutputParser, making output machine-readable."""

    parser = PydanticOutputParser(pydantic_object=TravelPlan)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "只输出 JSON，不要输出 Markdown。\n{format_instructions}"),
            ("human", "请为我规划一个两天的重庆旅行。"),
        ]
    )
    chain = prompt | model | parser
    return chain.invoke({"format_instructions": parser.get_format_instructions()})


def _format_docs(documents: list[Document]) -> str:
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in documents
    )


def run_rag(model: Any) -> dict[str, Any]:
    """Load -> split -> embed -> retrieve -> prompt -> model -> parse."""

    documents = [
        Document(
            page_content=(
                "奉节白帝城位于长江三峡入口，适合安排半天到一天的历史人文游览。"
                "游客可以了解三国文化，并眺望瞿塘峡。"
            ),
            metadata={"source": "fengjie.txt"},
        ),
        Document(
            page_content=(
                "巫山小三峡以峡谷、溪流和自然风光见长，适合乘船游览。"
                "行程通常需要预留大半天，并关注当天的水上交通安排。"
            ),
            metadata={"source": "wushan.txt"},
        ),
        Document(
            page_content=(
                "云阳龙缸有高山草甸、悬崖步道和观景平台，适合喜欢户外景观的游客。"
                "建议穿防滑鞋并预留一整天。"
            ),
            metadata={"source": "yunyang.txt"},
        ),
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=90, chunk_overlap=15)
    chunks = splitter.split_documents(documents)
    vector_store = InMemoryVectorStore.from_documents(chunks, KeywordEmbeddings())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    question = "哪里适合乘船看峡谷？"
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是旅游助手，只能根据 Context 回答；如果 Context 没有答案就说不知道。",
            ),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | model
        | StrOutputParser()
    )
    retrieved = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return {"chunks": chunks, "retrieved": retrieved, "answer": answer}


def run_memory(model: Any) -> list[str]:
    """Persist conversation state with LangGraph's current checkpointer API."""

    class ConversationState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def call_model(state: ConversationState) -> dict[str, list[BaseMessage]]:
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ConversationState)
    builder.add_node("chat", call_model)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "lesson-1"}}
    first_state = graph.invoke(
        {"messages": [HumanMessage(content="记住：我叫小林。")]}, config=config
    )
    second_state = graph.invoke(
        {"messages": [HumanMessage(content="我刚才告诉你的名字是什么？")]},
        config=config,
    )
    return [
        str(first_state["messages"][-1].content),
        str(second_state["messages"][-1].content),
    ]


@tool
def lookup_weather(city: str) -> str:
    """查询教学用的示例天气数据。"""

    weather = {
        "重庆": "重庆：多云，24°C，适合步行但请带雨具。",
        "北京": "北京：晴，20°C，空气良好。",
    }
    return weather.get(city, f"{city}：暂无教学数据。")


def run_tools() -> dict[str, str]:
    """Show the tool contract and a tiny deterministic tool-calling loop."""

    planner = RunnableLambda(
        lambda question: {"city": "重庆" if "重庆" in question else "北京"}
    )
    tool_chain = planner | lookup_weather
    result = tool_chain.invoke("请查询重庆天气")
    return {"tool_name": lookup_weather.name, "result": result}


class GraphState(TypedDict, total=False):
    question: str
    answer: str


def run_langgraph(model: Any) -> dict[str, str]:
    """Represent the answer step as a tiny explicit state graph."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是 LangGraph 节点。"), ("human", "回答：{question}")]
    )
    answer_chain = prompt | model | StrOutputParser()

    def answer_node(state: GraphState) -> GraphState:
        return {"answer": answer_chain.invoke({"question": state["question"]})}

    builder = StateGraph(GraphState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    return graph.invoke({"question": "LangGraph 和 Runnable 有什么关系？"})


def run_all(provider: str | None = None) -> dict[str, Any]:
    model = build_chat_model(provider)
    basic = run_basic_chain(model)
    structured = run_structured_output(model)
    rag = run_rag(model)
    memory = run_memory(model)
    tools = run_tools()
    graph = run_langgraph(model)
    return {
        "provider": provider or "env/default",
        "basic": basic,
        "structured": structured.model_dump(),
        "rag": rag,
        "memory": memory,
        "tools": tools,
        "graph": graph,
    }


def _print_stage(name: str, value: Any) -> None:
    print(f"\n=== {name} ===")
    if name == "RAG":
        print(f"切分块数: {len(value['chunks'])}")
        print("检索来源:", [doc.metadata.get("source") for doc in value["retrieved"]])
        print("回答:", value["answer"])
    else:
        print(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangChain 本地全流程教学演示")
    parser.add_argument(
        "--provider",
        choices=["offline", "openai"],
        default=None,
        help="模型提供方；默认读取 MODEL_PROVIDER，未设置时为 offline",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "basic", "structured", "rag", "memory", "tools", "graph"],
        default="all",
        help="只运行某一个阶段，便于逐步学习",
    )
    args = parser.parse_args(argv)
    model = build_chat_model(args.provider)
    print(f"使用模型: {model.__class__.__name__}")

    if args.stage == "all":
        _print_stage("LCEL 基础链", run_basic_chain(model))
        _print_stage("结构化输出", run_structured_output(model).model_dump())
        _print_stage("RAG", run_rag(model))
        _print_stage("会话历史", run_memory(model))
        _print_stage("工具流", run_tools())
        _print_stage("LangGraph", run_langgraph(model))
    elif args.stage == "basic":
        _print_stage("LCEL 基础链", run_basic_chain(model))
    elif args.stage == "structured":
        _print_stage("结构化输出", run_structured_output(model).model_dump())
    elif args.stage == "rag":
        _print_stage("RAG", run_rag(model))
    elif args.stage == "memory":
        _print_stage("会话历史", run_memory(model))
    elif args.stage == "tools":
        _print_stage("工具流", run_tools())
    elif args.stage == "graph":
        _print_stage("LangGraph", run_langgraph(model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
