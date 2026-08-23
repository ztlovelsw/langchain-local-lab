from langchain_lab.flow import (
    run_basic_chain,
    run_langgraph,
    run_memory,
    run_rag,
    run_structured_output,
    run_tools,
)
from langchain_lab.models import OfflineChatModel


def test_basic_chain_is_composed_and_returns_text() -> None:
    result = run_basic_chain(OfflineChatModel())
    assert isinstance(result, str)
    assert "Runnable" in result


def test_structured_output_matches_schema() -> None:
    plan = run_structured_output(OfflineChatModel())
    assert plan.destination == "重庆"
    assert plan.days == 2
    assert plan.highlights


def test_rag_splits_and_retrieves_relevant_documents() -> None:
    result = run_rag(OfflineChatModel())
    assert result["chunks"]
    assert result["retrieved"]
    assert any(doc.metadata["source"] == "wushan.txt" for doc in result["retrieved"])
    assert "retriever" in result["answer"]


def test_message_history_is_scoped_to_a_session() -> None:
    answers = run_memory(OfflineChatModel())
    assert answers[0]
    assert "小林" in answers[1]


def test_tool_is_invoked_through_a_runnable() -> None:
    result = run_tools()
    assert result["tool_name"] == "lookup_weather"
    assert "重庆" in result["result"]


def test_langgraph_returns_state() -> None:
    result = run_langgraph(OfflineChatModel())
    assert result["question"]
    assert result["answer"]
