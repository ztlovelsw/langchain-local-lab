"""Model and embedding adapters used by the tutorial.

The default model is deterministic and local.  It lets a learner understand
LangChain's composition model before adding credentials or downloading a
large model.  Set ``MODEL_PROVIDER=openai`` to use ``ChatOpenAI`` instead.
"""

from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class OfflineChatModel(BaseChatModel):
    """A tiny deterministic chat model for local teaching and tests.

    It is intentionally simple: it does not pretend to be intelligent or
    replace a production model.  Its job is to make every Runnable stage
    observable without a network connection or API key.
    """

    model_name: str = "offline-demo-model"

    @property
    def _llm_type(self) -> str:
        return "offline-demo"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        role_names = {
            "human": "Human",
            "user": "Human",
            "ai": "AI",
            "assistant": "AI",
            "system": "System",
        }
        text = "\n".join(
            f"{role_names.get(message.type, message.type)}: {message.content}"
            for message in messages
        )
        answer = self._respond(text)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=answer))]
        )

    @staticmethod
    def _last_user_text(messages_text: str) -> str:
        # ChatPromptTemplate renders role labels in a predictable way.  If a
        # raw prompt is passed, the complete text is still a useful fallback.
        matches = re.findall(r"(?:Human|用户|user)\s*:\s*(.*)", messages_text)
        return matches[-1].strip() if matches else messages_text.strip()

    def _respond(self, prompt_text: str) -> str:
        lower = prompt_text.lower()

        if "format instructions" in lower or "json" in lower:
            return json.dumps(
                {
                    "destination": "重庆",
                    "days": 2,
                    "highlights": ["山城步道", "长江夜景"],
                },
                ensure_ascii=False,
            )

        if "context:" in lower and "question:" in lower:
            return (
                "根据检索到的上下文：LangChain 把提示词、模型和输出解析器组合成 Runnable；"
                "做 RAG 时先切分文档，再用 embeddings 建索引，由 retriever 找回相关片段，"
                "最后把片段放进提示词交给模型回答。"
            )

        question = self._last_user_text(prompt_text)
        if "记住" in question or "名字" in question:
            if "小林" in prompt_text and "刚才" in question:
                return "你刚才说你叫小林。"
            return "好的，我会在本次会话中记住这条信息。"
        return f"（离线演示模型）我收到的问题是：{question}"


class KeywordEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings for the local RAG demo.

    Tokens are hashed into a fixed-size vector.  Shared words/characters
    therefore have a positive cosine contribution, which is enough to make
    retrieval behavior visible without downloading an embedding model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9]+", lowered)
        chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
        # Include character bigrams so short Chinese queries share signal with
        # Chinese documents, while preserving ordinary English word matches.
        bigrams = ["".join(chinese[i : i + 2]) for i in range(len(chinese) - 1)]
        return (*words, *chinese, *bigrams)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_chat_model(provider: str | None = None) -> BaseChatModel:
    """Build the configured chat model, with an explicit offline default."""

    load_dotenv()
    selected = (provider or os.getenv("MODEL_PROVIDER", "offline")).lower()
    if selected in {"offline", "fake", "demo"}:
        return OfflineChatModel()

    if selected != "openai":
        raise ValueError(
            f"Unsupported MODEL_PROVIDER={selected!r}; use 'offline' or 'openai'."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MODEL_PROVIDER=openai 需要 OPENAI_API_KEY；"
            "先复制 .env.example 为 .env 并填写密钥，或使用 --provider offline。"
        )

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": 0,
        "api_key": api_key,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
