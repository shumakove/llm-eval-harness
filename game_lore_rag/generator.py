from abc import ABC, abstractmethod

from anthropic import Anthropic
from dotenv import load_dotenv
from ollama import Client as OllamaClient

from game_lore_rag.retriever import Document

SYSTEM_PROMPT = (
    "You answer questions about game lore using only the context provided "
    "below. Do not use any knowledge outside of it. If the context does not "
    "contain the answer, say you don't know."
)


def _build_user_message(query: str, context: list[Document]) -> str:
    context_block = "\n\n".join(f"[{doc.id}] {doc.text}" for doc in context)
    return f"Context:\n{context_block}\n\nQuestion: {query}"


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, context: list[Document]) -> str:
        pass


class AnthropicGenerator(Generator):
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        load_dotenv()
        self.client = Anthropic()
        self.model = model

    def generate(self, query: str, context: list[Document]) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query, context)}],
        )
        return message.content[0].text


class OllamaGenerator(Generator):
    def __init__(self, model: str = "qwen3:1.7b"):
        self.client = OllamaClient()
        self.model = model

    def generate(self, query: str, context: list[Document]) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(query, context)},
            ],
        )
        return response["message"]["content"]
