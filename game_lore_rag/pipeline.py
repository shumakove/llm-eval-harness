from pydantic.dataclasses import dataclass

from game_lore_rag.generator import Generator
from game_lore_rag.retriever import Document, Retriever


@dataclass
class RagResult:
    query: str
    retrieved: list[tuple[Document, float]]
    answer: str


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: Generator):
        self.retriever = retriever
        self.generator = generator
        self._documents_by_id: dict[str, Document] = {}

    def build(self, documents: list[Document]) -> None:
        self.retriever.build(documents)
        self._documents_by_id = {document.id: document for document in documents}

    def answer(self, query: str, k: int = 10) -> RagResult:
        hits = self.retriever.retrieve(query, k)
        retrieved = [(self._documents_by_id[doc_id], score) for doc_id, score in hits]
        generated = self.generator.generate(query, [document for document, _ in retrieved])
        return RagResult(query=query, retrieved=retrieved, answer=generated)
