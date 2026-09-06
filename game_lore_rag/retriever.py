from pydantic.dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from fastembed import TextEmbedding

@dataclass
class Document:

    id: str
    text: str


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        pass

    @abstractmethod
    def build(self, documents: list[Document]) -> None:
        pass

class EmbeddingRetriever(Retriever):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = TextEmbedding(model_name=model_name)
        self._ids: list[str] = []
        self._vectors: np.ndarray = np.empty((0, 0))

    def build(self, documents: list[Document]) -> None:
        self._ids = [document.id for document in documents]
        self._vectors = np.array(list(self.embedding_model.embed([document.text for document in documents])))

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        query_vector = next(iter(self.embedding_model.embed([query])))
        scores = self._vectors @ query_vector
        top_k_indices = np.argsort(scores)[::-1][:k]
        return [(self._ids[i], float(scores[i])) for i in top_k_indices]