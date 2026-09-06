import pytest

from game_lore_rag.corpus import load_corpus
from game_lore_rag.retriever import EmbeddingRetriever
from game_lore_rag.generator import OllamaGenerator
from game_lore_rag.pipeline import RagPipeline

@pytest.mark.integration
def test_pipeline():
    docs = load_corpus("data/corpus/npc.yml")
    pipeline = RagPipeline(EmbeddingRetriever(), OllamaGenerator())
    pipeline.build(docs)

    question = "Who is Jordan?"
    k_results = 3
    result = pipeline.answer(question, k = k_results)
    assert result.query == question
    assert len(result.retrieved) == k_results
    print(result.answer)
