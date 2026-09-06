import pytest
import numpy
from game_lore_rag.corpus import load_corpus
from game_lore_rag.retriever import Document, EmbeddingRetriever

def test_corpus_load():
    records : list[Document]
    records = load_corpus("data/corpus/npc.yml")
    assert len(records) == 75
    assert isinstance(records[0], Document)
    assert records[0].id == "npc.magister_jordan.origin"
    assert records[0].text == "Jordan is the high magister of Dion. He was born in Caplin and lost his parents to the Black Knight Guild. He has sworn to uncover who ordered the attack."

def test_retriever():
    records = load_corpus("data/corpus/npc.yml")
    ret = EmbeddingRetriever()
    ret.build(records)
    assert len(ret._ids) == 75 # records
    assert ret._vectors[0].size == 384 # 384 dimnsions

    s = ret.retrieve("кто такой Jordan?", 5)
    print(s)
    