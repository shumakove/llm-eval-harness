from pathlib import Path
from game_lore_rag.retriever import Document
from yaml import safe_load

def load_corpus(path: Path) -> list[Document]:
    results = []
    

    with open(path, "r") as f:
        data = f.read()
        records = safe_load(data)

    for record in records:
        result = Document(id=record["id"],text=record["text"].strip())
        results.append(result)

    return results