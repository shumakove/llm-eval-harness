from pydantic.dataclasses import dataclass

@dataclass
class EvalCase:
    id: str
    query: str
    relevant_ids: list[str]
    