from pathlib import Path

from yaml import safe_load

from llm_eval_harness.datasets.eval_case import EvalCase


def load_golden_set(path: Path) -> list[EvalCase]:
    results = []

    with open(path, "r") as f:
        data = f.read()
        records = safe_load(data)

    for record in records:
        result = EvalCase(id=record["id"], query=record["query"], relevant_ids=record["relevant_ids"])
        results.append(result)

    return results
