import pytest
from llm_eval_harness.datasets.eval_case import EvalCase
from llm_eval_harness.metrics.retrieval_metrics import calculate_context_recall, calculate_context_precision

def test_context_recall():
    case = EvalCase(id="x", query="x", relevant_ids=["a", "b"])
    assert calculate_context_recall(case, ["a","c"]) == 0.5

def test_context_precision():
    case = EvalCase(id='x', query='x', relevant_ids=['origin', 'quest_log', 'speech'])
    retrieved = ['origin', 'X', 'speech', 'Y', 'quest_log']
    assert calculate_context_precision(case, retrieved) == 0.756

