import statistics
from llm_eval_harness.datasets.eval_case import EvalCase
from game_lore_rag.retriever import Retriever
from pydantic.dataclasses import dataclass
from llm_eval_harness.metrics.retrieval_metrics import calculate_context_precision, calculate_context_recall

@dataclass
class CaseResult:
    case_id: str
    context_precision: float | None
    context_recall: float | None

@dataclass
class RetrievalReport:
    per_case: list[CaseResult]
    context_mean_precision: float
    context_mean_recall: float
    n_scored: int
    n_skipped: int

def run_retrieval_eval(golden_set: list[EvalCase], retreiver: Retriever, k: int) -> RetrievalReport:
    per_case = []
    for case in golden_set:
        data = retreiver.retrieve(case.query, k)
        retreived_ids = [doc_id for doc_id, _ in data]
        context_precision = calculate_context_precision(case, retreived_ids)
        context_recall = calculate_context_recall(case, retreived_ids)
        
        per_case.append(CaseResult(case.id, context_precision=context_precision, context_recall=context_recall))
    scored = [r for r in per_case if r.context_precision is not None ]
    n_scored = len(scored)
    n_skipped = len(per_case)  - n_scored
    mean_recall = round(statistics.mean(r.context_recall for r in scored), 3)
    mean_precision = round(statistics.mean(r.context_precision for r in scored), 3)

    return(RetrievalReport(per_case, mean_precision, mean_recall,n_scored, n_skipped))
