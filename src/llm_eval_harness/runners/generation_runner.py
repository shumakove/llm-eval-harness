import statistics
from pydantic.dataclasses import dataclass
from llm_eval_harness.datasets.eval_case import EvalCase
from llm_eval_harness.judge.judge import JudgeError
from llm_eval_harness.metrics.generation_metrics import calculate_faithfulness
from game_lore_rag.pipeline import RagPipeline


@dataclass
class GenerationCaseResult:
    case_id: str
    faithfulness: float | None
    judge_error: bool


@dataclass
class GenerationReport:
    per_case: list[GenerationCaseResult]
    mean_faithfulness: float
    n_scored: int
    n_skipped: int
    n_judge_errors: int


def run_generation_eval(golden_set: list[EvalCase], pipeline: RagPipeline, k: int) -> GenerationReport:
    per_case = []


    for case in golden_set:
        data = pipeline.answer(case.query,k)
        context_chunks = [doc.text for doc, _ in data.retrieved]
        judge_error = False

        try:
            faithfulness = calculate_faithfulness(data.answer, context_chunks,query=case.query)
        except JudgeError:
            faithfulness = None
            judge_error = True
        per_case.append(GenerationCaseResult(case.id, faithfulness, judge_error))
            
    scored = [r for r in per_case if r.faithfulness is not None]
    judge_errors = [r for r in per_case if r.judge_error]
    n_scored = len(scored)
    n_judge_errors = len(judge_errors)
    n_skipped = len(per_case) - n_scored - n_judge_errors
    
    mean_faithfulness = round(statistics.mean(r.faithfulness for r in scored),3)    
    return GenerationReport(per_case, mean_faithfulness,n_scored, n_skipped,n_judge_errors)