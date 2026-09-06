from llm_eval_harness.datasets.eval_case import EvalCase

def calculate_context_recall(case: EvalCase, retreived_ids: list[str]) -> float | None:
    a = set(retreived_ids)
    b = set(case.relevant_ids)
    if len(case.relevant_ids) == 0:
        return None
    return len(a & b) / len (b)

def calculate_context_precision(case: EvalCase, retreived_ids: list[str]) -> float | None:
    if len(case.relevant_ids) == 0:
        return None
    relevant_found = 0
    total = 0
    for i, doc_id in enumerate(retreived_ids,start= 1):
        if doc_id in case.relevant_ids:
            relevant_found += 1
            total += relevant_found / i
    
    return round(total / len(case.relevant_ids), 3) 