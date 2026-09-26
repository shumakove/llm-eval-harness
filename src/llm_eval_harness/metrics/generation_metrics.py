from llm_eval_harness.judge.judge import decompose_claims, verify_claims


def calculate_faithfulness(answer: str, context: list[str], *, query: str) -> float | None:
    claims = decompose_claims(answer,query=query)
    if not claims:
        return None
    verdicts = verify_claims(claims, context)

    return round(sum([verdict.supported for verdict in verdicts]) / len(verdicts), 3)
