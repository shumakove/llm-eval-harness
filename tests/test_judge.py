import pytest
from llm_eval_harness.metrics.generation_metrics import calculate_faithfulness

CONTEXT_CHUNK = [
    "Jordan is the high magister of Dion.",
    "He was born in Caplin and lost his parents to the Black Knight Guild.",
    "He has sworn to uncover who ordered the attack."
]

@pytest.mark.integration
def test_judge_supported_claim():
    answer = "Jordan was born in Caplin"
    faithfulness = calculate_faithfulness(answer,CONTEXT_CHUNK, query="Who is Jordan?")

    assert faithfulness > 0.5


@pytest.mark.integration
def test_judge_unsupported_claim():
    answer = "Caplin is a capital of Pandamonia"
    faithfulness = calculate_faithfulness(answer,CONTEXT_CHUNK, query="What is the capital of Pandamonia")

    assert faithfulness == 0.0
    