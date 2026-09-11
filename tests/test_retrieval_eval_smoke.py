import pytest
from tests.conftest import GOLDEN_DATASET_PATH, CORPUS_DATA_PATH
from llm_eval_harness.datasets.golden import load_golden_set
from llm_eval_harness.runners.retrieval_runner import run_retrieval_eval
from game_lore_rag.corpus import load_corpus
from game_lore_rag.retriever import EmbeddingRetriever

@pytest.mark.integration
def test_retrieval_runner():
    golden_data_set = load_golden_set(GOLDEN_DATASET_PATH)
    corpus_data = load_corpus(CORPUS_DATA_PATH)
    retreiver = EmbeddingRetriever()
    retreiver.build(corpus_data)
    k = max(len(case.relevant_ids) for case in golden_data_set)
    report = run_retrieval_eval(golden_data_set, retreiver, k)

    assert len(report.per_case) == len(golden_data_set), "Number of cases processed not equal to original dataset"
    assert report.n_skipped + report.n_scored == len(golden_data_set), "Calculation of scored/skipped cases failed"
    assert report.n_skipped == 3, "Unrelated cases in golden dataset calculaton failed"
    assert 0 <= report.context_mean_precision <= 1
    assert 0 <= report.context_mean_recall <= 1
    assert {r.case_id for r in report.per_case} == {c.id for c in golden_data_set}, "Sequence of ids incorrect"
    print(report)
