import pytest
from tests.conftest import GOLDEN_DATASET_PATH, CORPUS_DATA_PATH
from llm_eval_harness.datasets.golden import load_golden_set
from llm_eval_harness.runners.generation_runner import run_generation_eval

from game_lore_rag.corpus import load_corpus
from game_lore_rag.retriever import EmbeddingRetriever
from game_lore_rag.pipeline import RagPipeline
from game_lore_rag.generator import AnyModelGenerator

@pytest.mark.integration
def test_generation_runner():
    golden_data_set = load_golden_set(GOLDEN_DATASET_PATH)
    corpus_data = load_corpus(CORPUS_DATA_PATH)
    retreiver = EmbeddingRetriever()
    k = max(len(case.relevant_ids) for case in golden_data_set)
    pipeline = RagPipeline(retreiver,AnyModelGenerator("ds/deepseek-v4-flash"))
    pipeline.build(corpus_data)
    report = run_generation_eval(golden_data_set,pipeline,k)
    assert len(report.per_case) == len(golden_data_set)
    assert report.n_skipped + report.n_scored + report.n_judge_errors == len(golden_data_set)
    assert 0 <= report.mean_faithfulness <= 1 
    print(report)
