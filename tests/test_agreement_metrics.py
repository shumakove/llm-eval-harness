"""Тесты слоя согласия.

Ожидаемые числа посчитаны на бумаге, а не сняты с прогона на собственных
данных: тест, запинненный своим же результатом, замораживает баг вместо
того, чтобы его ловить.
"""

import pytest

from llm_eval_harness.metrics.agreement_metrics import (
    AgreementMatrix,
    build_matrix,
    cohen_kappa,
    precision_unsupported,
    raw_agreement,
    recall_unsupported,
    summarize,
)

# Матрица с известным ответом, n = 100.
#
#   p0 = (50 + 25) / 100                              = 0.75
#   маргиналы эталона:  положительных 0.70, отрицательных 0.30
#   маргиналы судьи:    положительных 0.55, отрицательных 0.45
#   pe = 0.70 * 0.55 + 0.30 * 0.45 = 0.385 + 0.135    = 0.52
#   kappa = (0.75 - 0.52) / (1 - 0.52) = 0.23 / 0.48  = 0.479166...
#
# Пример выбран так, что типичные ошибки в pe дают другой ответ:
# квадраты маргиналов эталона дают kappa 0.405, судьи — 0.495.
KNOWN = AgreementMatrix(tp=50, fn=20, fp=5, tn=25)
KNOWN_KAPPA = 0.479


def _sequences(matrix: AgreementMatrix) -> tuple[list[bool], list[bool]]:
    """Разворачивает матрицу в две выровненные последовательности меток."""
    reference = [True] * (matrix.tp + matrix.fn) + [False] * (matrix.fp + matrix.tn)
    predicted = (
        [True] * matrix.tp
        + [False] * matrix.fn
        + [True] * matrix.fp
        + [False] * matrix.tn
    )
    return reference, predicted


def test_build_matrix_counts_four_cells():
    reference = [True, True, False, False]
    predicted = [True, False, True, False]

    assert build_matrix(reference, predicted) == AgreementMatrix(tp=1, fn=1, fp=1, tn=1)


def test_build_matrix_roundtrips_known_matrix():
    reference, predicted = _sequences(KNOWN)

    assert build_matrix(reference, predicted) == KNOWN
    assert KNOWN.n == 100


def test_build_matrix_rejects_length_mismatch():
    with pytest.raises(ValueError):
        build_matrix([True, False], [True])


def test_build_matrix_rejects_empty_input():
    with pytest.raises(ValueError):
        build_matrix([], [])


def test_raw_agreement_on_known_matrix():
    assert raw_agreement(KNOWN) == 0.75


def test_cohen_kappa_on_known_matrix():
    assert cohen_kappa(KNOWN) == pytest.approx(KNOWN_KAPPA, abs=0.0005)


def test_cohen_kappa_is_symmetric_in_its_raters():
    """kappa не зависит от того, кого считать эталоном, precision и recall —
    зависят. Тест держит эту разницу явной."""
    reference, predicted = _sequences(KNOWN)
    swapped = build_matrix(predicted, reference)

    assert cohen_kappa(swapped) == pytest.approx(cohen_kappa(KNOWN), abs=0.0005)
    assert swapped != KNOWN


def test_cohen_kappa_undefined_when_both_raters_constant():
    reference = [True] * 10
    predicted = [True] * 10

    assert cohen_kappa(build_matrix(reference, predicted)) is None


def test_cohen_kappa_near_zero_when_only_judge_is_constant():
    """Судья, штампующий один класс, согласия сверх случайного не даёт."""
    reference = [True] * 4 + [False] * 6
    predicted = [True] * 10

    assert cohen_kappa(build_matrix(reference, predicted)) == pytest.approx(0.0, abs=0.0005)


def test_precision_and_recall_on_known_matrix():
    assert precision_unsupported(KNOWN) == 0.909  # 50 / 55
    assert recall_unsupported(KNOWN) == 0.714  # 50 / 70


def test_precision_undefined_when_judge_flags_nothing():
    matrix = AgreementMatrix(tp=0, fn=7, fp=0, tn=3)

    assert precision_unsupported(matrix) is None
    assert recall_unsupported(matrix) == 0.0


def test_recall_undefined_when_reference_has_no_positives():
    matrix = AgreementMatrix(tp=0, fn=0, fp=2, tn=8)

    assert recall_unsupported(matrix) is None


def test_summarize_fills_every_field():
    reference, predicted = _sequences(KNOWN)
    report = summarize(reference, predicted)

    assert report.matrix == KNOWN
    assert report.raw_agreement == 0.75
    assert report.cohen_kappa == pytest.approx(KNOWN_KAPPA, abs=0.0005)
    assert report.precision_unsupported == 0.909
    assert report.recall_unsupported == 0.714
