"""Tests for src.evaluation — ranking metrics with hand-computed expected values."""

from __future__ import annotations

import math

import pytest

from src.data.ids import ItemIdx
from src.evaluation import (
    average_precision_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)


def _items(*ids: int) -> list[ItemIdx]:
    return [ItemIdx(i) for i in ids]


def _as_set(*ids: int) -> set[ItemIdx]:
    return {ItemIdx(i) for i in ids}


# -----------------------------------------------------------------------------
# hit_rate_at_k
# -----------------------------------------------------------------------------


def test_hit_rate_is_one_when_any_top_k_item_is_relevant() -> None:
    assert hit_rate_at_k(_items(3, 1, 5, 2, 4), _as_set(5), k=3) == 1.0


def test_hit_rate_is_zero_when_no_top_k_item_is_relevant() -> None:
    assert hit_rate_at_k(_items(3, 1, 5, 2, 4), _as_set(4), k=3) == 0.0


def test_hit_rate_ignores_tail_beyond_k() -> None:
    # concept: the relevant item sits at rank 4 — outside k=3 → miss.
    assert hit_rate_at_k(_items(1, 2, 3, 4, 5), _as_set(4), k=3) == 0.0


def test_hit_rate_returns_zero_when_relevant_is_empty() -> None:
    assert hit_rate_at_k(_items(1, 2, 3), set(), k=3) == 0.0


# -----------------------------------------------------------------------------
# ndcg_at_k
# -----------------------------------------------------------------------------


def test_ndcg_is_one_when_all_top_k_are_relevant_and_cover_ideal() -> None:
    # recommended=[3,1,5], relevant={1,3,5} ⇒ DCG == IDCG.
    assert ndcg_at_k(_items(3, 1, 5, 2, 4), _as_set(1, 3, 5), k=3) == pytest.approx(1.0)


def test_ndcg_matches_hand_computed_fraction() -> None:
    # recommended=[1,2,3,4,5], relevant={3,5}, k=5 ⇒ positions 3 and 5.
    # DCG = 1/log2(4) + 1/log2(6); IDCG = 1/log2(2) + 1/log2(3).
    dcg = 1.0 / math.log2(4) + 1.0 / math.log2(6)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    expected = dcg / idcg
    got = ndcg_at_k(_items(1, 2, 3, 4, 5), _as_set(3, 5), k=5)
    assert got == pytest.approx(expected)


def test_ndcg_is_zero_with_no_relevant_hits_in_top_k() -> None:
    assert ndcg_at_k(_items(3, 1, 5), _as_set(9), k=3) == 0.0


def test_ndcg_is_zero_when_relevant_is_empty() -> None:
    assert ndcg_at_k(_items(1, 2, 3), set(), k=3) == 0.0


# -----------------------------------------------------------------------------
# average_precision_at_k
# -----------------------------------------------------------------------------


def test_average_precision_perfect_ranking_is_one() -> None:
    assert average_precision_at_k(_items(3, 1, 5), _as_set(1, 3, 5), k=3) == pytest.approx(1.0)


def test_average_precision_matches_hand_computed_value() -> None:
    # recommended=[1,2,3,4,5], relevant={3,5}, k=5:
    # AP = (P@3 * 1 + P@5 * 1) / min(5, 2) = (1/3 + 2/5) / 2
    expected = (1.0 / 3.0 + 2.0 / 5.0) / 2.0
    got = average_precision_at_k(_items(1, 2, 3, 4, 5), _as_set(3, 5), k=5)
    assert got == pytest.approx(expected)


def test_average_precision_normalises_by_min_k_and_relevant_size() -> None:
    # recommended=[1,2,3], relevant={1,2,3,4}, k=3:
    # AP = (1/1 + 2/2 + 3/3) / min(3, 4) = 3/3 = 1.0
    assert average_precision_at_k(_items(1, 2, 3), _as_set(1, 2, 3, 4), k=3) == pytest.approx(1.0)


def test_average_precision_is_zero_with_no_hits() -> None:
    assert average_precision_at_k(_items(1, 2, 3), _as_set(9, 10), k=3) == 0.0


def test_average_precision_is_zero_when_relevant_is_empty() -> None:
    assert average_precision_at_k(_items(1, 2, 3), set(), k=3) == 0.0


# -----------------------------------------------------------------------------
# Shared input validation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [hit_rate_at_k, ndcg_at_k, average_precision_at_k])
def test_metrics_reject_non_positive_k(metric: object) -> None:
    from typing import Callable

    fn: Callable[..., float] = metric  # type: ignore[assignment]
    with pytest.raises(ValueError, match="k must"):
        fn(_items(1, 2, 3), _as_set(1), 0)
