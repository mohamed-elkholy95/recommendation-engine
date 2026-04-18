"""Tests for src.utils.seed.set_all_seeds."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.utils.seed import DEFAULT_SEED, set_all_seeds


def test_default_seed_is_forty_two() -> None:
    assert DEFAULT_SEED == 42


def test_python_random_is_deterministic_after_seeding() -> None:
    set_all_seeds(123)
    first = [random.random() for _ in range(5)]
    set_all_seeds(123)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_numpy_random_is_deterministic_after_seeding() -> None:
    set_all_seeds(7)
    first = np.random.rand(5).tolist()
    set_all_seeds(7)
    second = np.random.rand(5).tolist()
    assert first == second


def test_different_seeds_produce_different_streams() -> None:
    set_all_seeds(1)
    a = random.random()
    set_all_seeds(2)
    b = random.random()
    assert a != b


def test_default_seed_is_used_when_arg_omitted() -> None:
    set_all_seeds()
    a = random.random()
    set_all_seeds(DEFAULT_SEED)
    b = random.random()
    assert a == b


def test_negative_seed_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_all_seeds(-1)
