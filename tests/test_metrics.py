from __future__ import annotations

import math

from src.metrics import predictive_entropy


def test_predictive_entropy_is_zero_for_one_hot():
    assert predictive_entropy([1.0, 0.0, 0.0, 0.0]) < 1e-6


def test_predictive_entropy_is_one_for_uniform():
    n = 4
    uniform = [1.0 / n] * n
    assert math.isclose(predictive_entropy(uniform), 1.0, abs_tol=1e-6)


def test_predictive_entropy_is_bounded_in_unit_interval():
    probs = [0.7, 0.1, 0.1, 0.1]
    entropy = predictive_entropy(probs)
    assert 0.0 <= entropy <= 1.0


def test_predictive_entropy_single_class_is_zero():
    assert predictive_entropy([1.0]) == 0.0
