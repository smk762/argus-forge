from __future__ import annotations

import pytest

from argus_forge.heuristics import (
    CATEGORY_LABELS,
    DATASET_SIZE_GUIDE,
    TRAINING_BIAS,
    apply_overrides,
    dataset_size_status,
    suggest_training_params,
)
from argus_forge.models import TARGET_CATEGORIES, ParamOverrides


@pytest.mark.parametrize(
    ("count", "category", "repeats", "total_steps"),
    [
        # Parity cases mirroring suggestTrainingParams in argus-studio types.ts.
        (27, "identity", 6, 1620),  # 1500/270 = 5.55 -> 6
        (25, "identity", 6, 1500),  # exact division
        (20, "identity", 8, 1600),  # 7.5 rounds half-up like JS Math.round
        (60, "identity", 3, 1800),  # 2.5 -> 3 (banker's rounding would give 2)
        (1, "identity", 150, 1500),
        (200, "identity", 1, 200 * 10),  # floor at 1 repeat
        (30, "setting", 7, 2100),  # 2000/300 = 6.67 -> 7
        (0, "identity", 150, 0),  # empty set: repeats solve for n=1, steps stay 0
    ],
)
def test_suggest_training_params_parity(count: int, category: str, repeats: int, total_steps: int) -> None:
    p = suggest_training_params(count, category)  # type: ignore[arg-type]
    assert p.repeats == repeats
    assert p.epochs == 10
    assert p.total_steps == total_steps
    assert p.optimizer_steps == -(-total_steps // p.batch_size)


def test_category_bias() -> None:
    assert suggest_training_params(20, "identity").network_dim == 16
    assert suggest_training_params(20, "identity").network_alpha == 8
    assert suggest_training_params(20, "pose_composition").network_dim == 32
    assert suggest_training_params(20, "setting").network_alpha == 16


def test_apply_overrides_recomputes_steps() -> None:
    base = suggest_training_params(20, "identity")  # 8 repeats x 10 epochs
    out = apply_overrides(base, ParamOverrides(epochs=5, batch_size=1))
    assert out.epochs == 5
    assert out.repeats == base.repeats
    assert out.total_steps == 20 * base.repeats * 5
    assert out.optimizer_steps == out.total_steps
    # No overrides -> unchanged object semantics.
    assert apply_overrides(base, ParamOverrides()) == base


@pytest.mark.parametrize(
    ("count", "tone"),
    [(0, "empty"), (5, "low"), (20, "good"), (60, "high")],
)
def test_dataset_size_status_tones(count: int, tone: str) -> None:
    assert dataset_size_status(count, "identity").tone == tone


@pytest.mark.parametrize("category", TARGET_CATEGORIES)
def test_every_cortex_category_is_handled(category: str) -> None:
    """Every category in cortex's shared taxonomy has a bias, size guide and label.

    A new cortex category would otherwise only surface as a KeyError on the
    first manifest that used it.
    """
    assert category in TRAINING_BIAS
    assert category in DATASET_SIZE_GUIDE
    assert category in CATEGORY_LABELS
    assert suggest_training_params(25, category).images == 25  # type: ignore[arg-type]
    assert dataset_size_status(25, category).tone in {"low", "good", "high"}  # type: ignore[arg-type]


def test_tables_have_no_stale_categories() -> None:
    for table in (TRAINING_BIAS, DATASET_SIZE_GUIDE, CATEGORY_LABELS):
        assert set(table) == set(TARGET_CATEGORIES)
