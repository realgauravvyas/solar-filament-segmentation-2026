"""Which candidates are submitted? Expected-PQ scoring and selection.

Why a learned selector
----------------------
The competition scores a submission with Panoptic Quality,

    PQ = sum(IoU of matched pairs) / (TP + 0.5 FP + 0.5 FN),

pooled over every annotator's view of every frame. Take one candidate in a
frame that has V views and suppose it matches a drawing in m of them, with
IoUs that sum to s. Submitting it changes the pooled numerator by ``+s`` and
the denominator by ``+0.5 V`` (in a matching view a false negative, worth
0.5, becomes a true positive, worth 1; in every other view the candidate is a
false positive, worth 0.5). Submitting it therefore raises PQ exactly when

    s / V  >  PQ / 2,

i.e. when the candidate's *matched IoU averaged over views* exceeds half the
current PQ. That average, called the candidate's value here, is

    value = mean over views of (IoU with its matched drawing, or 0 if none).

It rewards what the metric rewards: being matched at all (IoU > 0.5), being
matched tightly, and being drawn by most annotators rather than one of three.

The value of a test candidate is unknown, so a gradient-boosted regressor
(LightGBM) is trained to predict it from the candidate's features, using
candidates generated on held-out folds, where the networks have not seen the
frame. Selection is then greedy: candidates are visited by decreasing
predicted value and accepted if the value clears a threshold and the
candidate does not overlap what has been accepted already. Submitted
instances never overlap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import config
from .candidates import FEATURE_NAMES, Candidate, feature_matrix


@dataclass
class SelectionParams:
    """Knobs of the greedy selection."""

    value_threshold: float = 0.2   # theory: half the expected PQ; tuned by cross-validation
    max_overlap: float = 0.25      # reject a candidate if this share of it is already taken
    min_area: int = 60             # after removing taken pixels


LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    verbosity=-1,
    n_jobs=4,
)
LGB_ROUNDS = 400

#: The scorer is a small bag of boosted models that differ only in their random row and
#: feature subsampling; averaging them removes most of the seed-to-seed noise of one model.
SCORER_SEEDS = (0, 1, 2, 3, 4)


def fit_scorer(
    features: np.ndarray, values: np.ndarray, seeds: tuple[int, ...] = SCORER_SEEDS
) -> list[lgb.Booster]:
    """Fit the value regressor on (candidate features, candidate values)."""
    dataset = lgb.Dataset(features, label=values, feature_name=FEATURE_NAMES, free_raw_data=False)
    return [
        lgb.train({**LGB_PARAMS, "seed": seed}, dataset, num_boost_round=LGB_ROUNDS) for seed in seeds
    ]


def predict_features(scorer: list[lgb.Booster], features: np.ndarray) -> np.ndarray:
    """Predicted value for each row of a feature matrix: the mean over the bag."""
    if len(features) == 0:
        return np.zeros(0)
    return np.mean([model.predict(features) for model in scorer], axis=0)


def save_scorer(scorer: list[lgb.Booster], path: Path) -> None:
    """Store the bag as one JSON file: the feature order and each model in LightGBM's text format."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"feature_names": FEATURE_NAMES,
                   "boosters": [model.model_to_string() for model in scorer]}, handle)


def load_scorer(path: Path) -> list[lgb.Booster]:
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    if stored["feature_names"] != FEATURE_NAMES:
        raise ValueError("the stored scorer was trained on a different feature list")
    return [lgb.Booster(model_str=text) for text in stored["boosters"]]


def cross_fitted_values(
    features: np.ndarray, values: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0
) -> np.ndarray:
    """Predicted value of every candidate from a scorer that never saw its frame.

    ``groups`` holds the frame index of each candidate; frames, not candidates,
    are split, so all candidates of a frame are predicted by the same model.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    fold_of_group = dict(zip(unique, rng.permutation(len(unique)) % n_splits))
    fold = np.array([fold_of_group[g] for g in groups])
    predicted = np.zeros(len(values), dtype=np.float64)
    for k in range(n_splits):
        held_out = fold == k
        if not held_out.any():
            continue
        scorer = fit_scorer(features[~held_out], values[~held_out])
        predicted[held_out] = predict_features(scorer, features[held_out])
    return predicted


def select_indexed(
    candidates: list[Candidate],
    predicted_values: np.ndarray,
    params: SelectionParams | None = None,
    size: int = config.IMAGE_SIZE,
) -> list[tuple[int, np.ndarray]]:
    """Greedy, non-overlapping selection for one frame.

    Candidates are visited by decreasing predicted value. One is accepted if
    its value clears the threshold and at most ``max_overlap`` of it is
    already taken; the taken pixels are removed from it, so accepted
    instances never overlap. Returns ``(candidate index, full-frame boolean
    mask)`` pairs in acceptance order.
    """
    params = params or SelectionParams()
    occupied = np.zeros((size, size), dtype=bool)
    accepted: list[tuple[int, np.ndarray]] = []
    for index in np.argsort(-predicted_values):
        if predicted_values[index] < params.value_threshold:
            break
        candidate = candidates[index]
        height, width = candidate.mask.shape
        window = occupied[candidate.y0:candidate.y0 + height, candidate.x0:candidate.x0 + width]
        if (candidate.mask & window).sum() > params.max_overlap * candidate.area:
            continue
        remaining = candidate.mask & ~window
        if remaining.sum() < params.min_area:
            continue
        window |= remaining
        full = np.zeros((size, size), dtype=bool)
        full[candidate.y0:candidate.y0 + height, candidate.x0:candidate.x0 + width] = remaining
        accepted.append((int(index), full))
    return accepted


def select(
    candidates: list[Candidate],
    predicted_values: np.ndarray,
    params: SelectionParams | None = None,
) -> list[np.ndarray]:
    """Like ``select_indexed`` but returns only the masks."""
    return [mask for _, mask in select_indexed(candidates, predicted_values, params)]


def predict_values(scorer: list[lgb.Booster], candidates: list[Candidate]) -> np.ndarray:
    """Predicted value of each candidate of a frame."""
    if not candidates:
        return np.zeros(0)
    return predict_features(scorer, feature_matrix(candidates))
