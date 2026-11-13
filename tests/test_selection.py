"""Greedy selection must respect the threshold and never produce overlapping instances."""

import numpy as np

from filaseg import selection
from filaseg.candidates import Candidate


def _candidate(y0, x0, height, width):
    return Candidate(y0, x0, np.ones((height, width), dtype=bool))


def test_selection_is_disjoint_and_ordered():
    candidates = [_candidate(0, 0, 20, 20), _candidate(0, 18, 20, 20), _candidate(5, 5, 20, 20)]
    values = np.array([0.9, 0.5, 0.8])
    params = selection.SelectionParams(value_threshold=0.2, max_overlap=0.25, min_area=10)
    accepted = selection.select_indexed(candidates, values, params, size=64)
    # The third candidate overlaps the first by more than 25 % and is rejected.
    assert [index for index, _ in accepted] == [0, 1]
    union = np.zeros((64, 64), dtype=int)
    for _, mask in accepted:
        union += mask
    assert union.max() == 1


def test_threshold_cuts_the_list():
    candidates = [_candidate(0, 0, 10, 10), _candidate(30, 30, 10, 10)]
    values = np.array([0.5, 0.1])
    params = selection.SelectionParams(value_threshold=0.2, min_area=10)
    assert len(selection.select_indexed(candidates, values, params, size=64)) == 1
