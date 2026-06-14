import numpy as np

try:
    import pytest
except ImportError:  # tests can also be run with the manual runner below
    pytest = None

from nugap.systems import tf
from nugap.metric import nu_gap, chordal_distance, winding_condition


def test_identity_is_zero():
    P = tf([1.0], [1.0, 1.0])  # 1/(s+1)
    assert nu_gap(P, P) < 1e-6


def test_range_and_symmetry():
    P1 = tf([1.0], [1.0, 1.0])      # 1/(s+1)
    P2 = tf([2.0], [1.0, 3.0, 2.0])  # (s+1)(s+2) denom
    d12 = nu_gap(P1, P2)
    d21 = nu_gap(P2, P1)
    assert 0.0 <= d12 <= 1.0
    assert abs(d12 - d21) < 1e-6


def test_close_stable_pair_is_small_and_well_posed():
    P1 = tf([1.0], [1.0, 1.0])  # 1/(s+1)
    P2 = tf([1.0], [1.0, 1.2])  # 1/(s+1.2)
    d, info = nu_gap(P1, P2, return_details=True)
    assert info["winding_ok"] is True
    assert 0.0 < d < 0.3


def test_increasing_separation_increases_metric():
    P1 = tf([1.0], [1.0, 1.0])
    near = nu_gap(P1, tf([1.0], [1.0, 1.5]))
    far = nu_gap(P1, tf([1.0], [1.0, 8.0]))
    assert near < far


def test_matlab_doc_example_unstable_vs_stable():
    # From MATLAB gapmetric docs: P1 = 1/(s-0.001) is unstable,
    # P2 = 1/(s+0.001) is stable, yet the systems are *close*.
    # The eta correction must make the winding condition hold so the
    # result is small, NOT 1.
    P1 = tf([1.0], [1.0, -0.001])
    P2 = tf([1.0], [1.0, 0.001])
    d, info = nu_gap(P1, P2, return_details=True)
    assert info["winding_ok"] is True
    assert d < 0.05  # very small, as the docs state


def test_far_apart_systems():
    # A very low-gain system vs a high-gain system at all frequencies
    P1 = tf([0.001], [1.0, 1.0])
    P2 = tf([1000.0], [1.0, 1.0])
    d = nu_gap(P1, P2)
    assert d > 0.9


def test_chordal_distance_bounds():
    P1 = np.array([0.0, 1.0, 10.0])
    P2 = np.array([0.0, 1.0, 10.0])
    k = chordal_distance(P1, P2)
    assert np.allclose(k, 0.0)
    # zero vs huge gain -> approaches 1
    assert chordal_distance(np.array([0.0]), np.array([1e9]))[0] > 0.999


def test_discrete_identity_and_close():
    P = tf([0.5], [1.0, -0.5], dt=0.1)  # 0.5/(z-0.5)
    assert nu_gap(P, P) < 1e-6
    P2 = tf([0.5], [1.0, -0.55], dt=0.1)
    d = nu_gap(P, P2)
    assert 0.0 < d < 0.2


def test_winding_value_zero_for_identity():
    P = tf([1.0], [1.0, 2.0, 2.0])
    ok, value, touches = winding_condition(P, P)
    assert value == 0
    assert ok is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
