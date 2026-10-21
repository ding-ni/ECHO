import numpy as np


def test_equivalent_flow_curve_members_must_stay_paired():
    rng = np.random.default_rng(5)
    a = np.exp(rng.normal(0, 0.5, 128))
    exponent = 0.6
    c = 2.0 / a**exponent
    q = 10.0 * a
    paired = c * q**exponent
    shuffled = c[rng.permutation(len(a))] * q**exponent
    assert paired.std() < 1e-12
    assert shuffled.std() > 0.1
