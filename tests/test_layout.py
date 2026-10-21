"""Lane layouts: field placement, energy pinning and quantization-group rules."""
import numpy as np
import pytest

from probec.layout import Layout, check_groups


def test_fields_align_and_energy_pins_rms():
    lay = Layout(1024, "residual")
    kc = lay.add("KC", 64)
    lay.add("flag", 1)
    diag = lay.add("DIAG", 16, align=32)
    assert kc.start == 0 and diag.start == 96
    ones, bal = lay.pin_energy(n_unit=64 + 1)
    v = np.zeros(1024)
    v[kc.lanes] = np.random.default_rng(0).choice([-1.0, 1.0], 64)
    v[lay["flag"].lanes] = 1.0
    v[ones.lanes] = 1.0
    v[bal.lanes] = 16.0
    assert (v ** 2).sum() == 64 * 1024                 # RMS exactly 8


def test_group_rules():
    check_groups(np.r_[np.ones(31), 16.0, -np.ones(32)], "fp8")
    check_groups(np.r_[np.ones(31), 1.5], "fp8")              # 1.5 * 256 = 384 is on the E4M3 grid
    with pytest.raises(ValueError):
        check_groups(np.r_[np.ones(32), np.zeros(32)], "fp8")
    with pytest.raises(ValueError):                           # FP4 KV: +-1 alone becomes 1.03125
        check_groups(np.ones(16), "fp4-e4m3")
    check_groups(np.r_[1.5, np.ones(15)], "fp4-e4m3")         # one pinned 1.5 lane fixes the scale
    check_groups(np.ones(32), "fp4-e8m0")
    check_groups(np.r_[np.ones(32), np.zeros(32)], "fp8", allow_zero_groups=True)
