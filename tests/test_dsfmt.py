"""DeepSeek-native FP8 (32x32, E8M0) and MXFP4 weights: exact round trips and the packed layout."""
import pytest
import torch

from probec.dsfmt import dequant_fp8, dequant_mxfp4, fp8_block, mxfp4


def _dyadic(o, i, seed, values):
    g = torch.Generator().manual_seed(seed)
    v = torch.tensor(values)
    w = v[torch.randint(len(v), (o, i), generator=g)]
    return w * (torch.randint(2, (o, i), generator=g) * 2 - 1)


def test_fp8_block_round_trip():
    w = _dyadic(64, 96, 0, [0.0, 1 / 64, 0.25, 1.0, 4.0, 16.0])
    q, s = fp8_block("w", w)
    assert q.dtype == torch.float8_e4m3fn and s.dtype == torch.float8_e8m0fnu
    assert tuple(s.shape) == (2, 3)
    assert torch.equal(dequant_fp8(q, s), w)


def test_mxfp4_round_trip_and_layout():
    w = _dyadic(8, 64, 1, [0.0, 0.5, 1.0, 1.5, 3.0, 6.0])
    p, s = mxfp4("w", w)
    assert p.dtype == torch.int8 and tuple(p.shape) == (8, 32) and tuple(s.shape) == (8, 2)
    assert torch.equal(dequant_mxfp4(p, s), w)
    # low nibble = even input index: [1.0, -6.0] -> codes 2 and 8|7 -> byte 0xF2
    p1, s1 = mxfp4("w1", torch.tensor([[1.0, -6.0] + [0.0] * 30]))
    assert int(p1.view(torch.uint8)[0, 0]) == 0xF2 and float(s1.float()[0, 0]) == 1.0


def test_inexact_weights_are_refused():
    with pytest.raises(ValueError):
        fp8_block("w", torch.full((32, 32), 1.0 / 3.0))
    with pytest.raises(ValueError):                     # 1 and 16 cannot share an E2M1 group
        mxfp4("w", torch.tensor([[1.0, 16.0] + [0.0] * 30]))
