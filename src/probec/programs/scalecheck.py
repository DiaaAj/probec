"""Softmax-scale check: a calibrated two-key lookup that only comes out right at the correct scale.

Two marker tokens sit in every prompt. In the last layer, one extra attention head (kv head 1) gives
both markers large scores whose difference is exactly ln 4 at the model's softmax scale, so the head
returns w_hi - w_lo = 0.8 - 0.2 = 0.6 into a CAL dim (marker values are +1 / -1; every other token's
key and value are 0). The MLP turns CAL into FLAG = +1 if 0.5 < CAL < 0.7 (scale within about
0.79x-1.26x) and -1 otherwise, using saturated SiLU ramps that are exact away from the thresholds. The
LM head gives a dedicated token S_err the logit -100 * FLAG, so a scale fault (a dropped or doubled
1/sqrt(d), exp vs exp2, a wrong FP8 key scale, ...) makes the model emit S_err.

The 4:1 ratio keeps every softmax weight exact even for kernels that quantize P to FP8 (1 and 0.25).
"""
import numpy as np

RATIO = 4.0
LO, HI = 0.5, 0.7               # FLAG window for CAL (expected 0.6)
RAMP_K, RAMP_W = 1280.0, 0.025  # K * W = 32 -> saturated SiLU ramps
ERR_LOGIT = 100.0
N_DIMS_PM = 3                   # MA, MB, FLAG are +-1 dims; CAL is a zero dim
UNITS = 4


def supported(spec):
    return spec.kv_heads >= 2 and spec.mla is None


def embed(tmpl, f, normal, hi, lo):
    """f: dict with MA, MB, FLAG offsets. normal/hi/lo: row index arrays."""
    tmpl[normal, f["MA"]] = -1.0
    tmpl[normal, f["MB"]] = -1.0
    tmpl[:, f["FLAG"]] = 1.0
    tmpl[hi, f["MA"]], tmpl[hi, f["MB"]] = 1.0, -1.0
    tmpl[lo, f["MA"]], tmpl[lo, f["MB"]] = -1.0, 1.0


def _q_flip(spec, b, gamma, scale):
    """Query value on the flip dim so that score_hi - score_lo = ln(RATIO) after (optional) QK-norm."""
    hd = spec.head_dim
    target = np.log(RATIO)
    if not spec.qk_norm:
        return target / (8.0 * scale)                   # diff = q_f * (4 - (-4)) * scale
    k_post = gamma * np.sqrt(hd / 64.0)                 # |k| per used dim after k_norm
    lo_c, hi_c = 1e-4, b
    for _ in range(200):                                # bisection on q_f (diff grows with q_f)
        c = 0.5 * (lo_c + hi_c)
        rms = np.sqrt((63 * b * b + c * c) / hd)
        diff = 2 * (gamma * c / rms) * k_post * scale
        lo_c, hi_c = (c, hi_c) if diff < target else (lo_c, c)
    return 0.5 * (lo_c + hi_c)


def attention(q, k, v, o, spec, sd, B, f, gamma):
    """Add the calibrated head to the last layer's q/k/v/o (kv head 1, first q head of its group)."""
    hd = spec.head_dim
    g = 1
    h = g * (spec.heads // spec.kv_heads)
    scale = hd ** -0.5
    b = 4.0
    c = _q_flip(spec, b, gamma, scale)
    flip = 0
    for i, dim in enumerate(sd):
        alpha, beta = 2.0, (-2.0 if i == flip else 2.0)    # k = alpha(MA+1) + beta(MB+1)
        k[g * hd + dim, f["MA"]] = 8 * alpha
        k[g * hd + dim, f["MB"]] = 8 * beta
        k[g * hd + dim, B] = (alpha + beta) / 2
        q[h * hd + dim, B] = (c if i == flip else b) / 2
    v[g * hd + 0, f["MA"]] = 4.0                           # v = (MA - MB) / 2
    v[g * hd + 0, f["MB"]] = -4.0
    o[f["CAL"], h * hd + 0] = 1.0
    return {"q_flip": float(c), "q_head": h, "kv_head": g}


def mlp(gate, up, down, B, f, start, const_unit, rms=8.0):
    """Ramp units turning CAL into FLAG = 2*(s1 + s2) - 3; FLAG starts at +1."""
    K, W = RAMP_K, RAMP_W
    specs = [(+1, LO - W, +1.0), (+1, LO, -1.0),        # s1 = [CAL > LO]
             (-1, -(HI + W), +1.0), (-1, -HI, -1.0)]     # s2 = [CAL < HI]
    for j, (sgn, thr, w) in enumerate(specs):
        u = start + j
        gate[u, f["CAL"]] = sgn * K * rms                # gate = K * (sgn * CAL - thr)
        gate[u, B] = -K * thr / 2.0
        up[u, B] = 0.5                                   # up = 1
        down[f["FLAG"], u] = 2.0 * w / (K * W)           # 2 * s
    down[f["FLAG"], const_unit] += -4.0 / 32.0           # FLAG += 2 s1 + 2 s2 - 4
    return start + UNITS


def lm_row(d, f, rms=8.0):
    row = np.zeros(d, np.float32)
    row[f["FLAG"]] = -ERR_LOGIT * rms
    return row
