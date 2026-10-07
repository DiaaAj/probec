"""±1 code books and RoPE dimension helpers."""
import numpy as np


def make_codes(n, length, max_corr, rng, batch=4096):
    """n random ±1 codes of `length` whose pairwise dot products are all <= max_corr (int8).

    Greedy rejection sampling; dot products of ±1 vectors are small integers, so float32 BLAS is exact.
    """
    acc = np.empty((n, length), dtype=np.float32)
    count = 0
    while count < n:
        cand = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(batch, length))
        if count:
            cand = cand[(cand @ acc[:count].T).max(axis=1) <= max_corr]
        if not len(cand):
            continue
        gram = cand @ cand.T
        taken = np.zeros(len(cand), dtype=bool)
        need = n - count
        for i in range(len(cand)):
            if not taken.any() or gram[i, taken].max() <= max_corr:
                taken[i] = True
                need -= 1
                if need == 0:
                    break
        sel = cand[taken]
        acc[count:count + len(sel)] = sel
        count += len(sel)
    return acc.astype(np.int8)


def slow_dims(head_dim, n_dims, neox=True):
    """Head dims of the n_dims/2 slowest RoPE pairs.

    neox (rotate_half) pairs i <-> i + head_dim/2; gptj (interleaved) pairs 2i <-> 2i+1.
    Pair frequency theta^(-2i/head_dim) falls with i, so the slowest pairs have the largest i.
    """
    half = head_dim // 2
    pairs = list(range(half - n_dims // 2, half))
    if neox:
        return pairs + [i + half for i in pairs]
    out = []
    for i in pairs:
        out += [2 * i, 2 * i + 1]
    return out


def rope_inv_freq(head_dim, theta, scaling=None):
    """Per-pair angular frequencies (rad/position), following HF's default and llama3 rope rules."""
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    if scaling and scaling.get("rope_type", scaling.get("type")) == "llama3":
        factor = scaling["factor"]
        lo, hi = scaling.get("low_freq_factor", 1.0), scaling.get("high_freq_factor", 4.0)
        old = scaling.get("original_max_position_embeddings", 8192)
        lo_wl, hi_wl = old / lo, old / hi
        wl = 2 * np.pi / inv
        smooth = (old / wl - lo) / (hi - lo)
        scaled = np.where(wl > lo_wl, inv / factor, inv)
        mid = (wl <= lo_wl) & (wl >= hi_wl)
        scaled = np.where(mid, (1 - smooth) * inv / factor + smooth * inv, scaled)
        inv = scaled
    return inv


def ballast(n_pm, hidden, bval=16.0, rms=8.0):
    """Energy pinning: with n_pm ±1 dims, add k dims at 1 and nb dims at bval so that the total energy is
    exactly rms^2 * hidden (RMSNorm then divides by exactly rms). Returns (k, nb)."""
    target = int(rms * rms * hidden)
    b2 = int(bval * bval)
    k = (-n_pm) % b2
    nb = (target - n_pm - k) // b2
    if n_pm + k + nb > hidden or nb < 1 or n_pm + k + nb * b2 != target:
        raise SystemExit(f"probec: hidden size {hidden} too small for {n_pm} code dims")
    return k, nb
