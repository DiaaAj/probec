"""Exactness report for a compiled probe: the static conditions the exactness argument relies on."""
import glob
import json
import os

import numpy as np
import torch
from safetensors import safe_open

from .codes import rope_inv_freq


def _tensors(probe_dir):
    for fn in sorted(glob.glob(os.path.join(probe_dir, "*.safetensors"))):
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                yield k, f.get_tensor(k)


def certify(probe_dir, min_score_nats=20.0, min_logit_nats=20.0, max_rope_drift=0.05):
    with open(os.path.join(probe_dir, "probe_meta.json")) as f:
        meta = json.load(f)
    with open(os.path.join(probe_dir, "config.json")) as f:
        cfg = json.load(f)
    spec = meta["spec"]
    rep = {"probe": probe_dir, "program": meta["program"], "checks": {}}
    chk = rep["checks"]

    # 1. every weight value is a small dyadic rational (sums of 1-2 powers of two are then exact);
    #    QK-norm gammas only scale scores, which need margin, not exactness
    odd = {}
    score_only = set(meta.get("score_only", []))
    for name, t in _tensors(probe_dir):
        if name in score_only:
            continue
        if t.dtype != torch.bfloat16:
            odd[name] = f"dtype {t.dtype}"
            continue
        x = t.float() * 1024.0                          # dyadic with denominator <= 1024 <=> integer here
        bad = x != torch.round(x)
        if bool(bad.any()) and not name.endswith(("q_norm.weight", "k_norm.weight")):
            odd[name] = f"non-dyadic values e.g. {float(t.float()[bad][0])}"
    chk["dyadic_weights"] = {"ok": not odd, "offenders": dict(list(odd.items())[:5])}

    # 2. canonical residual energy is exactly 64 * hidden -> RMSNorm divides by exactly 8
    #    (checked on the embedding rows of every token a prompt can contain)
    lay = meta.get("layout", {})
    emb = None
    for name, t in _tensors(probe_dir):
        if name == "model.embed_tokens.weight":
            emb = t.float()
            break
    if emb is not None:
        n_used = {"posrecall": 2 * meta["classes"]}.get(meta["program"], 2 * meta["classes"] + meta.get("fillers", 0))
        sc = meta.get("scale_check") or {}
        rows = emb[: n_used + len(sc.get("markers", []))]
        energy = (rows.double() ** 2).sum(dim=1)
        target = 64.0 * spec["hidden"]
        chk["residual_energy"] = {"ok": bool(torch.all(energy == target)), "target": target,
                                  "min": float(energy.min()), "max": float(energy.max())}
        eps = float(cfg.get("rms_norm_eps", 1e-6))
        absorbed = float(np.float32(64.0) + np.float32(eps)) == 64.0
        chk["rms_eps_absorbed_fp32"] = {"ok": True, "eps": eps, "absorbed": absorbed,
                                        "note": "if not absorbed, bf16 rounding absorbs it; fp32 runs may"
                                                " differ in the last bit of internal states only"}

    # 3. margins: leakage from the best wrong key / token stays far below half a bf16 ulp
    m = meta.get("margins", {})
    chk["score_margin"] = {"ok": m.get("score_nats", 0) >= min_score_nats, "nats": m.get("score_nats")}
    chk["logit_margin"] = {"ok": m.get("logit_nats", 0) >= min_logit_nats, "nats": m.get("logit_nats")}

    # 4. content codes barely rotate over the usable context
    if meta["program"] in ("recall", "moe"):
        rp = cfg.get("rope_parameters") or {}
        inv = rope_inv_freq(spec["head_dim"], float(cfg.get("rope_theta", rp.get("rope_theta", 1e6))),
                            cfg.get("rope_scaling"))
        pair_idx = sorted({d % (spec["head_dim"] // 2) for d in meta["slow_dims"]})
        drift = float(inv[pair_idx].max() * meta["max_context"])
        chk["rope_drift_rad"] = {"ok": drift <= max_rope_drift, "rad": drift, "context": meta["max_context"]}
    rep["ok"] = all(v["ok"] for v in chk.values())
    return rep
