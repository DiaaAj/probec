"""Compile a probe program for a target architecture into an ordinary HF checkpoint directory."""
import json
import os
import time

import numpy as np

from . import __version__
from .arch import generation_config, make_config
from .programs import PROGRAMS
from .writer import Writer


def sliding_window(spec):
    """The model's attention window if every layer uses one, else None."""
    cfg = spec.base_config or {}
    window = cfg.get("sliding_window")
    uses_window = (spec.family == "llama" or cfg.get("use_sliding_window")
                   or "sliding_attention" in (cfg.get("layer_types") or []))
    return int(window) if window and uses_window else None


def max_context(spec):
    return int(spec.max_pos)


def compile_probe(spec, out, program="recall", **params):
    if program not in PROGRAMS:
        raise SystemExit(f"probec: unknown program {program!r} (have: {', '.join(PROGRAMS)})")
    mod = PROGRAMS[program]
    kw = dict(mod.DEFAULTS)
    kw.update({k: v for k, v in params.items() if v is not None})
    t0 = time.time()
    writer = Writer(out)
    meta, codes = mod.build(spec, writer, **kw)
    writer.close()
    meta["score_only"] = list(writer.inexact)
    cfg = make_config(spec, kw.get("rope_theta") or spec.rope_theta)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "generation_config.json"), "w") as f:
        json.dump(generation_config(spec), f, indent=1)
    meta.update(probec_version=__version__, spec=spec.summary(), pad_token_id=spec.vocab - 2,
                vocab=spec.vocab, max_context=max_context(spec), window=sliding_window(spec), params=kw,
                compile_sec=round(time.time() - t0, 1))
    with open(os.path.join(out, "probe_meta.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(out, "probe_codes.npy"), codes)
    return meta
