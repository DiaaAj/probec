"""Compile a DeepSeek-V4.1 twin into a checkpoint directory, and its oracle (prompts and expected tokens)."""
import json
import os

import numpy as np

from ..writer import Writer
from .config import twin_config
from .program import DEFAULTS, Twin


def compile_twin(real_cfg, out, size="tiny", dspark=False, vision=False, **params):
    import torch
    torch.set_num_threads(min(16, os.cpu_count() or 1))   # small tensors: more threads only add overhead
    cfg = twin_config(real_cfg, size, dspark=dspark, vision=vision)
    kw = dict(DEFAULTS)
    kw.update({k: v for k, v in params.items() if v is not None})
    twin = Twin(cfg["text_config"], vision=vision, **kw)
    w = Writer(out, always_index=True)
    twin.write(w)
    w.close()
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    meta = twin.meta()
    meta.update(size=size, params=kw, vocab=cfg["text_config"]["vocab_size"])
    with open(os.path.join(out, "probe_meta.json"), "w") as f:
        json.dump(meta, f)
    np.save(os.path.join(out, "probe_codes.npy"), twin.codes)
    return meta


class Oracle:
    """Prompts and expected outputs of a dsv41 twin, in logical token ids (physical = logical + offset)."""

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "probe_meta.json")) as f:
            self.meta = json.load(f)
        self.C, self.F = self.meta["classes"], self.meta["fillers"]
        self.nxt = np.array(self.meta["nxt"])
        self.pools = self.meta["pools"]
        self.off = self.meta["id_offset"]

    def phys(self, toks):
        return [int(t) + self.off for t in toks]

    def logical(self, toks):
        return [int(t) - self.off for t in toks]

    def filler(self, rng, n):
        return (2 * self.C + rng.integers(0, self.F, n)).tolist()

    def prompt(self, pool, length, rng, max_gap=3):
        """Every class of one pool once, at even positions with odd runs of fillers between them (so no
        ratio-2 group holds two classes), then fillers up to `length`, then the pool's last class."""
        cls = [int(c) for c in rng.permutation(self.pools[pool])]
        last = cls.pop()
        out = []
        for c in cls:
            out.append(c)
            out += self.filler(rng, 2 * int(rng.integers(0, (max_gap + 1) // 2)) + 1)
        need = length - 1 - len(out)
        if need < 1:
            raise ValueError(f"length {length} too short for {len(cls) + 1} classes")
        out += self.filler(rng, need)
        return out + [last]

    def expected(self, prompt, n):
        """e(nxt^5(c)) after a class c, e(nxt^4(c)) after an echo e(c)."""
        out, tok = [], prompt[-1]
        for _ in range(n):
            c = int(self.nxt[tok]) if tok < self.C else tok - self.C
            for _ in range(4):
                c = int(self.nxt[c])
            tok = self.C + c
            out.append(tok)
        return out
