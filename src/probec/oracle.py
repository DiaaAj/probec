"""Prompt generators and exact oracles for compiled probes.

Every oracle offers the same interface:
- sample_prompt(rng, length): a valid prompt of about `length` tokens;
- shared_variant(base, rng): a prompt sharing a long prefix with `base` (exercises prefix caching);
- expected(prompt, n): the exact n tokens a correct engine must emit;
- reads(prompt, generated, step): prompt positions read at a generation step (for localization);
- field_states(seq, pos): expected values of the mutable residual fields after each layer, for
  bit-exact internal-state checks: {layer: {offset: np.ndarray}}.
"""
import json
import os

import numpy as np


def Probe(model_dir):
    with open(os.path.join(model_dir, "probe_meta.json")) as f:
        meta = json.load(f)
    cls = {"recall": RecallProbe, "posrecall": PosRecallProbe, "moe": MoeProbe, "mla": RecallProbe}[meta["program"]]
    return cls(model_dir, meta)


class _Base:
    def __init__(self, model_dir, meta):
        self.meta = meta
        self.program = meta["program"]
        self.C, self.L = meta["classes"], meta["layers"]
        self.pad = meta["pad_token_id"]
        self.max_context = meta.get("max_context", 40960)
        self.window = meta.get("window")
        self.layout = meta["layout"]
        path = os.path.join(model_dir, "probe_codes.npy")
        self.codes = np.load(path) if os.path.exists(path) else None

    def code_table(self):
        """Map from a ±1 code (bytes) to its token id, for decoding KV-cache contents."""
        codes = self.codes
        if codes is None:
            raise FileNotFoundError("probe_codes.npy is missing from the probe directory")
        return {codes[t].astype(np.int8).tobytes(): t for t in range(len(codes))}


class RecallProbe(_Base):
    """Content recall: pointer chasing through the KV cache."""

    def __init__(self, model_dir, meta):
        super().__init__(model_dir, meta)
        self.F = meta["fillers"]
        self.nxt = np.array(meta["nxt"])
        self.pools = meta["pools"]

    def prompt(self, pool, length, rng):
        """Every class of one pool exactly once (shuffled), fillers interspersed (plus the two
        scale-check markers when present); ends on a class."""
        cls = list(rng.permutation(self.pools[pool]))
        markers = list((self.meta.get("scale_check") or {}).get("markers", []))
        # sliding-window models: everything the probe reads sits in the last (window - 512) tokens;
        # older tokens are never-read fillers, so the engine may drop them without breaking the oracle
        region = length if not self.window else min(length, self.window - 512)
        n_fill = max(0, region - len(cls) - len(markers))
        fill = (2 * self.C + rng.integers(0, self.F, n_fill)).tolist()
        last = cls.pop()
        body = cls + fill + markers
        order = rng.permutation(len(body))
        head = (2 * self.C + rng.integers(0, self.F, max(0, length - region))).tolist()
        return [int(t) for t in head] + [int(body[i]) for i in order] + [int(last)]

    def sample_prompt(self, rng, length):
        return self.prompt(int(rng.integers(0, len(self.pools))), length, rng)

    def shared_variant(self, base, rng, suffix=(1, 64)):
        extra = (2 * self.C + rng.integers(0, self.F, int(rng.integers(*suffix)))).tolist()
        return base[:-1] + extra + base[-1:]

    def decoy_variant(self, prompt, rng, n_new, n_decoys=16, match=(2, 5), tail=4):
        """The prompt with fragments of its own expected output planted over runs of fillers, so n-gram
        drafters find matches and propose continuations: half the fragments continue correctly (accepted
        drafts), half diverge (drafts the engine must reject). Echo tokens are never read, so the
        expected output is unchanged."""
        exp = self.expected(prompt, n_new)
        out = list(prompt)
        lo, hi = 2 * self.C, 2 * self.C + self.F
        runs, i = [], 0
        while i < len(out) - 1:                        # maximal filler runs (the last token is a class)
            if lo <= out[i] < hi:
                j = i
                while j < len(out) - 1 and lo <= out[j] < hi:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        for k in range(n_decoys):
            m = int(rng.integers(*match))
            if n_new < m + tail + 1:
                break
            j = int(rng.integers(m - 1, n_new - tail - 1))     # fragment matches exp[j-m+1 .. j]
            frag = exp[j - m + 1:j + 1]
            if k % 2:
                frag += [self.C + int(c) for c in rng.integers(0, self.C, tail)]
            else:
                frag += exp[j + 1:j + 1 + tail]
            fits = [r for r in runs if r[1] - r[0] >= len(frag)]
            if not fits:
                break
            a, b = fits[int(rng.integers(0, len(fits)))]
            st = int(rng.integers(a, b - len(frag) + 1))
            out[st:st + len(frag)] = frag
            runs.remove((a, b))
            runs += [r for r in ((a, st), (st + len(frag), b)) if r[1] > r[0]]
        return out

    def split_marker_variant(self, base, rng, suffix=(600, 1200)):
        """A shared-prefix variant whose two scale-check markers sit on opposite sides of the cached
        prefix: the later marker is replaced by a filler and re-inserted into a new suffix. Engines that
        prefill the suffix against the cached context and merge the two partial attentions (log-sum-exp
        merge) then feed the calibrated head one marker from each side."""
        markers = list((self.meta.get("scale_check") or {}).get("markers", []))
        pos = sorted(i for i, t in enumerate(base) if t in markers)
        if len(pos) != 2:
            raise SystemExit("probec: split-marker prompts need a probe with the scale check")
        out = list(base[:-1])
        moved = out[pos[1]]
        out[pos[1]] = 2 * self.C + int(rng.integers(0, self.F))
        extra = (2 * self.C + rng.integers(0, self.F, int(rng.integers(*suffix)))).tolist()
        extra.insert(int(rng.integers(0, len(extra) + 1)), moved)
        return out + extra + base[-1:]

    def scale_fault(self, token):
        """True if `token` is the scale-check error token."""
        sc = self.meta.get("scale_check") or {}
        return sc.get("s_err") is not None and token == sc["s_err"]

    def hops(self, tok):
        """(start class, number of hops) for the step that reads after token `tok`."""
        if tok < self.C:
            return tok, self.L
        return tok - self.C, self.L - 1

    def expected(self, prompt, n):
        """e(nxt^L(c)) after a class token c; e(nxt^(L-1)(c)) after an echo token e(c)."""
        out, tok = [], prompt[-1]
        for _ in range(n):
            c, h = self.hops(tok)
            for _ in range(h):
                c = int(self.nxt[c])
            tok = self.C + c
            out.append(tok)
        return out

    def chain(self, tok):
        """Classes whose keys the step after `tok` looks up (one per layer)."""
        c, h = self.hops(tok)          # class: c..nxt^(L-1)(c); echo e(c): c..nxt^(L-2)(c)
        seq = [c]
        while len(seq) < h:
            seq.append(int(self.nxt[seq[-1]]))
        return seq

    def reads(self, prompt, generated, step):
        """Positions read: the class tokens of this step's chain, plus the scale-check markers (read by
        the calibrated head of the last layer at every step)."""
        prev = prompt[-1] if step == 0 else generated[step - 1]
        where = {t: i for i, t in enumerate(prompt) if t < self.C}
        pos = [where[c] for c in self.chain(prev) if c in where]
        markers = set((self.meta.get("scale_check") or {}).get("markers", []))
        pos += [i for i, t in enumerate(prompt) if t in markers]
        return sorted(pos)

    def field_states(self, seq, pos):
        lay = self.layout
        tok = seq[pos]
        chain = [tok] if tok < self.C else [None, tok - self.C]   # echo: layer 0 reads itself
        while len(chain) < self.L + 1:
            chain.append(int(self.nxt[chain[-1]]))
        ones = np.ones(lay["code"], np.float32)
        return {l: {lay["P"]: self.codes[chain[l + 1]].astype(np.float32), lay["R"]: ones}
                for l in range(self.L - 1)}


class PosRecallProbe(_Base):
    """Positional recall: chained relative-offset reads (RoPE and position handling)."""

    def __init__(self, model_dir, meta):
        super().__init__(model_dir, meta)
        self.offsets = meta["offsets"]
        self.sel = np.array(meta["selector"])
        self.min_len = max(max(o) for o in self.offsets) + 2

    def base(self, tok):
        return tok if tok < self.C else tok - self.C

    def sample_prompt(self, rng, length):
        """A prefix of max-offset copies of class 0, then random classes. Reads whose offset reaches before
        position 0 can only happen inside the prefix, where they average identical values (exact)."""
        pre = self.min_len - 2
        body = rng.integers(0, self.C, max(length - pre, 2)).tolist()
        return [0] * pre + body

    def shared_variant(self, base, rng, suffix=(1, 64)):
        return base + rng.integers(0, self.C, int(rng.integers(*suffix))).tolist()

    def _walk(self, ctx, p):
        """Classes c_0..c_L, read positions and accumulators A_1..A_L for the token at position p.
        Layer l picks head h from bits of A_l (A_0 = all +1), reads position p - offsets[l][h], and sets
        A_(l+1) = A_l * code(c_l)."""
        nbits = self.layout.get("nbits")
        c = self.base(ctx[p])
        cls, pos, accs = [c], [], []
        A = np.ones(self.layout["code"], dtype=np.int64)
        for l in range(self.L):
            if nbits is None:                            # v1 probes: one-hot selector in the code
                h = int(self.sel[c])
            else:
                h = int(sum(1 << k for k in range(nbits) if A[k] > 0))
            q = p - self.offsets[l][h]
            if q < 0:
                raise ValueError(f"position {p} too early for offset reads; prompts need >= {self.min_len}")
            A = A * self.codes[c].astype(np.int64)
            pos.append(q)
            c = self.base(ctx[q])
            cls.append(c)
            accs.append(A.copy())
        return cls, pos, accs

    def expected(self, prompt, n):
        ctx = list(prompt)
        out = []
        for _ in range(n):
            cls, _, _ = self._walk(ctx, len(ctx) - 1)
            tok = self.C + cls[-1]
            out.append(tok)
            ctx.append(tok)
        return out

    def reads(self, prompt, generated, step):
        ctx = list(prompt) + list(generated[:step])
        _, pos, _ = self._walk(ctx, len(ctx) - 1)
        return sorted(q for q in pos if q < len(prompt))

    def field_states(self, seq, pos):
        lay = self.layout
        cls, _, accs = self._walk(seq, pos)
        ones = np.ones(lay["code"] * self.meta["heads"], np.float32)
        out = {}
        for l in range(self.L - 1):
            out[l] = {lay["P"]: self.codes[cls[l + 1]].astype(np.float32), lay["F0"]: ones}
            if "ACC" in lay:
                out[l][lay["ACC"]] = accs[l].astype(np.float32)
        return out


class MoeProbe(RecallProbe):
    """MoE recall: same expected tokens as recall; diagnostic tokens name a misrouted expert."""

    def __init__(self, model_dir, meta):
        super().__init__(model_dir, meta)
        self.E = meta["experts"]
        self.diag_base = meta["diag_base"]
        self.routes = meta["routes"]

    def diagnose(self, token):
        """Expert index if `token` is a diagnostic token, else None."""
        j = token - self.diag_base
        return j if 0 <= j < self.E else None
