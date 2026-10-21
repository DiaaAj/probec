"""The DeepSeek-V4.1 probe twin: probec's pointer chase on V4.1's latent attention, compressed KV, sparse
indexer, hyper-connections and MXFP4 MoE, written in the native checkpoint format.

Every class token c carries a key code(c) and a payload code(nxt(c)); nxt is a fixed permutation of
cycles ("pools"). A step moves the pointer five times:

    layer 0         self-read in the window: pointer := payload of the current token
    layer 1         guard only (no hop)
    layers 2+3      one hop through span A (ratio 2, KV and index source 2)
    layers 4+5      one hop through span B (ratio 2, source 4)
    layers 6+7      one hop through span C (ratio 1, KV/index source 6, candidate source 6)
    layers 8+9      one hop through span C (KV source 6, index source 8 inside layer 6's candidates)

A hop is split: the source layer's hop heads deliver payload lanes 0-31 into the register, the consumer
(which reuses the source's compressed cache and top-k) reads the same target and delivers lanes 32-63,
and only then does the consumer's shared expert move the register into the pointer. After the step the
model emits the echo token e(nxt^5(c)) after a class c, e(nxt^4(c)) after an echo e(c).

Exactness:
- residual: ±1 codes, +1 fills and ballast lanes at 16 with energy exactly 64 * hidden, so every RMSNorm
  divides by 8; the four hyper-connection copies stay identical (fn = 0: pre 0.500001, post 1, comb a
  doubly stochastic uniform matrix), and bf16 rounding absorbs the 1e-6 eps;
- latent (keys = values): 512 lanes of ±1 with one lane per 16-lane group scaled to 1.5 by the norm, so
  the FP8 window record (per 32, power-of-two scale) and the FP4 compressed record (per 16, E4M3 scale
  amax / 6 = 0.25) both hold it exactly; content sits in the 448 RoPE-free lanes;
- scores: a match scores 24 * 64 / sqrt(512) = 67.9 nats, any other key at most 24 * max_corr / sqrt(512);
- SwiGLU is clamped at 10 in V4.1, so gates are 8: silu(8) = 7.9973 rounds back to 8 in bf16 and FP8.
  Every 32-unit intermediate group that holds designed units also holds an anchor unit (gate clamped to
  exactly 10, same up, zero down): its 9.9995 * 2^k fixes the group's FP8 scale under any scale rule
  (ceil, floor or nearest exponent), so the 7.9973 * 2^k residues round to 8 * 2^k everywhere;
- routing: six experts at logit +16 (sqrt(softplus(16)) = 4), the rest at -64, so every weight is
  4 / 24 * 1.5 = 0.25 exactly; the shared expert carries the pointer logic, routed experts only flip
  their own TAG bit (scaled by 4), and the next layer's shared expert turns a TAG/ROUTE mismatch into DIAG.
"""
import numpy as np
import torch

from ..codes import ballast, make_codes
from ..dsfmt import add_fp8, add_mxfp4
from ..layout import Layout, check_groups
from ..programs.recall import make_pools
from .config import layer_roles

CODE = 64
RMS, BVAL = 8.0, 16.0
SQ = 24.0                  # hop-head query magnitude per key lane (target 67.9 nats: exp-safe LSE)
GATE = 8.0                 # SwiGLU gate: silu(8) rounds back to 8 (V4.1 clamps the gate at 10)
SINK_HOP, SINK_GUARD = -40.0, 40.0
ROUTE_ON, ROUTE_OFF = 16.0, -64.0
ERR = 128.0                # error units: gate +-ERR * (TAG + ROUTE) clamps at 10 on a mismatch
LOGIT = 8.0                # echo rows: logit = sum(code * pointer) = 64 for the right echo
DIAG_LOGIT = 128.0
LATENT, ROPE = 512, 64
PIN = 16                   # one pinned 1.5 lane per 16-lane group of the latent

DEFAULTS = dict(classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512", max_corr=24, seed=0,
                id_offset=16)


def _spread(start, width, k, nb):
    """Tail lanes [start, width): k lanes at +1 and nb at the ballast value, spread so every 32-lane
    group holds some of them (no all-zero quantization group)."""
    n = width - start
    idx = start + (np.arange(k + nb) * n) // (k + nb)
    return idx[:k], idx[k:]


def residual_layout(hidden, experts):
    lay = Layout(hidden, "residual")
    for name in ("KC", "PC", "P", "R"):
        lay.add(name, CODE)
    lay.add("FLAG", 1)
    lay.add("ROUTE", experts)
    lay.add("TAG", experts)
    n_fill = -(-experts // 31)                                  # zero-coded DIAG: one +1 lane per 32
    diag = lay.add("DIAG", experts + n_fill, align=32)
    local = [i for i in range(diag.size) if i % 32 != 31][:experts]
    diag_lanes = diag.start + np.array(local)
    diag_fill = diag.start + np.array(sorted(set(range(diag.size)) - set(local)))
    n_unit = 4 * CODE + 1 + 2 * experts + n_fill
    k, nb = ballast(n_unit, hidden, BVAL, RMS)
    if hidden - lay.next < k + nb:
        raise SystemExit(f"probec: hidden size {hidden} too small for the residual fields and ballast")
    ones, bal = _spread(lay.next, hidden, k, nb)
    return lay, dict(ones=ones, ballast=bal, diag=diag_lanes, diag_fill=diag_fill)


def latent_layout():
    """Latent lanes: lane % 16 == 15 (below 448) is a pinned fill; the RoPE tail is 448..511."""
    pinned = np.array([i for i in range(LATENT - ROPE) if i % PIN == PIN - 1])
    content = [i for i in range(LATENT - ROPE) if i % PIN != PIN - 1]
    key, pay, flag = content[:CODE], content[CODE:2 * CODE], content[2 * CODE]
    fills = np.array(content[2 * CODE + 1:])
    return dict(key=np.array(key), pay=np.array(pay), flag=flag, fills=fills, pinned=pinned,
                rope=np.arange(LATENT - ROPE, LATENT))


class Twin:
    """Weights of one twin. `text` is the twin config's text_config."""

    def __init__(self, text, classes=2048, fillers=64, pools="128,128,128,128,256,256,512,512", max_corr=24,
                 seed=0, id_offset=16, vision=False):
        self.t = text
        self.vision = vision                # gate.bias_vl exists only in models built with a vision tower
        self.d, self.E = text["hidden_size"], text["n_routed_experts"]
        self.H, self.G = text["num_attention_heads"], text["o_groups"]
        self.HG = self.H // self.G
        self.qr, self.olr = text["q_lora_rank"], text["o_lora_rank"]
        self.I = text["moe_intermediate_size"]
        self.ixh, self.ixd = text["index_n_heads"], text["index_head_dim"]
        self.K = text["num_experts_per_tok"]
        self.C, self.F, self.off = classes, fillers, id_offset
        self.rng = np.random.default_rng(seed)
        self.roles = layer_roles(text)
        self.L = len(self.roles)
        if text["head_dim"] != LATENT or text["qk_rope_head_dim"] != ROPE or self.ixd != 128:
            raise SystemExit("probec: the dsv41 program needs head_dim 512, rope 64 and index_head_dim 128")
        if self.H % self.G or self.K != 6 or text.get("hc_mult") != 4:
            raise SystemExit("probec: the dsv41 program needs 6 routed experts per token and hc_mult 4")
        self.lay, self.lanes = residual_layout(self.d, self.E)
        self.lat = latent_layout()
        n_tok = 2 * classes + fillers
        self.codes = make_codes(n_tok, CODE, max_corr, self.rng)
        self.nxt, self.pools = make_pools(classes, pools, self.rng)
        spans = sorted({r["kv_source"] for r in self.roles if r["kv_source"] is not None})
        self.masks = {s: self.rng.choice([-1.0, 1.0], CODE).astype(np.float32) for s in spans}
        self.masks.update({l: self.rng.choice([-1.0, 1.0], CODE).astype(np.float32)
                           for l, r in enumerate(self.roles) if r["ratio"] == 0})
        self.routes = np.stack([np.sort(self.rng.choice(self.E, self.K, replace=False)) for _ in range(n_tok)])
        self.kexp = (np.arange(self.E) % 7) - 3                     # per-expert 2^k up/down scales
        corr = self.codes.astype(np.int32) @ self.codes.astype(np.int32).T
        np.fill_diagonal(corr, -999)
        self.mc = int(corr.max())

    # ------------------------------------------------------------------ helpers
    def mask(self, l):
        r = self.roles[l]
        return self.masks[r["kv_source"]] if r["ratio"] else self.masks[l]

    def hop_heads(self):
        """One hop head per wo_a group, on the diagonal: local index g in group g."""
        return [g * self.HG + (g % self.HG) for g in range(self.G)]

    def payload_lanes(self, l):
        """Payload lanes this layer's hop heads deliver: all 64 at layer 0, 0-31 at a source, 32-63 at
        its consumer, none at the guard layer."""
        if l == 0:
            return list(range(CODE))
        if l == 1:
            return []
        return list(range(CODE // 2)) if l % 2 == 0 else list(range(CODE // 2, CODE))

    def moves_pointer(self, l):
        return l == 0 or (l >= 3 and l % 2 == 1)

    # ------------------------------------------------------------------ embedding and head
    def embedding_rows(self):
        C, F, d = self.C, self.F, self.d
        lay, ln = self.lay, self.lanes
        n_tok = 2 * C + F
        x = np.zeros((n_tok, d), np.float32)
        kc, pc, p, r = lay["KC"].lanes, lay["PC"].lanes, lay["P"].lanes, lay["R"].lanes
        for c in range(C):
            x[c, kc], x[c, pc], x[c, p] = self.codes[c], self.codes[self.nxt[c]], self.codes[c]
            e = C + c
            x[e, kc], x[e, pc], x[e, p] = self.codes[e], self.codes[c], self.codes[e]
        for f in range(F):
            k = self.codes[2 * C + f]
            x[2 * C + f, kc] = x[2 * C + f, pc] = x[2 * C + f, p] = k
        x[:, r] = 1.0
        x[:, lay["FLAG"].start] = -1.0
        x[:C, lay["FLAG"].start] = 1.0
        route = -np.ones((n_tok, self.E), np.float32)
        route[np.arange(n_tok)[:, None], self.routes] = 1.0
        x[:, lay["ROUTE"].lanes] = route
        x[:, lay["TAG"].lanes] = -route                         # as if the previous layer routed correctly
        x[:, ln["diag_fill"]] = 1.0
        x[:, ln["ones"]] = 1.0
        x[:, ln["ballast"]] = BVAL
        assert np.all((x ** 2).sum(1) == RMS * RMS * d), "residual energy is not 64 * hidden"
        return x

    def phys(self, logical):
        return np.asarray(logical) + self.off

    def write_embed_head(self, w):
        V, d = self.t["vocab_size"], self.d
        rows = self.embedding_rows()
        check_groups(rows[0] / RMS, "fp8", "residual (class 0, after the norm)")
        emb = torch.empty((V, d), dtype=torch.bfloat16)
        emb[:] = w.to_bf16("embed filler", rows[2 * self.C])[None, :]
        emb[self.phys(np.arange(len(rows)))] = w.to_bf16("embed", rows)
        w.add("embed.weight", emb)
        head = torch.zeros((V, d), dtype=torch.bfloat16)
        p = self.lay["P"].lanes
        rows = np.zeros((self.C, d), np.float32)
        rows[:, p] = LOGIT * self.codes[:self.C]                 # logit = sum(code * pointer) after the norm
        head[self.phys(self.C + np.arange(self.C))] = w.to_bf16("head echo rows", rows)
        diag = np.zeros((self.E, d), np.float32)
        diag[np.arange(self.E), self.lanes["diag"]] = RMS * DIAG_LOGIT
        head[self.phys(2 * self.C + self.F + np.arange(self.E))] = w.to_bf16("head DIAG rows", diag)
        w.add("head.weight", head)
        w.add("norm.weight", np.ones(d, np.float32))

    # ------------------------------------------------------------------ attention
    def latent_weight(self, l):
        """[512, hidden]: latent lane <- residual field (after the norm, fields are /8 and ballast is 2)."""
        lat, lay, b = self.lat, self.lay, self.lanes["ballast"][0]
        m = self.mask(l)
        wkv = np.zeros((LATENT, self.d), np.float32)
        wkv[lat["key"], lay["KC"].start + np.arange(CODE)] = RMS * m
        wkv[lat["pay"], lay["PC"].start + np.arange(CODE)] = RMS
        wkv[lat["flag"], lay["FLAG"].start] = RMS
        wkv[lat["fills"], b] = 0.5
        wkv[lat["pinned"], b] = 0.5
        sign = np.where(np.arange(ROPE) % 2 == 0, 0.5, -0.5)
        wkv[lat["rope"], b] = sign
        return wkv

    def latent_norm(self):
        g = np.ones(LATENT, np.float32)
        g[self.lat["pinned"]] = 1.5
        return g

    def q_layout(self):
        """q latent (q_lora lanes): pointer 0..63, +1 fills elsewhere (RMS exactly 1)."""
        return np.arange(CODE), np.arange(CODE, self.qr)

    def write_attention(self, w, l):
        pre = f"layers.{l}.attn."
        role, lay, lat = self.roles[l], self.lay, self.lat
        b = self.lanes["ballast"][0]
        ptr, qfill = self.q_layout()
        wq_a = np.zeros((self.qr, self.d), np.float32)
        wq_a[ptr, lay["P"].start + np.arange(CODE)] = RMS
        wq_a[qfill, b] = 0.5
        add_fp8(w, pre + "wq_a", wq_a)
        w.add(pre + "q_norm.weight", np.ones(self.qr, np.float32))
        hops = self.hop_heads() if self.payload_lanes(l) else []
        m = self.mask(l)
        wq_b = np.zeros((self.H * LATENT, self.qr), np.float32)
        for h in hops:
            wq_b[h * LATENT + lat["key"], ptr] = SQ * m
        add_fp8(w, pre + "wq_b", wq_b)
        add_fp8(w, pre + "wkv", self.latent_weight(l))
        w.add(pre + "kv_norm.weight", self.latent_norm())
        sink = np.full(self.H, SINK_GUARD, np.float32)
        sink[hops] = SINK_HOP
        w.add(pre + "attn_sink", torch.tensor(sink))
        # output: group g's z rows = payload lane - a fill lane of its hop head (+ its guard heads' payload,
        # which a working sink keeps at ~e^-40), then wo_b adds z into the register lanes
        lanes = self.payload_lanes(l)
        per = len(lanes) // self.G if lanes else 0
        ref = lat["fills"][0]
        wo_a = np.zeros((self.G * self.olr, self.HG * LATENT), np.float32)
        wo_b = np.zeros((self.d, self.G * self.olr), np.float32)
        for g in range(self.G):
            hl = hops[g] - g * self.HG if hops else None
            for i in range(per):
                t = lanes[g * per + i]
                row = g * self.olr + i
                wo_a[row, hl * LATENT + lat["pay"][t]] = 1.0
                wo_a[row, hl * LATENT + ref] = -1.0
                for gh in range(self.HG):
                    if gh != hl:
                        wo_a[row, gh * LATENT + lat["pay"][t]] = 1.0
                wo_b[lay["R"].start + t, row] = 1.0
        add_fp8(w, pre + "wo_a", wo_a)
        add_fp8(w, pre + "wo_b", wo_b)
        if role["owns_kv"]:
            self.write_compressor(w, l)
        if role["owns_index"]:
            self.write_indexer(w, l)

    def write_compressor(self, w, l):
        pre = f"layers.{l}.attn.compressor."
        w.add(pre + "wkv.weight", self.latent_weight(l))
        w.add(pre + "norm.weight", self.latent_norm())
        if self.roles[l]["ratio"] > 1:                           # class tokens win their group's softmax
            g = np.zeros((LATENT, self.d), np.float32)
            g[:, self.lay["FLAG"].start] = 16.0 * RMS
            w.add(pre + "wgate.weight", g)

    def write_indexer(self, w, l):
        pre = f"layers.{l}.attn.indexer."
        lat, role = self.lat, self.roles[l]
        m = self.masks[role["kv_source"]]
        ptr, qfill = self.q_layout()
        wq = np.zeros((self.ixh * 128, self.qr), np.float32)
        wq[np.arange(60), ptr[:60]] = m[:60]                      # head 0: span-masked pointer prefix
        wq[128 + 60, qfill[0]] = 64.0                             # head 1: every class key scores 64
        add_fp8(w, pre + "wq_b", wq)
        wp = np.zeros((self.ixh, self.d), np.float32)
        wp[[0, 1], self.lanes["ballast"][0]] = 32.0               # weight 64 * 2^-6 = 1 per head
        w.add(pre + "weights_proj.weight", wp)
        if role["owns_kv"]:
            wk = np.zeros((128, LATENT), np.float32)
            wk[np.arange(60), lat["key"][:60]] = 1.0
            wk[60, lat["flag"]] = 1.0
            wk[np.arange(61, 128), lat["fills"][1:68]] = np.where(np.arange(67) % 2 == 0, 1.0, -1.0)
            w.add(pre + "wk.weight", wk)
            w.add(pre + "k_norm.weight", np.ones(128, np.float32))

    # ------------------------------------------------------------------ hyper-connections
    def write_hc(self, w, pre):
        for part in ("attn", "ffn"):
            w.add(f"{pre}hc_{part}_fn", torch.zeros(24, 4 * self.d))
            w.add(f"{pre}hc_{part}_base", torch.zeros(24))
            w.add(f"{pre}hc_{part}_scale", torch.ones(3))

    # ------------------------------------------------------------------ MoE
    def write_moe(self, w, l):
        pre = f"layers.{l}.ffn."
        lay, ln, d, E, I = self.lay, self.lanes, self.d, self.E, self.I
        b = ln["ballast"][0]
        gate = np.zeros((E, d), np.float32)
        gate[np.arange(E), lay["ROUTE"].start + np.arange(E)] = (ROUTE_ON - ROUTE_OFF) / 2 * RMS
        gate[:, b] = (ROUTE_ON + ROUTE_OFF) / 2 / 2.0
        w.add(pre + "gate.weight", gate)
        w.add(pre + "gate.bias", torch.zeros(E))
        if self.vision:
            w.add(pre + "gate.bias_vl", torch.zeros(E))
        g1, g3, g2 = (np.zeros((I, d), np.float32), np.zeros((I, d), np.float32), np.zeros((d, I), np.float32))
        slots = (u for u in range(I) if u % 32 != 31)            # slot 31 of each group: its anchor
        try:
            if self.moves_pointer(l):                             # pointer := register, register := 1
                reg = [next(slots) for _ in range(CODE)]
                ptr = [next(slots) for _ in range(CODE)]
                one = next(slots)
                for t in range(CODE):
                    g1[reg[t], b] = g1[ptr[t], b] = GATE / 2
                    g3[reg[t], lay["R"].start + t] = RMS
                    g3[ptr[t], lay["P"].start + t] = RMS
                    g2[lay["P"].start + t, reg[t]] = 1 / GATE
                    g2[lay["P"].start + t, ptr[t]] = -1 / GATE
                    g2[lay["R"].start + t, reg[t]] = -1 / GATE
                    g2[lay["R"].start + t, one] = 1 / GATE
                g1[one, b], g3[one, b] = GATE / 2, 0.5
            const = next(slots)                                   # constant unit: silu(8) * 1
            g1[const, b], g3[const, b] = GATE / 2, 0.5
            for j in range(E):                                    # TAG := 1 (routed experts then flip theirs)
                u = next(slots)
                g1[u, b] = GATE / 2
                g3[u, lay["TAG"].start + j] = RMS
                g2[lay["TAG"].start + j, u] = -1 / GATE
                g2[lay["TAG"].start + j, const] = 1 / GATE
            if l > 0:                                             # TAG must equal -ROUTE: else DIAG += 1.25
                for j in range(E):
                    for sgn in (1.0, -1.0):
                        u = next(slots)
                        g1[u, lay["TAG"].start + j] = g1[u, lay["ROUTE"].start + j] = sgn * ERR * RMS
                        g3[u, b] = 0.5
                        g2[ln["diag"][j], u] = 1 / GATE
        except StopIteration:
            raise SystemExit(f"probec: moe_intermediate_size {I} is too small for layer {l}'s units") from None
        for grp in range(I // 32):                                # anchors: gate 16 -> clamped to 10, up 1
            if np.any(g1[grp * 32:grp * 32 + 31]):
                g1[grp * 32 + 31, b], g3[grp * 32 + 31, b] = GATE, 0.5
        for name, m in (("w1", g1), ("w3", g3), ("w2", g2)):
            add_fp8(w, pre + f"shared_experts.{name}", m)
        for e in range(E):                                        # routed: TAG_e -= 8 * 0.25 = -2
            e1, e3, e2 = np.zeros((I, d), np.float32), np.zeros((I, d), np.float32), np.zeros((d, I), np.float32)
            up = 2.0 ** self.kexp[e] / 2
            e1[0, b], e3[0, b] = GATE / 2, up
            e1[1, b], e3[1, b] = GATE, up                         # anchor: clamped to 10, zero down
            e2[lay["TAG"].start + e, 0] = -(2.0 ** -self.kexp[e])
            for name, m in (("w1", e1), ("w3", e3), ("w2", e2)):
                add_mxfp4(w, pre + f"experts.{e}.{name}", m)

    # ------------------------------------------------------------------ all
    def write(self, w):
        self.write_embed_head(w)
        for l in range(self.L):
            pre = f"layers.{l}."
            w.add(pre + "attn_norm.weight", np.ones(self.d, np.float32))
            w.add(pre + "ffn_norm.weight", np.ones(self.d, np.float32))
            self.write_hc(w, pre)
            self.write_attention(w, l)
            self.write_moe(w, l)

    def meta(self):
        return {
            "program": "dsv41", "classes": self.C, "fillers": self.F, "layers": self.L,
            "nxt": self.nxt.tolist(), "pools": self.pools, "max_corr": self.mc, "id_offset": self.off,
            "diag_base": 2 * self.C + self.F, "experts": self.E, "routes": self.routes.tolist(),
            "roles": self.roles, "layout": self.lay.summary(),
            "margins": {"score_nats": (CODE - self.mc) * SQ / np.sqrt(LATENT),
                        "logit_nats": (CODE - self.mc) * LOGIT / RMS * 1.0},
        }
