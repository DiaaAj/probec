"""Figures of the technical report (docs/REPORT.md): the real-model comparison, the fault boundary, the release
sweep and the bug-cost estimate, plus the README's overview diagram, in the style of tools/figstyle.py. Each function returns a figure; tools/make_figures.py
registers every function in FIGURES and saves it under the function's name.

Data (docs/data): realdiff_configs.csv, realdiff_pairs.csv (tools/realdiff_analyze.py), boundary_*.csv and
readmap_*.csv (tools/boundary_analyze.py), sweep.csv (tools/sweep_analyze.py).
"""
import csv
import json
import math
import statistics
from pathlib import Path

import figstyle as fs
from figstyle import C

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"


def rows(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ---- §1 and the README's first figure: real model vs probe twin ---------------------------------------------

CASES = [("sglang", "Qwen/Qwen3-0.6B", "SGLang 0.5.21, Qwen3-0.6B", "NGRAM speculative decoding: sglang#44997"),
         ("vllm", "Qwen/Qwen3.5-0.8B", "vLLM 0.31, Qwen3.5-0.8B (hybrid)", "n-gram speculative decoding: vLLM #40875")]
SHORT = {"FLASH_ATTN/rerun": "same settings, rerun", "flashinfer/rerun": "same settings, rerun",
         "FLASHINFER/default": "FlashInfer attention", "TRITON_ATTN/default": "Triton attention",
         "triton/default": "Triton attention", "FLASH_ATTN/no_prefix_cache": "prefix caching off",
         "flashinfer/no_radix": "radix cache off", "FLASH_ATTN/chunk512": "chunked prefill 512",
         "flashinfer/chunk512": "chunked prefill 512", "FLASH_ATTN/eager": "eager (no CUDA graphs)",
         "flashinfer/no_cuda_graph": "CUDA graphs off", "flashinfer/no_overlap": "overlap scheduler off",
         "FLASH_ATTN/one_seq": "one request at a time", "flashinfer/one_seq": "one request at a time",
         "FLASH_ATTN/ngram": "n-gram speculative decoding", "flashinfer/ngram": "NGRAM speculative decoding"}


def _case_rows(cfg, engine, model):
    """(label, outputs differ, logprob check failed, probe wrong, probe flags it), counts of 192, per setting."""
    real = {r["config"]: r for r in cfg if r["side"] == "real" and r["engine"] == engine and r["model"] == model}
    twin = {r["config"]: r for r in cfg if r["side"] == "probe" and r["engine"] == engine
            and r["model"] == {"Qwen/Qwen3-0.6B": "dense", "Qwen/Qwen3.5-0.8B": "h08"}[model]}
    out = []
    for name, r in real.items():
        if not r.get("reference"):
            continue                                    # the reference setting itself
        p = twin.get(name)
        out.append((SHORT.get(name, name), int(num(r["differ_vs_reference"])),
                    int(num(r.get("logprob_check_failed_vs_reference"))),
                    int(num(p["wrong_requests"])) if p else 0, bool(p and num(p["wrong_requests"]))))
    out.sort(key=lambda t: (t[4], t[1]))
    return out


def realdiff(cases=None):
    cfg = rows("realdiff_configs.csv")
    cases = [(c, _case_rows(cfg, c[0], c[1])) for c in (cases or CASES)]
    cases = [(c, r) for c, r in cases if r]
    n = sum(len(r) + 1 for _, r in cases)
    fig = fs.figure(1.55 + 0.24 * n)
    fs.header(fig, "The same engine settings, on a real model and on its probe twin",
              "Each row changes one setting from the engine's default; 192 greedy requests. Real model (real-text prompts), "
              "compared with the default run: outputs that differ at all, and outputs that fail a logprob-tolerance check "
              "(vLLM's check_logprobs_close, top 5). Probe twin: outputs that differ from the known answer. "
              "Red: the setting the probe flags.")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 0.9, 0.9])
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    titles = ["real model:\noutputs differ", "real model:\nlogprob check fails", "probe twin:\nwrong"]
    for k, ax in enumerate(axes):
        groups = []
        for (engine, model, title, bug), rs in cases:
            groups.append((title if k == 0 else " ",
                           [(lab if k == 0 else "", vals[k], C.engine_bug if flag else C.exact, "")
                            for lab, *vals, flag in rs]))
        fs.hbar_rows(ax, groups, xmax=192 if k == 0 else 48)
        ax.set_xticks([0, 64, 128, 192] if k == 0 else [0, 16, 32, 48])
        ax.set_xlabel("requests", fontsize=fs.SMALL)
        ax.set_title(titles[k], fontsize=fs.SMALL, loc="left", color=C.text)
        if k:
            ax.set_yticks([])
    fs.footer(fig, "Data: docs/data/realdiff_configs.csv (tools/realdiff.py, RTX 3090) · python tools/make_figures.py")
    return fig


PAIR_CASES = [("SGLang 0.5.21 · Qwen3-0.6B", "Qwen/Qwen3-0.6B", ("sglang",)),
              ("vLLM 0.31 · Qwen3-0.6B (and HF)", "Qwen/Qwen3-0.6B", ("vllm", "hf")),
              ("vLLM 0.31 · Qwen3.5-0.8B (and HF)", "Qwen/Qwen3.5-0.8B", ("vllm", "hf"))]


def realdiff_pairs():
    """Every pair of settings of one engine (plus HF transformers): outputs that differ, logprob-check failures."""
    import numpy as np
    pairs = [p for p in rows("realdiff_pairs.csv") if "(no logprobs)" not in p["a"] + p["b"]]
    fig = fs.figure(2.8)
    fs.header(fig, "Correct settings disagree on a real model; a tolerance check finds the bugs",
              "Each dot is one pair of settings of the same engine (192 greedy requests, real-text prompts). Gray: two correct "
              "settings (or HF transformers). Red: pairs with the setting the probe flags.")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1])
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    labels = []
    for i, (title, model, engines) in enumerate(PAIR_CASES):
        ps = [p for p in pairs if p["model"] == model and p["a"].split(":")[0] in engines and p["b"].split(":")[0] in engines]
        if not ps:
            continue
        labels.append((i, f"{title}\n{len(ps)} pairs"))
        for p in ps:
            bad = p["a_flagged"] == "1" or p["b_flagged"] == "1"
            y = i + rng.uniform(-0.18, 0.18)
            for ax, v in ((ax1, num(p["differ"])), (ax2, num(p["logprob_check_failed"]))):
                ax.plot([v], [y], "o", ms=5 if bad else 4, color=C.engine_bug if bad else C.exact,
                        alpha=0.95 if bad else 0.75, mec="none", zorder=3 if bad else 2)
    for ax, xmax, xl in ((ax1, 192, "requests whose outputs differ"), (ax2, 40, "requests failing the logprob check")):
        ax.set_xlim(-4, xmax)
        ax.set_ylim(len(PAIR_CASES) - 0.5, -0.5)
        ax.set_xlabel(xl, fontsize=fs.SMALL)
        fs.style_axes(ax, grid="x", left=False)
    ax1.set_yticks([i for i, _ in labels], [l for _, l in labels], fontsize=fs.SMALL)
    ax1.tick_params(axis="y", labelcolor=C.text)
    ax2.set_yticks([])
    fs.footer(fig, "Data: docs/data/realdiff_pairs.csv · python tools/make_figures.py")
    return fig


# ---- §2.3: from a wrong token to the faulty block -------------------------------------------------------------

def readmap():
    meta = json.load(open(DATA / "readmap_trial.json"))
    reads = rows("readmap_reads.csv")
    L, pos, layer, bad = meta["layers"], int(meta["position"]), int(meta["layer"]), int(meta["first_bad_step"])
    smax = 255
    fig = fs.figure(3.6)
    fs.header(fig, "The first wrong token appears at the first step that reads the fault",
              f"vLLM 0.31, Qwen3-0.6B twin: one KV slot (prompt position {pos}) corrupted in layer {layer} only, "
              f"then the request reruns from the prefix cache.\nDots: the oracle's reads of that position, by layer and "
              f"decode step. Other layers read the slot without effect.")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.6, 1])
    ax = fig.add_subplot(gs[0])
    ax.axhspan(layer - 0.5, layer + 0.5, color=C.engine_bug, alpha=0.12, lw=0)
    xs = [int(r["step"]) for r in reads if int(r["step"]) <= smax]
    ys = [int(r["layer"]) for r in reads if int(r["step"]) <= smax]
    other = [(x, y) for x, y in zip(xs, ys) if y != layer]
    hit = [(x, y) for x, y in zip(xs, ys) if y == layer]
    ax.plot([x for x, _ in other], [y for _, y in other], "o", ms=4.5, color=C.text2, mec="none")
    ax.plot([x for x, _ in hit], [y for _, y in hit], "o", ms=6.5, color=C.engine_bug, mec="none")
    ax.axvline(bad, color=C.engine_bug, lw=1.2, ls=(0, (3, 2)))
    ax.annotate(f"engine's first wrong token: step {bad}", (bad, -0.8), xytext=(5, 0), textcoords="offset points",
                fontsize=fs.SMALL, color=C.engine_bug, va="center", annotation_clip=False)
    ax.set_xlim(-1, smax)
    ax.set_ylim(L - 0.5, -1.5)
    ax.set_xlabel("decode step")
    ax.set_ylabel("layer")
    fs.style_axes(ax, grid="x")
    # right: every content fault, by whether the generation reads the corrupted data
    st = rows("readmap_agreement.csv")
    ax2 = fig.add_subplot(gs[1])
    read = [r for r in st if r["predicted_step"] != ""]
    unread = [r for r in st if r["predicted_step"] == ""]
    exact = sum(1 for r in read if r["detected"] == "1" and r["observed_minus_predicted"] == "0")
    fs.hbar_rows(ax2, [(None, [("read", sum(int(r["detected"]) for r in read), C.exact, f"of {len(read)}"),
                               ("never read", sum(int(r["detected"]) for r in unread), C.exact, f"of {len(unread)}")])],
                 xmax=max(1, len(read)) * 1.6)
    ax2.set_xlabel("faults detected", fontsize=fs.SMALL)
    ax2.set_title(f"all {len(st)} content faults:\n{exact} went wrong exactly at\nthe step the oracle predicts",
                  fontsize=fs.SMALL, loc="left", color=C.text)
    fs.footer(fig, "Data: docs/data/readmap_*.csv (tools/fault_boundary.py, tools/boundary_analyze.py) · python tools/make_figures.py")
    return fig


# ---- §3.2: what it detects ------------------------------------------------------------------------------------

STRUCT = [("block_other_request", "block replaced by another request's block"),
          ("block_same_request", "block replaced by another block of the request"),
          ("block_zeroed", "block zeroed"), ("block_layer_below", "block holds the layer below's values"),
          ("block_stale_one_layer", "block stale in one layer only"), ("one_slot", "one slot replaced"),
          ("one_slot_one_layer", "one slot replaced, in one layer only"), ("one_kv_head", "one KV head replaced"),
          ("negate_k", "keys negated"), ("negate_v", "values negated"),
          ("block_other_request_random", "a random cached block replaced")]


def boundary():
    """Content faults: detected whenever the generation reads the corrupted data."""
    by = {(r["fault"], r["eps"]): r for r in rows("boundary_faults.csv")}
    agree = rows("readmap_agreement.csv")
    fig = fs.figure(3.9)
    fs.header(fig, "Content faults are detected whenever they are read",
              "Faults injected into vLLM 0.31's paged KV cache (Qwen3-0.6B twin), one block of one cached request at a time; "
              "the request then reruns from the cache. Read: the generation reads the corrupted data, in the corrupted "
              "layer (oracle replay). Every fault that is read is detected, at the step the oracle predicts.")
    ax = fig.add_subplot(1, 1, 1)
    groups = [(None, [])]
    for key, label in STRUCT:
        r = by.get((key, ""))
        if not r:
            continue
        a = [x for x in agree if x["fault"] == key]
        n = int(r["trials"])
        read = sum(1 for x in a if x["predicted_step"] != "")
        det_read = sum(1 for x in a if x["predicted_step"] != "" and x["detected"] == "1")
        det_unread = sum(1 for x in a if x["predicted_step"] == "" and x["detected"] == "1")
        note = f"{det_read} of {read} read" + ("" if read == n else f"; {n - read} never read, {det_unread} detected")
        groups[0][1].append((label, round(100 * det_read / max(1, read)), C.exact, note))
    fs.hbar_rows(ax, groups, xmax=100)
    ax.set_xlabel("detected when read (%)", fontsize=fs.SMALL)
    fs.footer(fig, "Data: docs/data/boundary_faults.csv, readmap_agreement.csv (RTX 3090) · python tools/make_figures.py")
    return fig


def boundary_numeric():
    """Numeric faults by size, and the softmax-scale check's tolerance band."""
    by = {(r["fault"], r["eps"]): r for r in rows("boundary_faults.csv")}
    fig = fs.figure(3.2)
    fs.header(fig, "Magnitude errors are absorbed; softmax-scale errors past ±35% are flagged",
              "Left: K or V of one read KV block scaled, noised or partly sign-flipped (vLLM 0.31, Qwen3-0.6B twin, 6 trials per "
              "point). Right: every attention layer's softmax scale multiplied by a factor (48 requests per point; shaded: "
              "the band the scale check tolerates by design).")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for kind, color, label in (("scale", fs.CATEGORICAL[0], "× (1+ε)"), ("noise", fs.CATEGORICAL[1], "+ ε · rms noise"),
                               ("flip", fs.CATEGORICAL[2], "fraction ε of signs flipped")):
        for part, ls in (("v", "-"), ("k", (0, (3, 2)))):
            pts = sorted((num(e), 100 * num(r["detected"]) / max(1, num(r["trials"])))
                         for (f, e), r in by.items() if f == f"{kind}_{part}")
            if pts:
                ax2.plot([p for p, _ in pts], [d for _, d in pts], ls=ls, marker="o", ms=3.2, color=color,
                         label=label if part == "v" else None)
    ax2.set_xscale("log")
    ax2.set_ylim(-5, 105)
    ax2.set_xlabel("fault size ε (solid: values, dashed: keys)", fontsize=fs.SMALL)
    ax2.set_ylabel("detected (%)", fontsize=fs.SMALL)
    ax2.legend(loc="center left", fontsize=7.5)
    fs.style_axes(ax2, grid="y")
    sc = sorted((num(r["factor"]), 100 * num(r["wrong_requests"]) / max(1, num(r["requests"]))) for r in rows("boundary_scale.csv"))
    ax3.axvspan(0.75, 1.35, color=C.exact, alpha=0.15, lw=0)
    ax3.plot([f for f, _ in sc], [d for _, d in sc], "o-", ms=3.2, color=C.text)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks([0.5, 0.71, 1, 1.41, 2], ["½", "1/√2", "1", "√2", "2"])
    ax3.set_ylim(-5, 105)
    ax3.set_xlabel("softmax scale × factor", fontsize=fs.SMALL)
    ax3.set_ylabel("requests flagged (%)", fontsize=fs.SMALL)
    fs.style_axes(ax3, grid="y")
    fs.footer(fig, "Data: docs/data/boundary_faults.csv, boundary_scale.csv (RTX 3090) · python tools/make_figures.py")
    return fig


# ---- §3.3: release history ------------------------------------------------------------------------------------

SWEEP_COLS = {"vllm": [("dense", "default/prefix", "dense"), ("dense", "default/ngram_spec_decoys", "dense\nn-gram"),
                       ("dense", "default/fp8kv", "dense\nFP8 KV"), ("mla", "default/prefix", "MLA"),
                       ("mla", "default/shared_prefix_long_suffix", "MLA\nlong suffix"),
                       ("dense8b-fp8", "default/prefix", "FP8\nweights"), ("dense-nvfp4", "default/prefix", "NVFP4"),
                       ("moe", "default/prefix", "MoE"), ("h08", "default/prefix", "hybrid"),
                       ("h08", "default/ngram_spec_decoys", "hybrid\nn-gram")],
              "sglang": [("dense", "flashinfer/default", "dense"), ("dense_nsc", "flashinfer/ngram_spec", "dense\nNGRAM"),
                         ("mla", "triton/default", "MLA"), ("moe-awq", "flashinfer/default", "MoE AWQ\nrun 1"),
                         ("moe-awq-rep", "flashinfer/default", "MoE AWQ\nrun 2"),
                         ("dense-fp8-block", "flashinfer/default", "FP8 block\nrun 1"),
                         ("dense-fp8-block-rep", "flashinfer/default", "FP8 block\nrun 2"),
                         ("dense-nvfp4", "flashinfer/default", "NVFP4"), ("h08", "flashinfer/default", "hybrid")]}


def sweep():
    data = rows("sweep.csv")
    import numpy as np
    from matplotlib.patches import Rectangle
    engines = [e for e in ("vllm", "sglang") if any(r["engine"] == e for r in data)]
    vers = {e: sorted({r["version"] for r in data if r["engine"] == e}, key=lambda v: [int(x) for x in v.split(".")])
            for e in engines}
    h = 1.3 + sum(0.26 * len(vers[e]) + 0.75 for e in engines)
    fig = fs.figure(h)
    fs.header(fig, "One fixed probe suite on every release",
              "Wrong requests per entry (of 192; MLA: of 96 and 33), RTX 3090. Gray: the release cannot load that twin; "
              "×: the engine crashed or a kernel failed to build. Requests end by length, which hides sglang#44997 from "
              "0.5.22 on (#42720); Blackwell-only bugs (vLLM #52796) do not show on this GPU.")
    gs = fig.add_gridspec(len(engines), 1, height_ratios=[len(vers[e]) + 2.5 for e in engines])
    for k, e in enumerate(engines):
        ax = fig.add_subplot(gs[k])
        cols = SWEEP_COLS[e]
        cell = {(r["version"], r["twin"], r["config"]): r for r in data if r["engine"] == e}
        for i, v in enumerate(vers[e]):
            for j, (twin, conf, _) in enumerate(cols):
                r = cell.get((v, twin, conf))
                if r is None:
                    color, txt, tc = "none", "", C.text2
                elif r["error"] or r["wrong_requests"] == "":
                    color, txt, tc = C.error, "" if r["error"].startswith("unsupported") else "×", C.text2
                elif int(r["wrong_requests"]) == 0:
                    color, txt, tc = C.surface, "0", C.text2
                else:
                    color, txt, tc = C.engine_bug, r["wrong_requests"], C.on_fill
                ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=color, edgecolor=C.grid, lw=0.6))
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=tc)
        ax.set_xlim(-0.6, len(cols) - 0.4)
        ax.set_ylim(len(vers[e]) - 0.5, -0.6)
        ax.set_yticks(range(len(vers[e])), [f"{'vLLM' if e == 'vllm' else 'SGLang'} {v}" for v in vers[e]], fontsize=8)
        ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=7.5)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fs.footer(fig, "Data: docs/data/sweep.csv (RTX 3090) · python tools/make_figures.py")
    return fig


# ---- README: how probec works (a diagram, drawn in inches on one axes) -------------------------------------

def _card(ax, cx, top, w, h, eyebrow, title, sub=None, fill=None, edge=None, mark=None, mark_color=None):
    from matplotlib.patches import FancyBboxPatch
    x0 = cx - w / 2
    ax.add_patch(FancyBboxPatch((x0, top - h), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill or C.card, edgecolor=edge or C.card_edge, linewidth=0.9))
    x = x0 + 0.2
    ax.text(x, top - 0.2, eyebrow, fontsize=7.5, fontweight="bold", color=C.text2, va="center")
    x_title = x
    if mark:
        ax.text(x, top - 0.43, mark, fontsize=12, fontweight="bold", color=mark_color, va="center")
        x_title = x + 0.24
    ax.text(x_title, top - 0.43, title, fontsize=11.5, fontweight="bold", color=C.text, va="center")
    if sub:
        ax.text(x, top - 0.62, sub, fontsize=fs.SMALL, color=C.text2, va="top", linespacing=1.4)
    return x0, top - h


def _arrow(ax, verts, codes=None, head=True):
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path as MPath
    style = "-|>,head_length=0.45,head_width=0.22" if head else "-"
    ax.add_patch(FancyArrowPatch(path=MPath(verts, codes), arrowstyle=style, mutation_scale=12, color=C.axis,
                                 linewidth=1.1, shrinkA=0, shrinkB=0))


def _elbow(ax, x0, y0, x1, y1, r=0.14):
    """Down from (x0, y0), across at the midpoint, down to (x1, y1), with rounded corners."""
    from matplotlib.path import Path as MPath
    ym, d = (y0 + y1) / 2, (1 if x1 > x0 else -1)
    verts = [(x0, y0), (x0, ym + r), (x0, ym), (x0 + d * r, ym), (x1 - d * r, ym), (x1, ym), (x1, ym - r),
             (x1, y1)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO]
    _arrow(ax, verts, codes)


def overview():
    from matplotlib.patches import FancyBboxPatch
    W, H = fs.WIDTH, 7.7
    fs.apply()
    fig = fs.plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    xl, xr, cw = 2.3, 5.7, 3.1          # the two columns: centers and card width
    mono = dict(family="DejaVu Sans Mono", fontsize=fs.SMALL)

    top = H - 0.12
    _card(ax, W / 2, top, 4.3, 0.78, "INPUT", "A real model's config.json",
          "a Hub id or a local path, e.g. Qwen/Qwen3-0.6B")
    y = top - 0.78
    pill_y, pill_w, pill_h = y - 0.42, 1.9, 0.36       # the compile step, on the stem
    _arrow(ax, [(W / 2, y), (W / 2, pill_y + pill_h / 2)], head=False)
    ax.add_patch(FancyBboxPatch((W / 2 - pill_w / 2, pill_y - pill_h / 2), pill_w, pill_h,
                                boxstyle=f"round,pad=0,rounding_size={pill_h / 2}", facecolor=fs.CATEGORICAL[0],
                                edgecolor="none"))
    ax.text(W / 2, pill_y, "probec compile", color=C.on_fill, ha="center", va="center", fontweight="bold", **mono)

    top = pill_y - pill_h / 2 - 0.5
    _elbow(ax, W / 2, pill_y - pill_h / 2, xl, top)
    _elbow(ax, W / 2, pill_y - pill_h / 2, xr, top)
    blue = dict(fill=C.accent_soft, edge=fs.CATEGORICAL[0])
    _card(ax, xl, top, cw, 0.98, "BUILT BY PROBEC", "Probe twin",
          "the real architecture and shapes;\nweights chosen so the output is known", **blue)
    _card(ax, xr, top, cw, 0.98, "BUILT BY PROBEC", "Oracle",
          "the right token at every step, and\nthe cache entries each step reads", **blue)

    y = top - 0.98
    top = y - 0.5
    _arrow(ax, [(xl, y), (xl, top)])
    ax.text(xl + 0.1, (y + top) / 2, "loads like any checkpoint", fontsize=8, color=C.text2, va="center")
    _card(ax, xl, top, cw, 0.98, "UNDER TEST", "vLLM or SGLang",
          "unmodified, real kernels, any settings:\ncaching, speculation, FP8, FP4, …")

    e_top = top - 0.98 - 0.5
    _arrow(ax, [(xl, top - 0.98), (xl, e_top)])
    ax.text(xl + 0.1, top - 0.98 - 0.25, "generated tokens", fontsize=8, color=C.text2, va="center")
    _arrow(ax, [(xr, y), (xr, e_top)])
    ax.text(xr + 0.1, (y + e_top) / 2, "expected tokens", fontsize=8, color=C.text2, va="center")

    e_h, e_w = 1.12, xr - xl + cw
    x0, _ = _card(ax, W / 2, e_top, e_w, e_h, "CHECK", "Compare every token",
                  "no tolerance and no reference run:\na correct engine matches bit for bit")
    # two token streams; the engine's goes wrong at one step and may differ after it
    n, s, g, k = 16, 0.135, 0.045, 10    # tokens per stream, size, gap, first wrong one
    sx = x0 + e_w - 0.22 - n * (s + g) + g
    for row, (label, yy) in enumerate((("oracle", e_top - 0.40), ("engine", e_top - 0.72))):
        ax.text(sx - 0.12, yy, label, ha="right", va="center", color=C.text2, family="DejaVu Sans Mono", fontsize=8)
        for i in range(n):
            bad = row == 1 and i >= k
            ax.add_patch(FancyBboxPatch((sx + i * (s + g), yy - s / 2), s, s, boxstyle="round,pad=0,rounding_size=0.03",
                                        facecolor=(C.engine_bug if i == k else C.bug_soft) if bad else C.exact,
                                        edgecolor=C.engine_bug if bad else "none", linewidth=0.8))
    bx = sx + k * (s + g) + s / 2
    ax.text(bx, e_top - 0.72 - s / 2 - 0.1, "first wrong token", ha="center", va="top", fontsize=7.5,
            color=C.engine_bug)

    top = e_top - e_h - 0.48
    _arrow(ax, [(xl, e_top - e_h), (xl, top)])
    _arrow(ax, [(xr, e_top - e_h), (xr, top)])
    _card(ax, xl, top, cw, 1.0, "RESULT", "Exact", "every token of every request matches",
          fill=C.exact_soft, edge=C.exact, mark="✓", mark_color=C.exact)
    _card(ax, xr, top, cw, 1.0, "RESULT", "Engine bug",
          "the request, the step, and the KV block,\nexpert or state it read wrongly",
          fill=C.bug_soft, edge=C.engine_bug, mark="✗", mark_color=C.engine_bug)
    return fig


# ---- §4 and the README: how long each bug shipped, and what it costs (tools/bug_costs.py) ------------------

FEATURES = [("mla_prefix", "MLA models (DeepSeek-style) with prefix caching"),
            ("ngram", "n-gram speculative decoding"),
            ("fp8_ampere", "FP8 weights on Ampere GPUs (Marlin)"),
            ("nvfp4_marlin", "NVFP4 weights on GPUs before Blackwell (Marlin)"),
            ("awq_moe", "AWQ MoE weights on Ampere GPUs (Marlin MoE)")]


def _date(s):
    import datetime as dt
    return dt.date.fromisoformat(s)


def bug_timeline():
    import bug_costs
    import matplotlib.dates as mdates
    from matplotlib import transforms
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    costs = {r["id"]: r for r in rows("bug_costs.csv")}
    bugs = rows("bugs.csv")
    groups = [(title, [b for b in bugs if b["feature"] == f]) for f, title in FEATURES]
    slots = sum(len(bs) + 1 for _, bs in groups)
    fixed = sorted(int(costs[b["id"]]["days"]) for b in bugs if b["first_good"])
    med = statistics.median(fixed)
    fig = fs.figure(2.0 + 0.27 * slots)
    fs.header(fig, "How long each bug shipped",
              f"Bars: the releases with the bug, from the first release known to have it (the release sweep, or git "
              f"ancestry for #44735 and #34577) to the first release without it, or to today. probec flags each bug in "
              f"the first release it tested that has it. The fixed bugs shipped for {fixed[0]}–{fixed[-1]} days (median "
              f"{med:g}). Right: releases (patch releases included) and days.")
    handles = [Line2D([], [], color=C.engine_bug, lw=6), Line2D([], [], color=C.engine_bug, lw=1.2, ls=(0, (3, 2))),
               Line2D([], [], ls="", marker="o", ms=6.5, mfc=C.surface, mec=C.text, mew=1.3),
               Line2D([], [], ls="", marker="D", ms=5.5, color=C.text),
               Line2D([], [], ls="", marker=">", ms=6, color=C.engine_bug)]
    lax = fs.strip_axes(fig, 0.22)
    lax.axis("off")
    lax.legend(handles, ["in releases", "known earlier", "first public report", "fix merged", "still open"], ncols=5, loc="center left", bbox_to_anchor=(0, 0.5), borderaxespad=0,
               handlelength=1.5, columnspacing=1.2, handletextpad=0.5, labelcolor=C.text2)
    fs.footer(fig, "Data: docs/data/bug_costs.csv (python tools/bug_costs.py), bugs.csv, release_dates.csv (PyPI) · "
                   "python tools/make_figures.py")
    ax = fig.subplots()
    num = mdates.date2num
    today = num(bug_costs.TODAY)
    y, ticks, labels = 0.0, [], []
    left = transforms.blended_transform_factory(fig.transFigure, ax.transData)
    for title, bs in groups:
        if y:
            y += 0.35
        ax.text(fs.PAD / fig.get_figwidth(), y + 0.1, title, transform=left, ha="left", va="center", fontsize=fs.SMALL,
                fontweight="bold", color=C.text, in_layout=False)
        head_y, y0 = y + 0.1, y + 0.5
        y += 1
        for b in bs:
            c = costs[b["id"]]
            s, e = num(_date(c["start"])), num(_date(c["end"]))
            ax.barh(y, e - s, left=s, height=0.56, color=C.engine_bug, linewidth=0, zorder=3)
            if c["start_open"] == "1" and not b["reported"]:    # earlier releases had it too, or could not be tested
                ax.plot([s - 7, s], [y, y], color=C.engine_bug, lw=2.0, ls=(0, (1, 1.2)), solid_capstyle="butt",
                        zorder=3)
            if c["open"] == "1":
                ax.plot([e + 1.5], [y], marker=">", ms=6, color=C.engine_bug, mec="none", zorder=4, clip_on=False)
            if b["reported"]:
                r = num(_date(b["reported"]))
                if r < s:                     # reported before the first release the sweep could test
                    ax.plot([r, s], [y, y], color=C.engine_bug, lw=1.2, ls=(0, (3, 2)), zorder=2)
                ax.plot([r], [y], "o", ms=6.5, mfc=C.surface, mec=C.text, mew=1.3, zorder=5)
            if b["fix_merged"]:
                ax.plot([num(_date(b["fix_merged"]))], [y], "D", ms=5.5, color=C.text, mec=C.surface, mew=0.8, zorder=5)
            more = "≥ " if c["start_open"] == "1" or c["open"] == "1" else ""
            nrel = int(c["releases"])
            ax.annotate(f"{more}{nrel} release{'s' * (nrel != 1)} · {more}{c['days']} days", (1.0, y),
                        xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points", va="center",
                        fontsize=fs.SMALL, color=C.text, annotation_clip=False)
            ticks.append(y)
            labels.append(b["label"].split(" · ")[0].replace("vLLM (unreported)", "vLLM, unreported (#50613)"))
            y += 1
        if title.startswith("MLA"):      # the stretch with no vLLM release free of these bugs (on every GPU)
            vl = [c for c in (costs[b["id"]] for b in bs) if c["engine"] == "vLLM" and c["gpus"] == "all"]
            a, z = min(num(_date(c["start"])) for c in vl), max(num(_date(c["end"])) for c in vl)
            ax.add_patch(Rectangle((a, y0), z - a, y - 0.5 - y0, facecolor=C.bug_soft, edgecolor="none", zorder=1))
            x_txt = fs.PAD + fs.text_width(fig, title + "  ", fs.SMALL, "bold")
            ax.text(x_txt / fig.get_figwidth(), head_y, f"no vLLM release without one of these for {int(z - a)} days "
                    "(shaded)", transform=left, ha="left", va="center", fontsize=8, color=C.engine_bug, in_layout=False)
    ax.axvline(today, color=C.axis, lw=0.9, ls=(0, (2, 2)), zorder=2)
    ax.text(today, -0.5, "today", ha="right", va="bottom", fontsize=7.5, color=C.text2)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.4, -0.6)
    first = min([num(_date(c["start"])) for c in costs.values()] + [num(_date(b["reported"])) for b in bugs if b["reported"]])
    ax.set_xlim(first - 12, today + 6)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlabel(str(bug_costs.TODAY.year), color=C.text2, fontsize=fs.SMALL)
    fs.style_axes(ax, grid="x", left=False)
    ax.tick_params(axis="y", labelcolor=C.text, labelsize=fs.SMALL)
    return fig


def _usd(x):
    """Dollars, escaped for matplotlib's mathtext."""
    return r"\$" + (f"{x / 1000:.0f}k" if x >= 9500 else f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}")


def _one_in(share, noun=True):
    if share > 0.97:
        return "every request"
    if share >= 0.9:
        return "almost every request"
    return f"1 in {round(1 / share)}" + (" requests" if noun else "")


def _times(text):
    """'2.3x' -> '2.3×' (the CSVs use a plain x)."""
    import re
    return re.sub(r"(\d)x\b", "\\1×", text)


def bug_costs():
    """One card per feature: what users saw, what turning the feature off costs, what leaving it on does, and how
    long the bugs shipped; and a card for what the check costs."""
    import bug_costs as bc
    from matplotlib.patches import FancyBboxPatch
    costs = rows("bug_costs.csv")
    bugs = {b["id"]: b for b in rows("bugs.csv")}
    work = {a["workaround"]: a for a in rows("cost_assumptions.csv")}
    price = bc.constants()
    chk = bc.check()
    W, cols, cw, ch, gap = fs.WIDTH, 2, 3.84, 1.98, 0.16
    n = len(FEATURES) + 1
    nrows = -(-n // cols)
    fig = fs.figure(1.25 + nrows * ch + (nrows - 1) * gap + 0.3)
    fs.header(fig, "What each bug costs while it ships (back-of-envelope)",
              f"Until a fix ships, a deployment that uses the affected feature can turn it off and pay for the lost "
              f"efficiency, or leave it on and serve wrong tokens. Dollars: extra GPU spend for one {bc.NODE_GPUS}-GPU node "
              f"serving around the clock, at the median on-demand price of an H100 (\\${price['price:all']:.2f}/h) or, for "
              f"Ampere-only bugs, an A100 (\\${price['price:Ampere']:.2f}/h). Sources: docs/REPORT.md §4.")
    fs.footer(fig, "Data: docs/data/bug_costs.csv and its inputs (python tools/bug_costs.py) · python tools/make_figures.py")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_in_layout(False)
    H = fig.get_figheight()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    top0 = H - fig._probec_bands[0] - 0.02
    x0s = [fs.PAD, W - fs.PAD - cw]
    for k in range(n):
        x, top = x0s[k % cols], top0 - (k // cols) * (ch + gap)
        last = k == n - 1
        ax.add_patch(FancyBboxPatch((x, top - ch), cw, ch, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor=C.accent_soft if last else C.card,
                                    edgecolor=fs.CATEGORICAL[0] if last else C.card_edge, linewidth=0.9))
        tx, rx = x + 0.18, x + 2.2
        if last:
            eyebrow, ids = "THE CHECK", "probec's release suite (report §3.3), one RTX 3090"
            saw = "the same twins and settings on every release; no reference run, no tolerance"
            big, unit = r"\$" + f"{chk['usd_per_release_high']:.2f}", "or less per release"
            right = [("time per release", f"vLLM: {chk['vllm_minutes']:.0f} min\nSGLang: {chk['sglang_minutes']:.0f} min",
                      C.text)]
            foot = (f"flags each of these bugs in the first release that has it; all {chk['releases']} releases: "
                    f"{chk['gpu_hours']:.1f} GPU-hours, \\${chk['usd_total']:.2f} at \\${chk['usd_per_gpu_hour']:.2f}/h")
        else:
            feature, title = FEATURES[k]
            cs = [c for c in costs if c["feature"] == feature]
            eyebrow = title.upper()
            ids = " · ".join(c["label"].split(" · ")[0].replace(" (unreported)", ", unreported") for c in cs)
            saws = [bugs[c["id"]]["users_saw"] for c in cs if bugs[c["id"]]["users_saw"]]
            saw = "users saw: " + saws[0] if saws else "no user report before probec's"
            usd = [c for c in cs if c["metric"] == "gpus"]
            speed = [c for c in cs if c["metric"] == "speed"]
            if usd:
                lo = min(int(c["extra_usd_per_node_month_low"]) for c in usd)
                hi = max(int(c["extra_usd_per_node_month_high"]) for c in usd)
                big = f"up to {_usd(hi)}" if lo == 0 else f"{_usd(lo)}–{_usd(hi)}"
                unit = "a month per node, to work around"
            elif speed:
                lo, hi = min(float(c["factor_low"]) for c in speed), max(float(c["factor_high"]) for c in speed)
                big, unit = f"{lo:g}–{hi:g}×", "slower decoding, to turn it off"
            else:
                big, unit = "not measured", "the cost of the workaround"
            shares = sorted(float(c["wrong_share"]) for c in cs)
            wrong = _one_in(shares[0]) if _one_in(shares[0]) == _one_in(shares[-1]) else \
                f"{_one_in(shares[0], noun=False)} to {_one_in(shares[-1])}"
            right = [("left on, in our runs", f"{wrong} wrong", C.engine_bug)]
            foot = "; ".join(dict.fromkeys(_times(work[bugs[c["id"]]["workaround"]]["basis"]) for c in cs))
        ax.text(tx, top - 0.2, eyebrow, fontsize=7.5, fontweight="bold", color=C.text2, va="center")
        ax.text(tx, top - 0.4, ids, fontsize=8, color=C.text2, va="center")
        ax.text(tx, top - 0.56, fs.wrap(fig, saw, 8, cw - 0.36), fontsize=8, color=C.text, va="top",
                fontstyle="italic", linespacing=1.3)
        ax.text(tx, top - 1.16, big, fontsize=18 if not big.startswith("not") else 13, fontweight="bold",
                color=C.text, va="center")
        ax.text(tx, top - 1.42, unit, fontsize=8, color=C.text2, va="center")
        yy = top - 0.98
        for label, value, color in right:
            ax.text(rx, yy, label, fontsize=7.5, color=C.text2, va="top")
            value = fs.wrap(fig, value, 8.5, x + cw - 0.15 - rx)
            ax.text(rx, yy - 0.15, value, fontsize=8.5, color=color, va="top", linespacing=1.25)
            yy -= 0.15 + 0.17 * (value.count("\n") + 1) + 0.06
        ax.text(tx, top - ch + 0.12, fs.wrap(fig, foot, 7.5, cw - 0.36), fontsize=7.5, color=C.text2, va="bottom",
                linespacing=1.3)
    return fig


FIGURES = [realdiff, realdiff_pairs, readmap, boundary, boundary_numeric, sweep, overview, bug_timeline, bug_costs]
