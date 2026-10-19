"""Regenerate the report figures (docs/figures/*.svg) from the data in docs/data/*.csv.

usage: python tools/make_figures.py [NAME ...] [--preview DIR]

Every figure is a function below registered with @figure: it reads the CSVs it needs and returns a matplotlib
figure, and figstyle.save writes it as docs/figures/<name>.svg and <name>-dark.svg. To add one, write a function,
decorate it and run this script with its name. --preview DIR also writes PNGs of both themes at README size
(keep them out of the repository).

Data:
- entries.csv: one row per matrix entry, from the raw run records (run: the GPU run that produced it)
- bugs.csv: the engine bugs, their release ranges and dates, and how entries.csv's cause names them
- sglang_ngram.csv: the SGLang NGRAM bug's runs (docs/RESULTS.md §3)
- realdiff_*.csv, boundary_*.csv, readmap_*.csv, sweep.csv, bug_*.csv: the report's experiments and cost estimate
  (tools/figures_report.py)
"""
import argparse
import collections
import csv
import statistics

from matplotlib.patches import Patch

import figstyle as fs
from figstyle import C

DATA = fs.ROOT / "docs" / "data"
GPUS = ["RTX 3090", "H100", "B200"]
ENGINES = {"vllm": "vLLM", "sglang": "SGLang"}
FIGS = {}


def figure(fn):
    FIGS[fn.__name__] = fn
    return fn


def read(name):
    with open(DATA / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def source(*names):
    return "Data: " + ", ".join(f"docs/data/{n}.csv" for n in names) + " · python tools/make_figures.py"


def family(gpu):
    """RTX 3090, H100 or B200: the 2-GPU runs count as RTX 3090, the H100 SXM and PCIe as H100."""
    return {"2x RTX 3090": "RTX 3090", "H100 PCIe": "H100", "H100 SXM": "H100"}.get(gpu, gpu)


def matrix_entries():
    """entries.csv rows with tokens; stops on a GPU, engine or program the figures do not know yet."""
    rows = [r for r in read("entries") if r["gen_tokens"]]
    for r in rows:
        if family(r["gpu"]) not in GPUS or r["engine"] not in ENGINES or r["program"] not in fs.PROGRAMS:
            raise SystemExit(f"unknown GPU, engine or program in {r['run']}: {r['gpu']!r}, {r['engine']!r}, "
                             f"{r['program']!r} (make_figures.GPUS, figstyle.PROGRAMS)")
    return rows


def pooled(label, entries, note_unit="entries"):
    """One hbar_rows row for entries that all have the same number of wrong requests."""
    vals = sorted({int(r["mismatched_requests"]) for r in entries})
    assert len(vals) == 1, (label, vals)
    v = vals[0]
    return (label, v, C.engine_bug if v else C.exact, f"{len(entries)} {note_unit}" if len(entries) > 1 else "")


@figure
def coverage():
    rows = matrix_entries()
    tok, ent = collections.Counter(), collections.Counter()
    for r in rows:
        key = (r["engine"], family(r["gpu"]))
        tok[key + (r["program"],)] += int(r["gen_tokens"])
        ent[key] += 1
    total = sum(tok.values())
    fig = fs.figure(4.5)
    fs.header(fig, "Generated tokens checked against the oracle",
              f"{total / 1e6:.1f}M tokens in {len(rows)} matrix entries, by engine, GPU and probe program; every "
              "release and build tested, replays and bisection included. "
              "RTX 3090 includes the 2-GPU parallelism runs, H100 the SXM and PCIe cards.")
    fs.footer(fig, source("entries"))
    axes = fig.subplots(2, 1, sharex=True)
    widest = max(sum(tok[(e, g, p)] for p in fs.PROGRAMS) for e in ENGINES for g in GPUS) / 1e6
    for ax, (eng, name) in zip(axes, ENGINES.items()):
        left = [0.0] * len(GPUS)
        for p in fs.PROGRAMS:
            w = [tok[(eng, g, p)] / 1e6 for g in GPUS]
            ax.barh(range(len(GPUS)), w, left=left, height=0.62, color=fs.PROGRAM[p], edgecolor=C.surface,
                    linewidth=1.2, label=p)
            left = [a + b for a, b in zip(left, w)]
        for i, g in enumerate(GPUS):
            ax.annotate(f"{left[i]:.2f}M · {ent[(eng, g)]} entries", (left[i], i), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=fs.SMALL, color=C.text)
        ax.set_yticks(range(len(GPUS)), GPUS)
        ax.set_ylim(len(GPUS) - 0.45, -0.55)
        ax.set_title(name)
        fs.style_axes(ax, grid="x", left=False)
        ax.tick_params(axis="y", labelcolor=C.text, labelsize=fs.FONT)
    axes[0].set_xlim(0, widest * 1.3)
    axes[0].set_xticks(range(int(widest) + 1))
    axes[-1].set_xlabel("generated tokens (millions)", color=C.text2, fontsize=fs.SMALL)
    axes[0].legend(ncols=5, loc="lower right", bbox_to_anchor=(1, 1.0), columnspacing=1.2, handletextpad=0.5)
    return fig


@figure
def outcomes():
    rows, bugs = read("entries"), read("bugs")
    n = collections.Counter(r["outcome"] for r in rows)
    per_bug, probe = collections.Counter(), collections.Counter()
    for r in rows:
        if r["outcome"] == "engine_bug":
            match = [b["id"] for b in bugs if b["cause_match"] in r["cause"]]
            assert len(match) == 1, (r["cause"], match)
            per_bug[match[0]] += 1
        elif r["outcome"] == "probe_bug":
            probe[r["program"]] += 1
    bad = n["engine_bug"] + n["probe_bug"] + n["unexplained"]
    fig = fs.figure(6.0)
    fs.header(fig, "Outcome of every matrix entry",
              f"{len(rows)} entries: {n['exact']} exact; {bad} with wrong tokens ({n['engine_bug']} from engine bugs, "
              f"{n['probe_bug']} from probe bugs fixed during validation, {n['unexplained']} unexplained); "
              f"{n['error']} crashed or did not start (unsupported configurations, out of memory, a bad host), "
              "so they have no tokens.")
    fs.footer(fig, source("entries", "bugs"))
    top = fs.strip_axes(fig, 0.55)
    order = ["exact", "engine_bug", "probe_bug", "error", "unexplained"]
    left = 0
    for o in order:
        if n[o]:
            top.barh(0, n[o], left=left, height=1, color=fs.OUTCOME[o], edgecolor=C.surface, linewidth=1.5)
            left += n[o]
    top.set_xlim(0, len(rows))
    top.set_ylim(2.6, -0.5)
    top.axis("off")
    top.legend([Patch(facecolor=fs.OUTCOME[o]) for o in order], [f"{fs.OUTCOME_LABEL[o]}: {n[o]}" for o in order],
               ncols=5, loc="lower left", bbox_to_anchor=(0, -0.08), columnspacing=1.6, handletextpad=0.5,
               borderaxespad=0, labelcolor=C.text)
    bottom = fig.subplots()
    kinds = [b["kind"].split(" (")[0] for b in bugs]
    groups = [(f"The {bad} entries with wrong tokens, by cause (bugs in the order they were found)",
               [(b["label"], per_bug[b["id"]], C.engine_bug, k) for b, k in zip(bugs, kinds)]),
              ("", [("probe bugs, found and fixed", n["probe_bug"], C.probe_bug,
                     ", ".join(f"{p} {probe[p]}" for p in fs.PROGRAMS if probe[p])),
                    ("unexplained", n["unexplained"], C.unexplained, "")])]
    fs.hbar_rows(bottom, groups, xmax=max(max(per_bug.values()), n["probe_bug"]) * 1.9, header_gap=0.5)
    bottom.set_xlabel("matrix entries", color=C.text2, fontsize=fs.SMALL)
    return fig


@figure
def hybrid_isolation():
    # entries.csv lists each run's files in name order, so the 0.8B twin's entries (v_h08*, s_h08*) come
    # before the other twins' single default entries
    first = {}
    for r in read("entries"):
        if r["program"] == "hybrid" and r["run"] in ("s26a", "s26b", "s27b"):
            first.setdefault((r["run"], r["config"]), r)

    def rows(run, *configs):
        got = [first[(run, c)] for c in configs]
        assert all(r["requests"] == "192" for r in got)
        return got

    a, b = "s26a", "s27b"
    fa = "FLASH_ATTN/"
    groups = [
        ("n-gram speculation, k = 4, decoy prompts", [
            pooled("prefix caching on (the default)", rows(a, fa + "ngram_spec_decoys")),
            pooled("VLLM_GDN_DECODE_KERNEL=triton", rows(a, fa + "ngram_spec_decoys_triton_verify")),
            pooled("prefix caching off", rows(a, fa + "ngram_spec_decoys_no_prefix_cache"))]),
        ("the same entry with one setting changed (another host)", [
            pooled("unchanged, and a repeat", rows(b, fa + "ngram_spec_decoys", fa + "ngram_spec_decoys_rep"), "runs"),
            pooled("async_scheduling=False", rows(b, fa + "ngram_spec_decoys_no_async")),
            pooled("enforce_eager=True", rows(b, fa + "ngram_spec_decoys_eager")),
            pooled("max_num_seqs=1 (one request at a time)", rows(b, fa + "ngram_spec_decoys_seqs1")),
            pooled("num_speculative_tokens=2", rows(b, fa + "ngram_spec_decoys_k2")),
            pooled("num_speculative_tokens=1", rows(b, fa + "ngram_spec_decoys_k1")),
            pooled('method="ngram_gpu"', rows(b, fa + "ngram_gpu_decoys")),
            pooled("no decoy prompts: every draft accepted", rows(b, fa + "ngram_spec_plain"))]),
        ("without speculation", [
            pooled("FLASH_ATTN, FLASHINFER, TRITON_ATTN",
                   rows(a, fa + "default", "FLASHINFER/default", "TRITON_ATTN/default")),
            pooled("shared prefixes (short, long suffixes); no prefix cache",
                   rows(a, fa + "shared_prefix", fa + "shared_prefix_long_suffix", fa + "no_prefix_cache")),
            pooled("chunked prefill + sampling; eager; preemption",
                   rows(a, fa + "chunk512_sample", fa + "eager", fa + "preempt")),
            pooled("bf16 SSM cache; non-packed GDN decode; FP8 KV",
                   rows(a, fa + "ssm_cache_bf16", fa + "decode_gating_kernel", "FLASHINFER/fp8kv"))]),
        ("SGLang 0.5.22, the same twin", [
            pooled("NGRAM with decoys; also with overlap scheduler off",
                   rows("s26b", "flashinfer/ngram_spec_decoys", "flashinfer/ngram_spec_decoys_nooverlap"))]),
    ]
    slots = sum(len(g[1]) + 1 for g in groups)
    fig = fs.figure(1.35 + 0.255 * slots)
    fs.header(fig, "vLLM #40875 (fixed in 0.32.0): n-gram speculation on a hybrid (Gated DeltaNet) twin",
              "Wrong requests of 192 per entry. vLLM 0.31, Qwen3.5-0.8B twin at full depth (24 layers, 18 of them "
              "Gated DeltaNet), RTX 3090. The same requests fail the same way in every run with 8 wrong; it needs "
              "drafts that are only partly accepted.")
    fs.footer(fig, source("entries"))
    ax = fig.subplots()
    fs.hbar_rows(ax, groups, xmax=24)
    ax.set_xlabel("wrong requests (of 192)", color=C.text2, fontsize=fs.SMALL)
    return fig


@figure
def sglang_ngram():
    headers = {"0.5.21": "0.5.21, one setting changed at a time (RTX 3090 unless noted)",
               "main": "main, bisected from bd78095 to d0a098b, 813 commits",
               "0.5.22": "0.5.22 (pip wheels)"}
    by = collections.OrderedDict()
    for r in read("sglang_ngram"):
        by.setdefault(r["group"], collections.OrderedDict()).setdefault(r["label"], []).append(r)
    groups = []
    for g, labels in by.items():
        items = []
        for label, rs in labels.items():
            vals = sorted({int(r["wrong_requests"]) for r in rs})
            assert len(vals) == 1 and all(r["requests"] == "192" for r in rs), (label, vals)
            if g == "main":
                note = f"{len({r['version'] for r in rs})} commits"
            else:
                note = f"{len(rs)} runs" if len(rs) > 1 else ""
            items.append((label, vals[0], C.engine_bug if vals[0] else C.exact, note))
        groups.append((headers.get(g, g), items))
    slots = sum(len(g[1]) + 1 for g in groups)
    fig = fs.figure(1.4 + 0.255 * slots)
    fs.header(fig, "SGLang #44997: NGRAM speculative decoding",
              "Wrong requests of 192 on the repro workload: 192 concurrent requests sharing prefixes, 256 greedy "
              "tokens each, flashinfer. Requests end by length unless noted. From 0.5.22 on (#42720) the bug "
              "stays hidden when requests end by length; with stop tokens it still shows.")
    fs.footer(fig, source("sglang_ngram") + " (docs/RESULTS.md §3, §3b, §3c)")
    ax = fig.subplots()
    fs.hbar_rows(ax, groups, xmax=10)
    ax.set_xlabel("wrong requests (of 192)", color=C.text2, fontsize=fs.SMALL)
    return fig


@figure
def runtime():
    rows = matrix_entries()
    keys = [(e, g) for e in ENGINES for g in GPUS]
    sel = {k: [r for r in rows if (r["engine"], family(r["gpu"])) == k] for k in keys}
    total = [float(r["init_sec"]) + float(r["gen_sec"]) for r in rows]
    share = sum(float(r["init_sec"]) for r in rows) / sum(total)
    per_req = {int(r["gen_tokens"]) // int(r["requests"]) for r in rows}
    req = sorted(int(r["requests"]) for r in rows)
    assert len(per_req) == 1, per_req
    fig = fs.figure(4.3)
    fs.header(fig, "Time per matrix entry",
              f"Each entry starts an engine (weights, compilation, CUDA graphs) and generates {per_req.pop()} tokens "
              f"for each of {req[0]}-{req[-1]} requests. Median {statistics.median(total):.0f} s; engine start-up is {share:.0%} of the "
              f"time of all {len(rows)} entries ({sum(total) / 3600:.1f} h).")
    fs.footer(fig, source("entries"))
    ax, bx = fig.subplots(1, 2, sharey=True, width_ratios=[1.75, 1])
    ys, labels = [], []
    for i, k in enumerate(keys):
        y = i + (0.7 if k[0] == "sglang" else 0)
        rs = sel[k]
        t = [float(r["init_sec"]) + float(r["gen_sec"]) for r in rs]
        jit = [((j * 0.618034) % 1 - 0.5) * 0.5 for j in range(len(t))]
        ax.scatter(t, [y + d for d in jit], s=9, color=C.text2, alpha=0.4, linewidths=0, zorder=2)
        m = statistics.median(t)
        ax.plot([m, m], [y - 0.34, y + 0.34], color=C.text, lw=2, solid_capstyle="butt", zorder=3)
        ax.annotate(f"{m:.0f} s", (m, y - 0.34), xytext=(0, 1), textcoords="offset points", ha="center",
                    va="bottom", fontsize=fs.SMALL - 1, color=C.text, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor=C.surface, edgecolor="none", alpha=0.85))
        init = statistics.median(float(r["init_sec"]) for r in rs)
        gen = statistics.median(float(r["gen_sec"]) for r in rs)
        bx.barh(y, init, height=0.55, color=fs.CATEGORICAL[0], edgecolor=C.surface, linewidth=1.2)
        bx.barh(y, gen, left=init, height=0.55, color=fs.CATEGORICAL[1], edgecolor=C.surface, linewidth=1.2)
        bx.annotate(f"{init:.0f} + {gen:.0f} s", (init + gen, y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=fs.SMALL, color=C.text)
        ys.append(y)
        labels.append(f"{ENGINES[k[0]]}  {k[1]}  ({len(rs)})")
    ax.set_yticks(ys, labels)
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.set_xlim(0, max(total) * 1.04)
    fs.style_axes(ax, grid="x", left=False)
    fs.style_axes(bx, grid="x", left=False)
    ax.tick_params(axis="y", labelcolor=C.text, labelsize=fs.SMALL)
    ax.set_xlabel("seconds per entry (one dot per entry, line: median; entries in parentheses)", color=C.text2,
                  fontsize=fs.SMALL)
    bx.set_xlabel("median seconds", color=C.text2, fontsize=fs.SMALL)
    bx.set_xlim(0, 130)
    bx.legend([Patch(facecolor=fs.CATEGORICAL[0]), Patch(facecolor=fs.CATEGORICAL[1])],
              ["start-up", "generation"], ncols=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.1,
              handletextpad=0.5, columnspacing=1.0)
    return fig


# the report's experiment figures (real-model comparison, fault boundary, release sweep, bug costs)
import figures_report  # noqa: E402

for _fn in figures_report.FIGURES:
    FIGS[_fn.__name__] = _fn


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help=f"figures to make (default: all of {', '.join(FIGS)})")
    ap.add_argument("--preview", metavar="DIR", help="also write PNG previews of both themes to DIR")
    a = ap.parse_args()
    for name in a.names or FIGS:
        fig = FIGS[name]()
        for p in fs.save(fig, name, preview=a.preview):
            print(p.relative_to(fs.ROOT))
        fs.plt.close(fig)


if __name__ == "__main__":
    main()
