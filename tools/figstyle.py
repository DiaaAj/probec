"""Shared matplotlib style for probec's report figures (README, docs/RESULTS.md).

Draw each figure once, in the light theme, with the colors defined here (C.text, C.engine_bug, PROGRAM["mla"],
...). save(fig, name) then writes two SVGs with transparent backgrounds:

    docs/figures/<name>.svg        light theme: dark text, for GitHub's white page
    docs/figures/<name>-dark.svg   every color of this palette swapped for its dark-theme twin (light text)

and the README picks one with prefers-color-scheme:

    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/figures/coverage-dark.svg">
      <img alt="..." src="docs/figures/coverage.svg">
    </picture>

A color that is not in the palette is left alone in the dark variant, so take every color from here.

Sizes: figures are WIDTH (8 in = 576 pt) wide, which GitHub shows at 768 CSS px, about the width of a README
column; text is 8-12 pt (11-16 px there). Text is stored as paths in DejaVu Sans (bundled with matplotlib), so
the SVGs look the same in every browser, and with the same matplotlib version they regenerate byte for byte.

Colors (checked with the dataviz skill's validator against GitHub's #ffffff and #0d1117 pages):
- outcomes (C.exact, C.engine_bug, C.probe_bug, C.error, C.unexplained) keep one meaning in every figure;
  all pairs separate under protanopia and deuteranopia (OKLab dE >= 9.1) and in normal vision (dE >= 16.6).
  exact, error and unexplained are neutral on purpose: what is fine recedes, an engine bug stands out.
- CATEGORICAL is for identity (programs, GPUs, ...), assigned in this order and never cycled. It leaves out red
  and orange, which mean "engine bug". Adjacent slots pass the CVD checks (stacked bars, lines); in scatter-like
  forms where any two marks can meet, use at most the first three. Slots 2-4 are below 3:1 on white: label them.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, transforms  # noqa: E402
from matplotlib.collections import Collection  # noqa: E402
from matplotlib.colors import to_hex, to_rgb, to_rgba  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
PAGE = {"light": "#ffffff", "dark": "#0d1117"}   # GitHub's page backgrounds (used by previews and C.surface)

# role: (light, dark)
_ROLES = {
    "text": ("#1f2328", "#e6edf3"),     # titles, labels, values
    "text2": ("#59636e", "#9198a1"),    # subtitles, tick labels, notes
    "axis": ("#8c959f", "#656c76"),     # axis lines and ticks
    "grid": ("#e1e4e8", "#262c36"),     # hairline grid
    "surface": ("#ffffff", "#0d1117"),  # the page: thin gaps between touching marks
    "on_fill": ("#fcfcfc", "#fcfcfc"),  # text inside a saturated fill
    "exact": ("#7f9187", "#788a7f"),
    "engine_bug": ("#d6402a", "#ec5d3c"),
    "probe_bug": ("#2a78d6", "#3987e5"),
    "error": ("#cfd2d4", "#3f454d"),
    "unexplained": ("#1f2328", "#e6edf3"),
    # diagram cards (tools/figures_report.overview): neutral, probec's own, and the two outcomes
    "card": ("#f6f8fa", "#151b23"),
    "card_edge": ("#d1d9e0", "#3d444d"),
    "accent_soft": ("#eaf2fc", "#122039"),
    "exact_soft": ("#eef2f0", "#172019"),
    "bug_soft": ("#fcece9", "#341a17"),
}
_CATEGORICAL = [("#2a78d6", "#3987e5"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),
                ("#e87ba4", "#d55181"), ("#6250d6", "#9085e9"), ("#008300", "#008300")]

C = SimpleNamespace(**{k: v[0] for k, v in _ROLES.items()})
CATEGORICAL = [light for light, _ in _CATEGORICAL]
OUTCOME = {"exact": C.exact, "engine_bug": C.engine_bug, "probe_bug": C.probe_bug, "error": C.error,
           "unexplained": C.unexplained}
OUTCOME_LABEL = {"exact": "exact", "engine_bug": "engine bug", "probe_bug": "probe bug (fixed)",
                 "error": "error (no tokens)", "unexplained": "unexplained"}
PROGRAMS = ["recall", "posrecall", "moe", "mla", "hybrid"]
PROGRAM = dict(zip(PROGRAMS, CATEGORICAL))

_DARK = {}
for _light, _dark in list(_ROLES.values()) + _CATEGORICAL:
    assert _DARK.setdefault(to_hex(_light), to_hex(_dark)) == to_hex(_dark), f"{_light} has two dark twins"

WIDTH = 8.0          # inches; 576 pt = 768 CSS px
FONT = 10            # body text, labels, values
SMALL = 9            # ticks, notes, legends
TITLE = 12
PREVIEW_DPI = 96     # CSS pixels: previews show the README size
PAD = 0.10           # inches between the figure edge and its title, subtitle, footer and group headers


def apply():
    """Set matplotlib's rcParams to this style (figure() calls it)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT, "text.color": C.text,
        "axes.labelcolor": C.text, "axes.labelsize": FONT, "axes.titlesize": FONT, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 6, "axes.edgecolor": C.axis, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "none", "axes.grid": False,
        "axes.axisbelow": True, "axes.prop_cycle": matplotlib.cycler(color=CATEGORICAL),
        "grid.color": C.grid, "grid.linewidth": 0.8,
        "xtick.color": C.axis, "ytick.color": C.axis, "xtick.labelcolor": C.text2, "ytick.labelcolor": C.text2,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "patch.linewidth": 0,
        "legend.frameon": False, "legend.fontsize": SMALL, "legend.handlelength": 1.0, "legend.handleheight": 0.8,
        "legend.borderaxespad": 0.2, "legend.labelcolor": C.text2,
        "figure.facecolor": "none", "figure.dpi": PREVIEW_DPI, "savefig.transparent": True,
        "svg.fonttype": "path", "svg.hashsalt": "probec-figures",
    })


def figure(height, width=WIDTH, **kw):
    """A new figure, `height` inches tall, with constrained layout. Add header()/footer() before saving."""
    apply()
    fig = plt.figure(figsize=(width, height), layout="constrained", **kw)
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.04, hspace=0.04)
    fig._probec_bands = [0.0, 0.0]   # inches reserved at the top (header) and bottom (footer)
    return fig


def _reserve(fig):
    h = fig.get_figheight()
    top, bottom = fig._probec_bands
    fig.get_layout_engine().set(rect=(0, bottom / h, 1, 1 - (top + bottom) / h))


def text_width(fig, s, size=FONT, weight="normal"):
    """Width of one line of text in inches, as the renderer draws it."""
    prop = font_manager.FontProperties(family="DejaVu Sans", size=size, weight=weight)
    w, _, _ = fig.canvas.get_renderer().get_text_width_height_descent(s, prop, ismath=False)
    return w / fig.dpi


def wrap(fig, s, size, width):
    """Break s into lines no wider than `width` inches (newlines in s are kept)."""
    out = []
    for para in s.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = f"{line} {word}" if line else word
            if line and text_width(fig, cand, size) > width:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return "\n".join(out)


def header(fig, title, subtitle=None):
    """A left-aligned title and an optional subtitle above the plots; the subtitle wraps to the figure width."""
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.text(PAD / w, 1 - PAD / h, title, ha="left", va="top", fontsize=TITLE, fontweight="bold", color=C.text)
    used = PAD + TITLE / 72 * 1.3
    if subtitle:
        subtitle = wrap(fig, subtitle, SMALL, w - 2 * PAD)
        used += 0.05
        fig.text(PAD / w, 1 - used / h, subtitle, ha="left", va="top", fontsize=SMALL, color=C.text2,
                 linespacing=1.35)
        used += SMALL / 72 * 1.35 * (subtitle.count("\n") + 1)
    fig._probec_bands[0] = used + 0.08
    _reserve(fig)


def strip_axes(fig, height):
    """An axes `height` inches tall right under the header, as wide as the figure (outside the layout grid):
    for a summary bar above the main plot. Call after header()."""
    w, h = fig.get_figwidth(), fig.get_figheight()
    top = fig._probec_bands[0]
    ax = fig.add_axes((PAD / w, 1 - (top + height) / h, 1 - 2 * PAD / w, height / h))
    ax.set_in_layout(False)
    fig._probec_bands[0] = top + height + 0.08
    _reserve(fig)
    return ax


def footer(fig, text):
    """A small source line at the bottom left (where the data comes from)."""
    pad = 0.08
    fig.text(PAD / fig.get_figwidth(), pad / fig.get_figheight(), text, ha="left", va="bottom",
             fontsize=SMALL - 1, color=C.text2)
    fig._probec_bands[1] = pad + (SMALL - 1) / 72 * 1.3 + 0.04
    _reserve(fig)


def style_axes(ax, grid="x", left=True):
    """Minimal chrome: no top/right spines, hairline grid on one axis (or none), no y ticks for bar rows."""
    if grid:
        ax.grid(True, axis=grid, color=C.grid, linewidth=0.8)
    if not left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    return ax


def hbar_rows(ax, groups, xmax=None, height=0.62, header_gap=0.25):
    """Horizontal bars in titled groups, top to bottom. groups: [(header, rows)]; header None starts the group
    right below the previous one, "" after a gap, any other text after a gap under that bold title (at the
    figure's left edge, so it can be wider than the plot). Each row is (label, value, color, note): the label
    goes left of the axis, the value (and the note, muted) right of the bar. A value of 0 is drawn as a dot in
    C.exact so it reads as a measured zero, not a missing bar. Returns the number of row slots used."""
    y, ticks, labels = 0.0, [], []
    xmax = xmax or max(r[1] for _, rows in groups for r in rows) or 1
    left_edge = transforms.blended_transform_factory(ax.figure.transFigure, ax.transData)
    for header, rows in groups:
        if header is not None and y:
            y += header_gap
        if header:
            ax.text(PAD / ax.figure.get_figwidth(), y + 0.1, header, transform=left_edge, ha="left", va="center",
                    fontsize=SMALL, fontweight="bold", color=C.text, in_layout=False)
            y += 1
        for label, value, color, note in rows:
            if value:
                ax.barh(y, value, height=height, color=color, edgecolor=C.surface, linewidth=0)
            else:
                ax.plot([0], [y], "o", ms=5.5, color=C.exact, mec=C.surface, mew=1.2, clip_on=False, zorder=3)
            text = f"{value:g}"
            ax.annotate(text, (value, y), xytext=(6 if value else 7, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=SMALL, color=C.text, annotation_clip=False)
            if note:
                skip = 6 + text_width(ax.figure, text, SMALL) * 72 + 5
                ax.annotate(note, (value, y), xytext=(skip, 0), textcoords="offset points", va="center",
                            ha="left", fontsize=SMALL, color=C.text2, annotation_clip=False)
            ticks.append(y)
            labels.append(label)
            y += 1
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, xmax)
    style_axes(ax, grid="x", left=False)
    ax.tick_params(axis="y", labelcolor=C.text, labelsize=SMALL)
    return y


def _recolor(fig, table):
    """Swap every palette color for its twin in `table`; return the setters that undo it."""
    undo, seen = [], set()

    def twin(color):
        r, g, b, a = to_rgba(color)
        new = table.get(to_hex((r, g, b))) if a > 0 else None
        return None if new is None else (*to_rgb(new), a)

    def swap(get, put):
        old = get()
        if isinstance(old, str) and old in ("none", "auto"):
            return
        if getattr(old, "ndim", 1) == 2:   # a collection's per-element colors
            new = [twin(c) or tuple(c) for c in old]
            if any(twin(c) for c in old):
                undo.append((put, old.copy()))
                put(new)
            return
        new = twin(old)
        if new is not None:
            undo.append((put, old))
            put(new)

    def visit(a):
        if id(a) in seen:
            return
        seen.add(id(a))
        if isinstance(a, Text):
            swap(a.get_color, a.set_color)
            if a.get_bbox_patch() is not None:
                visit(a.get_bbox_patch())
            if isinstance(a, Annotation) and a.arrow_patch is not None:
                visit(a.arrow_patch)
        elif isinstance(a, Line2D):
            swap(a.get_color, a.set_color)
            swap(a.get_markerfacecolor, a.set_markerfacecolor)
            swap(a.get_markeredgecolor, a.set_markeredgecolor)
        elif isinstance(a, Patch):
            swap(a.get_facecolor, a.set_facecolor)
            swap(a.get_edgecolor, a.set_edgecolor)
            if hasattr(a, "get_hatchcolor"):
                swap(a.get_hatchcolor, a.set_hatchcolor)
        elif isinstance(a, Collection):
            swap(a.get_facecolor, a.set_facecolor)
            swap(a.get_edgecolor, a.set_edgecolor)

    for artist in fig.findobj():
        visit(artist)
    return undo


def save(fig, name, outdir=None, preview=None):
    """Write docs/figures/<name>.svg (light) and <name>-dark.svg (dark), transparent. With `preview` (a
    directory, or the PROBEC_FIG_PREVIEW environment variable) also write PNGs of both on GitHub's page colors
    at README size, for checking layout by eye; previews do not belong in the repository."""
    outdir = Path(outdir or FIGURES)
    outdir.mkdir(parents=True, exist_ok=True)
    preview = preview or os.environ.get("PROBEC_FIG_PREVIEW")
    meta = {"Date": None, "Creator": "probec tools/figstyle.py"}
    paths = [outdir / f"{name}.svg", outdir / f"{name}-dark.svg"]
    # both SVGs before any PNG: a raster draw nudges the layout, and the SVGs must not depend on previews
    fig.savefig(paths[0], transparent=True, metadata=meta)
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
    undo = _recolor(fig, _DARK)
    try:
        fig.savefig(paths[1], transparent=True, metadata=meta)
        if preview:
            fig.savefig(Path(preview) / f"{name}-dark.png", dpi=PREVIEW_DPI, facecolor=PAGE["dark"])
    finally:
        for put, old in reversed(undo):
            put(old)
    if preview:
        fig.savefig(Path(preview) / f"{name}.png", dpi=PREVIEW_DPI, facecolor=PAGE["light"])
    return paths
