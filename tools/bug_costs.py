"""The bug timeline and the back-of-envelope cost estimate of the technical report (docs/REPORT.md §4).

Inputs (docs/data):
- bugs.csv: every engine bug probec found or re-caught: the first release known to have it (first_bad: from
  the release sweep, or from git ancestry where the note says so), the first release without it (first_good,
  empty while it is open), whether earlier releases may have it too (start_open), the dates of the first public
  report and of the fix's merge, the share of requests it got wrong in our runs, the GPUs and feature it
  affects, and the workaround a deployment would use until a fix ships.
- release_dates.csv: the PyPI upload date of every vLLM and SGLang release.
- cost_assumptions.csv: per workaround, what it costs (a factor on GPU time, or on decode time) and the source.
- cost_constants.csv: GPU prices per hour, with their sources.
- sweep.csv: the running time of probec's release suite (what the check costs).

Writes docs/data/bug_costs.csv: one row per bug with the days and releases it shipped, the days from its first
public report to a release with the fix, and the estimate.

    python tools/bug_costs.py
"""
import csv
import datetime as dt
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "docs" / "data"
TODAY = dt.date(2026, 10, 19)          # the date of this estimate: open bugs are counted up to it
HOURS_PER_MONTH = 730
NODE_GPUS = 8                           # the unit of the estimate: one 8-GPU node, serving around the clock


def read(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def releases():
    """{engine: [(version, date)]} in release order."""
    out = {}
    for r in read("release_dates.csv"):
        out.setdefault(r["engine"], []).append((r["version"], dt.date.fromisoformat(r["date"])))
    for v in out.values():
        v.sort(key=lambda x: x[1])
    return out


def share(wrong):
    """'8/192' -> 8/192; a range '1-5/192' -> its midpoint."""
    num, den = wrong.split("/")
    lo, _, hi = num.partition("-")
    return (float(lo) + float(hi or lo)) / 2 / float(den)


def constants():
    return {r["key"]: float(r["value"]) for r in read("cost_constants.csv")}


def check():
    """What probec's release suite costs: minutes per release on one RTX 3090 (sweep.csv), and dollars."""
    per = {}
    for r in read("sweep.csv"):
        t = float(r["init_sec"] or 0) + float(r["gen_sec"] or 0)
        per.setdefault(r["engine"], {}).setdefault(r["version"], 0.0)
        per[r["engine"]][r["version"]] += t
    usd = constants()["price:RTX 3090"]
    out = {"gpu_hours": sum(sum(v.values()) for v in per.values()) / 3600,
           "releases": sum(len(v) for v in per.values()), "usd_per_gpu_hour": usd}
    for e, v in per.items():
        out[f"{e}_minutes"] = sum(v.values()) / len(v) / 60
    out["usd_per_release_low"] = min(out[f"{e}_minutes"] for e in per) / 60 * usd
    out["usd_per_release_high"] = max(out[f"{e}_minutes"] for e in per) / 60 * usd
    out["usd_total"] = out["gpu_hours"] * usd
    return out


def estimate():
    rel = releases()
    work = {a["workaround"]: a for a in read("cost_assumptions.csv")}
    price = constants()
    rows = []
    for b in read("bugs.csv"):
        eng = b["engine"].lower()
        dates = dict(rel[eng])
        start = dates[b["first_bad"]]
        end = dates[b["first_good"]] if b["first_good"] else TODAY
        shipped = [v for v, d in rel[eng] if start <= d < end]
        a = work[b["workaround"]]
        node_month = NODE_GPUS * price["price:" + b["gpus"]] * HOURS_PER_MONTH
        lo, hi = float(a["factor_low"] or 1), float(a["factor_high"] or 1)
        months = (end - start).days / (365.25 / 12)
        usd = a["metric"] == "gpus"          # a cost in GPUs (dollars); "speed" is a cost in decode time
        reported = dt.date.fromisoformat(b["reported"]) if b["reported"] else None
        rows.append({
            "id": b["id"], "engine": b["engine"], "feature": b["feature"], "label": b["label"],
            "first_bad": b["first_bad"], "first_good": b["first_good"] or "(open)", "start": start.isoformat(),
            "end": end.isoformat(), "open": int(not b["first_good"]), "start_open": b["start_open"],
            "days": (end - start).days, "releases": len(shipped),
            "reported": b["reported"], "days_report_to_release": (end - reported).days if reported else "",
            "days_before_report": max(0, ((reported or end) - start).days),
            "wrong": b["wrong"], "wrong_share": round(share(b["wrong"]), 4), "gpus": b["gpus"],
            "workaround": a["description"], "metric": a["metric"], "factor_low": lo, "factor_high": hi,
            "node_usd_per_month": round(node_month),
            "extra_usd_per_node_month_low": round((lo - 1) * node_month) if usd else "",
            "extra_usd_per_node_month_high": round((hi - 1) * node_month) if usd else "",
            "extra_usd_per_node_while_shipped_low": round((lo - 1) * node_month * months) if usd else "",
            "extra_usd_per_node_while_shipped_high": round((hi - 1) * node_month * months) if usd else "",
        })
    return rows


def main():
    rows = estimate()
    with open(DATA / "bug_costs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {DATA / 'bug_costs.csv'}: {len(rows)} bugs")
    c = check()
    print(f"check: {c['releases']} releases, {c['gpu_hours']:.1f} GPU-hours, vLLM {c['vllm_minutes']:.0f} min and SGLang "
          f"{c['sglang_minutes']:.0f} min per release, ${c['usd_per_release_low']:.2f}-{c['usd_per_release_high']:.2f} "
          f"per release, ${c['usd_total']:.2f} in all")
    for r in rows:
        cost = (f"${r['extra_usd_per_node_month_low']:,}-{r['extra_usd_per_node_month_high']:,}/node-month"
                if r["metric"] == "gpus" else f"{r['factor_low']}-{r['factor_high']}x {r['metric']}")
        print(f"{r['label']:45s} {r['first_bad']:>7s} -> {r['first_good']:8s} {r['days']:4d} d {r['releases']:3d} rel"
              f"  report->release {r['days_report_to_release']!s:>4}  silent {r['days_before_report']:4d}  "
              f"wrong {r['wrong']:>8s}  {cost}")


if __name__ == "__main__":
    main()
