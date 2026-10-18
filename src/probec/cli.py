"""probec command line: compile probes, certify them, check them on CPU, and run them against engines."""
import argparse
import json
import sys


def _spec(a):
    from .spec import builtin, from_config, load_config
    if a.like:
        cfg, src = load_config(a.like)
        return from_config(cfg, layers=a.layers, source=src)
    return builtin(a.builtin or "tiny-llama", layers=a.layers, vocab=a.vocab)


def cmd_compile(a):
    from .compile import compile_probe
    spec = _spec(a)
    extra = {"scale_check": False} if a.no_scale_check else {}
    meta = compile_probe(spec, a.out, program=a.program, classes=a.classes, fillers=a.fillers,
                         pools=a.pools, max_corr=a.max_corr, rope_theta=a.rope_theta, seed=a.seed, **extra)
    print(f"probec: wrote {a.out}: {spec.arch} L={spec.layers} hidden={spec.hidden} "
          f"heads={spec.heads}/{spec.kv_heads} vocab={spec.vocab} program={meta['program']}")
    print(f"probec: margins {json.dumps(meta.get('margins', {}))}  compile {meta['compile_sec']}s")


def cmd_certify(a):
    from .certify import certify
    from .quant import WEIGHT_FORMATS
    quant = list(WEIGHT_FORMATS) if a.quant == "all" else [f for f in (a.quant or "").split(",") if f]
    rep = certify(a.probe, quant=quant)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def cmd_check(a):
    if a.quant_act or a.quant_weight:
        from .quant import check_quant
        rep = check_quant(a.probe, act=a.quant_act, weight=a.quant_weight, device=a.device, seed=a.seed)
        print(json.dumps(rep))
        return 0 if rep["mismatches"] == 0 else 1
    from .check import run_checks
    rep = run_checks(a.probe, seed=a.seed, quick=a.quick)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    zfa = rep["zero_false_alarm"]
    print(f"probec check: {sum(z['tokens'] for z in zfa)} tokens, "
          f"{sum(z['mismatches'] for z in zfa)} mismatches; bitwise "
          f"{sum(b['layer_states_checked'] - b['non_bitwise'] for b in rep['bitwise_internal'])}/"
          f"{sum(b['layer_states_checked'] for b in rep['bitwise_internal'])} layer states; "
          f"{'OK' if rep['ok'] else 'FAIL'}")
    return 0 if rep["ok"] else 1


def cmd_report(a):
    rows = []
    for fn in a.results:
        with open(fn) as f:
            rows += [json.loads(l) for l in f if l.strip()]
    print("| engine | program | config | generated tokens | mismatched | notes |")
    print("|---|---|---|---|---|---|")
    tot = bad = 0
    for r in rows:
        note = ""
        if "error" in r:
            note = "error: " + r["error"][:80].replace("|", "/")
        elif r.get("diagnoses"):
            note = f"diagnoses {r['diagnoses']}"
        elif r.get("failures"):
            f0 = r["failures"][0]
            note = f"{r.get('mismatched_requests', '?')} requests; first bad step {f0['first_bad_step']} (prompt {f0['prompt_len']} tok)"
        print(f"| {r.get('engine', '?')} | {r.get('program', '?')} | {r['config']} | {r.get('gen_tokens', 0)} | "
              f"{r.get('mismatched_tokens', '-')} | {note} |")
        tot += r.get("gen_tokens", 0)
        bad += r.get("mismatched_tokens", 0) or 0
    print(f"\n{len(rows)} configurations, {tot} generated tokens checked, {bad} mismatched")


def cmd_quantize(a):
    from .quantize import quantize_twin
    amax = None
    if a.format == "nvfp4":                 # static activation scales come from a calibration pass
        from .quant import calibrate_input_amax
        amax = calibrate_input_amax(a.probe, device=a.device)
    quantize_twin(a.probe, a.out, a.format, allow_inexact=a.allow_inexact, input_amax=amax,
                  nvfp4_scales=a.nvfp4_scales)
    return 0


def cmd_run(a):
    from .harness import run_engine
    return run_engine(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="probec", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="compile a probe checkpoint")
    src = c.add_mutually_exclusive_group()
    src.add_argument("--like", help="HF model id or path to a config.json: build a probe twin of it")
    src.add_argument("--builtin", help="built-in shape: tiny-llama, tiny-qwen3, tiny-qwen3-moe, tiny-deepseek-v3, "
                     "tiny-deepseek-v32, qwen3-0.6b-shape")
    c.add_argument("--layers", type=int, help="number of layers (default: the model's own)")
    c.add_argument("--vocab", type=int, help="vocab size override for built-in shapes")
    c.add_argument("--program", default="recall",
                   help="recall (default; hybrid for hybrid architectures), posrecall, moe, mla or hybrid")
    c.add_argument("--classes", type=int)
    c.add_argument("--fillers", type=int)
    c.add_argument("--pools", help="comma-separated pool (cycle) sizes summing to --classes")
    c.add_argument("--max-corr", type=int)
    c.add_argument("--rope-theta", type=float)
    c.add_argument("--seed", type=int)
    c.add_argument("--no-scale-check", action="store_true", help="recall, moe, mla: omit the softmax-scale check")
    c.add_argument("-o", "--out", required=True)
    c.set_defaults(fn=cmd_compile)

    c = sub.add_parser("certify", help="exactness report for a compiled probe")
    c.add_argument("probe")
    c.add_argument("--quant", help="also check that the weights quantize exactly: comma-separated formats "
                                   "(fp8, fp8-block, int4-awq, int4-gptq, nvfp4, mxfp4) or 'all'")
    c.set_defaults(fn=cmd_certify)

    c = sub.add_parser("check", help="CPU checks with HF transformers (reference semantics)")
    c.add_argument("probe")
    c.add_argument("--quick", action="store_true")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out")
    c.add_argument("--quant-act", help="instead: generate with linear inputs fake-quantized "
                                       "(fp8-tensor, fp8-token, fp8-group128, nvfp4, mxfp4)")
    c.add_argument("--quant-weight", help="instead: generate with dequantized weights of this format")
    c.add_argument("--device", default="cpu", help="device for --quant-act / --quant-weight runs")
    c.set_defaults(fn=cmd_check)

    c = sub.add_parser("quantize", help="write a quantized copy of a compiled probe in an engine checkpoint format")
    c.add_argument("probe")
    c.add_argument("--format", required=True, help="fp8 (per-tensor scales), fp8-block (128x128 block scales), "
                                                   "int4-awq, int4-gptq or nvfp4 (ModelOpt; calibrates input scales)")
    c.add_argument("--device", default="cpu", help="device for the nvfp4 calibration pass")
    c.add_argument("--nvfp4-scales", choices=["unit", "modelopt"], default="unit",
                   help="nvfp4: weight_scale_2 = 1 with exact power-of-two block scales (unit), or ModelOpt's "
                        "amax / (6 * 448) with block scales near 448 (modelopt)")
    c.add_argument("-o", "--out", required=True)
    c.add_argument("--allow-inexact", action="store_true", help="write even if some weights would not be exact")
    c.set_defaults(fn=cmd_quantize)

    c = sub.add_parser("run", help="run a probe through an inference engine and compare with the oracle")
    c.add_argument("probe")
    c.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    c.add_argument("--matrix", default="default", help="quick | medium | default | full | hopper | ep2 | replay | mladebug (see --list)")
    c.add_argument("--only", default="", help="run a single matrix entry by name")
    c.add_argument("--list", action="store_true", help="list matrix entries and exit")
    c.add_argument("--requests", type=int, default=192)
    c.add_argument("--new-tokens", type=int, default=256)
    c.add_argument("--max-len", type=int, default=16000)
    c.add_argument("--fault-requests", type=int, default=64)
    c.add_argument("--out", default="out/results.jsonl")
    c.add_argument("--strict", action="store_true", help="exit with status 2 if any token mismatched (for CI)")
    c.add_argument("--in-process", action="store_true",
                   help="run every entry in this process (by default each runs in its own: SGLang kills its whole "
                        "process tree when a scheduler fails, and engines torn down in-process can keep GPU memory)")
    c.add_argument("--entry-timeout", type=int, default=1800,
                   help="seconds allowed per entry when entries run in their own processes")
    c.set_defaults(fn=cmd_run)

    c = sub.add_parser("report", help="summarize result .jsonl files as a markdown table")
    c.add_argument("results", nargs="+")
    c.set_defaults(fn=cmd_report)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
