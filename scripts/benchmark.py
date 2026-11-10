"""Run every strategy at several process counts and collect the results for the site.

    .venv/bin/python scripts/benchmark.py            # full run (~10 min on a laptop CPU)
    .venv/bin/python scripts/benchmark.py --quick    # a few steps, to check it works
    .venv/bin/python scripts/benchmark.py --only resume,precision   # redo some parts, keep the rest

Writes runs/<strategy>-<world>/metrics.json and site/data/results.json (+ results.js).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = [("single", 1), ("ddp", 2), ("ddp", 4), ("fsdp", 2), ("fsdp", 4), ("pipeline", 2), ("pipeline", 4)]
# Checkpoint demo: train the first half one way, save, finish under another strategy.
RESUME_FROM, RESUME_TO = ("ddp", 4), ("fsdp", 2)
# Mixed precision: the same loop in bf16, compared with the bf16 single run and with fp32.
PRECISION_RUNS = [("single", 1), ("ddp", 4), ("fsdp", 4), ("pipeline", 4)]
PARTS = ("runs", "resume", "precision")


def run(strategy, world, extra, name=None):
    out = ROOT / "runs" / (name or f"{strategy}-{world}")
    train = ["-m", "disttrain.train", "--strategy", strategy, "--out", str(out), *extra]
    if world == 1:
        cmd = [sys.executable, *train]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={world}", *train]
    print(f"\n=== {strategy} x{world} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads((out / "metrics.json").read_text())


def resume_demo(steps, base_losses):
    ckpt = str(ROOT / "runs" / "resume" / "checkpoint.pt")
    common = ["--microbatches", "8"]
    first = run(*RESUME_FROM, ["--steps", str(steps // 2), "--save", ckpt, *common], "resume-first")
    second = run(*RESUME_TO, ["--steps", str(steps), "--resume", ckpt, *common], "resume-second")
    keep = ("start_step", "losses", "world")
    part = lambda r: {"strategy": r["config"]["strategy"], **{k: r[k] for k in keep}}
    joined = first["losses"] + second["losses"]
    demo = {"first": part(first), "second": part(second),
            "max_diff_vs_single": max(abs(a - b) for a, b in zip(joined, base_losses))}
    print(f"resume demo: {RESUME_FROM} -> {RESUME_TO} at step {second['start_step']}, "
          f"max |diff| vs single {demo['max_diff_vs_single']:.1e}")
    return demo


def precision_runs(steps, fp32_runs):
    # Compared step by step with the start of the fp32 single run; no speed numbers, since
    # bf16 on a CPU without native support says nothing about bf16 on a GPU.
    extra = ["--steps", str(steps), "--microbatches", "8", "--precision", "bf16"]
    runs = [run(s, w, extra, f"bf16-{s}-{w}") for s, w in PRECISION_RUNS]
    base = runs[0]["losses"]
    fp32 = fp32_runs[0]
    for r in runs:
        r["max_diff_vs_single"] = max(abs(a - b) for a, b in zip(r["losses"], base))
        print(f"bf16 {r['config']['strategy']:>8} x{r['world']}  final {r['final_loss']:.4f}  "
              f"max |diff| vs bf16 single {r['max_diff_vs_single']:.1e}  {r['tokens_per_s']:,.0f} tok/s")
    keep = ("world", "losses", "final_loss", "max_diff_vs_single")
    return {
        "steps": steps,
        "runs": [{"strategy": r["config"]["strategy"], **{k: r[k] for k in keep}} for r in runs],
        "max_diff_vs_fp32": max(abs(a - b) for a, b in zip(base, fp32["losses"])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="10 steps per run")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--precision-steps", type=int, default=40,
                   help="steps for the bf16 runs (bf16 is slow on CPUs without native support)")
    p.add_argument("--only", default=",".join(PARTS),
                   help=f"comma-separated parts to (re)run: {', '.join(PARTS)}; "
                        "the others are kept from the existing results.json")
    args = p.parse_args()
    parts = set(args.only.split(","))
    if parts - set(PARTS):
        p.error(f"unknown part(s): {', '.join(parts - set(PARTS))}")
    steps = 10 if args.quick else args.steps
    extra = ["--steps", str(steps), "--microbatches", "8"]
    dest = ROOT / "site" / "data" / "results.json"
    old = json.loads(dest.read_text()) if dest.exists() and parts != set(PARTS) else {}

    if "runs" in parts:
        results = [run(s, w, extra) for s, w in RUNS]
    else:
        results = old["runs"]
        steps = len(results[0]["losses"])
        extra = ["--steps", str(steps), "--microbatches", "8"]
    base = results[0]["losses"]
    resume = resume_demo(steps, base) if "resume" in parts else old.get("resume")
    precision = (precision_runs(min(steps, args.precision_steps), results)
                 if "precision" in parts else old.get("precision"))
    for r in results:
        r["max_diff_vs_single"] = max(abs(a - b) for a, b in zip(r["losses"], base))
        print(f"{r['config']['strategy']:>8} x{r['world']}  final {r['final_loss']:.4f}  "
              f"max |diff| vs single {r['max_diff_vs_single']:.1e}  {r['tokens_per_s']:,.0f} tok/s")

    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"runs": results, "resume": resume, "precision": precision}, indent=1)
    dest.write_text(payload)
    # Same data as a script, so the site also works when opened straight from disk.
    dest.with_suffix(".js").write_text(f"window.RESULTS = {payload};\n")
    print(f"\nwrote {dest.relative_to(ROOT)} and results.js")


if __name__ == "__main__":
    main()
