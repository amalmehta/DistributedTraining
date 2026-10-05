"""Run every strategy at several process counts and collect the results for the site.

    .venv/bin/python scripts/benchmark.py            # full run (~10 min on a laptop CPU)
    .venv/bin/python scripts/benchmark.py --quick    # a few steps, to check it works
    .venv/bin/python scripts/benchmark.py --resume-only   # redo just the checkpoint demo

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="10 steps per run")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--resume-only", action="store_true",
                   help="keep the existing runs in results.json, redo only the checkpoint demo")
    args = p.parse_args()
    steps = 10 if args.quick else args.steps
    extra = ["--steps", str(steps), "--microbatches", "8"]
    dest = ROOT / "site" / "data" / "results.json"

    if args.resume_only:
        results = json.loads(dest.read_text())["runs"]
        steps = len(results[0]["losses"])
    else:
        results = [run(s, w, extra) for s, w in RUNS]
    base = results[0]["losses"]
    resume = resume_demo(steps, base)
    for r in results:
        r["max_diff_vs_single"] = max(abs(a - b) for a, b in zip(r["losses"], base))
        print(f"{r['config']['strategy']:>8} x{r['world']}  final {r['final_loss']:.4f}  "
              f"max |diff| vs single {r['max_diff_vs_single']:.1e}  {r['tokens_per_s']:,.0f} tok/s")

    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"runs": results, "resume": resume}, indent=1)
    dest.write_text(payload)
    # Same data as a script, so the site also works when opened straight from disk.
    dest.with_suffix(".js").write_text(f"window.RESULTS = {payload};\n")
    print(f"\nwrote {dest.relative_to(ROOT)} and results.js")


if __name__ == "__main__":
    main()
