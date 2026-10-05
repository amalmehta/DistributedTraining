"""Run every strategy at several process counts and collect the results for the site.

    .venv/bin/python scripts/benchmark.py            # full run (~10 min on a laptop CPU)
    .venv/bin/python scripts/benchmark.py --quick    # a few steps, to check it works

Writes runs/<strategy>-<world>/metrics.json and site/data/results.json (+ results.js).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = [("single", 1), ("ddp", 2), ("ddp", 4), ("fsdp", 2), ("fsdp", 4), ("pipeline", 2), ("pipeline", 4)]


def run(strategy, world, extra):
    out = ROOT / "runs" / f"{strategy}-{world}"
    train = ["-m", "disttrain.train", "--strategy", strategy, "--out", str(out), *extra]
    if world == 1:
        cmd = [sys.executable, *train]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               f"--nproc-per-node={world}", *train]
    print(f"\n=== {strategy} x{world} ===", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads((out / "metrics.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="10 steps per run")
    p.add_argument("--steps", type=int, default=150)
    args = p.parse_args()
    extra = ["--steps", str(10 if args.quick else args.steps), "--microbatches", "8"]

    results = [run(s, w, extra) for s, w in RUNS]
    base = results[0]["losses"]
    for r in results:
        r["max_diff_vs_single"] = max(abs(a - b) for a, b in zip(r["losses"], base))
        print(f"{r['config']['strategy']:>8} x{r['world']}  final {r['final_loss']:.4f}  "
              f"max |diff| vs single {r['max_diff_vs_single']:.1e}  {r['tokens_per_s']:,.0f} tok/s")

    dest = ROOT / "site" / "data" / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"runs": results}, indent=1)
    dest.write_text(payload)
    # Same data as a script, so the site also works when opened straight from disk.
    dest.with_suffix(".js").write_text(f"window.RESULTS = {payload};\n")
    print(f"\nwrote {dest.relative_to(ROOT)} and results.js")


if __name__ == "__main__":
    main()
