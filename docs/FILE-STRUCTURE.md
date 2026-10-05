# Distributed Training Demo — File Structure

```
disttrain/                    the training package
  config.py                   Config dataclass + command-line flags
  data.py                     MarkovData: synthetic batches, rebuilt from (seed, step) on any rank
  model.py                    TinyGPT as a flat layer list: Embed, Block × N, Head
  dist_utils.py               join the process group (Gloo/NCCL) or run as a world of one
  train.py                    entry point and the strategy-agnostic training loop
  strategies/
    base.py                   Strategy interface + memory accounting
    single.py                 one process (the reference)
    ddp.py                    DistributedDataParallel
    fsdp.py                   FullyShardedDataParallel, one shard group per Block
    pipeline.py               hand-written GPipe schedule with send/recv
tests/test_disttrain.py       unit tests + "every strategy matches single" via real torchrun
scripts/benchmark.py          runs 7 configurations, writes site/data/results.{json,js}
site/                         the static website (GitHub Pages serves this folder)
  index.html, style.css       page and styling
  app.js                      animated strategy diagrams + Chart.js charts
  data/results.json, .js      benchmark output (the .js copy lets the page open from disk)
docs/
  INSTRUCTIONS.md             setup, run, test, benchmark
  SYSTEM-DESIGN.md            architecture, flows, decisions, limits
  FILE-STRUCTURE.md           this file
  screenshot.png              site screenshot used in the README
.github/workflows/pages.yml   publishes site/ to GitHub Pages on push to main
distributed_training_pytorch.md   the project brief
pyproject.toml                package metadata, pytest config
runs/                         (git-ignored) per-run metrics.json from the benchmark
```
