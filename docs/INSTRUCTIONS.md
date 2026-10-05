# Distributed Training Demo — Instructions

## Setup

Needs Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python torch "numpy<2" pytest
```

On an Intel Mac the newest PyTorch available is 2.2.2; uv picks it automatically. Apple Silicon,
Linux and Windows get the current release. No GPU is needed.

## Train

The same command with a different `--strategy`. Distributed strategies are launched with
`torchrun`, which starts one process per rank and sets `RANK`, `WORLD_SIZE` and friends.

```bash
.venv/bin/python -m disttrain.train --strategy single
.venv/bin/torchrun --nproc-per-node 4 -m disttrain.train --strategy ddp
.venv/bin/torchrun --nproc-per-node 4 -m disttrain.train --strategy fsdp
.venv/bin/torchrun --nproc-per-node 4 -m disttrain.train --strategy pipeline --microbatches 8
```

Every run prints the loss every 10 steps and, with `--out some/folder`, writes
`some/folder/metrics.json` (losses, timing, memory and communication per rank).

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--strategy` | `single` | `single`, `ddp`, `fsdp` or `pipeline` |
| `--steps` | 150 | optimizer steps |
| `--global-batch` | 32 | sequences per step, summed over all ranks (must divide by the process count) |
| `--microbatches` | 4 | pipeline only: pieces each batch is cut into (must divide the batch) |
| `--d-model`, `--n-heads`, `--n-layers`, `--seq-len`, `--vocab-size` | 128, 4, 4, 64, 64 | model size |
| `--lr`, `--weight-decay` | 3e-3, 0.01 | AdamW settings |
| `--seed`, `--data-seed` | 0, 1234 | model init, data |
| `--log-every` | 10 | print interval |
| `--out` | (none) | folder to write `metrics.json` into |
| `--save` | (none) | checkpoint file to write at the end of the run |
| `--save-every` | 0 | also save every N steps (0 = only at the end) |
| `--resume` | (none) | checkpoint file to continue from |

Pipeline needs at least one transformer block per process: `--n-layers` ≥ processes.

### Save and resume

`--save` writes one checkpoint file (weights, AdamW state and the step number). `--resume`
continues from it up to `--steps`. The file is the same whatever strategy wrote it, so you can
switch strategy or process count in between:

```bash
.venv/bin/torchrun --nproc-per-node 4 -m disttrain.train --strategy ddp --steps 75 --save ckpt.pt
.venv/bin/torchrun --nproc-per-node 2 -m disttrain.train --strategy fsdp --steps 150 --resume ckpt.pt
```

Use `--save-every 25` to keep a recent checkpoint during a long run (each save replaces the last).
The model flags (`--vocab-size`, `--d-model`, `--n-heads`, `--n-layers`, `--seq-len`) must match
the saved run; batch size, learning rate and the rest may change. With several machines, the
file has to be on storage every machine can read.

### GPUs and multiple machines

If CUDA is available, each process takes GPU `LOCAL_RANK` and the NCCL backend; otherwise
Gloo on CPU. To span machines, run on each node:

```bash
torchrun --nnodes 2 --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint HOST:29500 \
  -m disttrain.train --strategy fsdp
```

(The demo has only been run on CPU; see Known limits in [SYSTEM-DESIGN.md](SYSTEM-DESIGN.md).)

## Test

```bash
.venv/bin/python -m pytest
```

About five minutes on an idle laptop: the slow part launches real `torchrun` jobs and checks that
DDP, FSDP and pipeline (2 and 4 stages) reproduce the single-process loss at every step, and that
a run saved under one strategy resumes correctly under another.

## Benchmark and refresh the site

```bash
.venv/bin/python scripts/benchmark.py          # ~10 minutes
.venv/bin/python scripts/benchmark.py --quick  # 10 steps per run, to check it works
```

`--resume-only` keeps the existing runs and redoes just the checkpoint demo (DDP ×4 for half
the steps, saved, then finished under FSDP ×2).

This runs the seven configurations shown on the site, plus the checkpoint demo, and rewrites
`site/data/results.json` and `site/data/results.js`.

## View the site

Open `site/index.html` in a browser (it works straight from disk), or serve it:

```bash
.venv/bin/python -m http.server -d site 8000
```

The published copy is at https://amalmehta.github.io/DistributedTraining/.
