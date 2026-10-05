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

Pipeline needs at least one transformer block per process: `--n-layers` ≥ processes.

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

About three minutes on a laptop: the slow part launches real `torchrun` jobs and checks that
DDP, FSDP and pipeline (2 and 4 stages) reproduce the single-process loss at every step.

## Benchmark and refresh the site

```bash
.venv/bin/python scripts/benchmark.py          # ~10 minutes
.venv/bin/python scripts/benchmark.py --quick  # 10 steps per run, to check it works
```

This runs the seven configurations shown on the site and rewrites
`site/data/results.json` and `site/data/results.js`.

## View the site

Open `site/index.html` in a browser (it works straight from disk), or serve it:

```bash
.venv/bin/python -m http.server -d site 8000
```

The published copy is at https://amalmehta.github.io/DistributedTraining/.
