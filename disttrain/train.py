"""Entry point. The training loop below knows nothing about how work is split.

    python -m disttrain.train --strategy single
    torchrun --nproc-per-node 4 -m disttrain.train --strategy ddp
    torchrun --nproc-per-node 4 -m disttrain.train --strategy fsdp
    torchrun --nproc-per-node 4 -m disttrain.train --strategy pipeline --microbatches 8

Stop and resume, even under a different strategy:

    torchrun --nproc-per-node 4 -m disttrain.train --strategy ddp --steps 75 --save ckpt.pt
    torchrun --nproc-per-node 2 -m disttrain.train --strategy fsdp --steps 150 --resume ckpt.pt
"""

import json
import os
import platform
import time

import torch

from . import checkpoint
from .config import Config, parse_args
from .data import MarkovData
from .dist_utils import cleanup, gather_objects, setup
from .model import build_model
from .strategies import get_strategy

WARMUP_STEPS = 5   # left out of the timing numbers


def train(cfg: Config):
    ctx = setup()
    if cfg.global_batch % ctx.world:
        raise SystemExit("--global-batch must be divisible by the number of processes")
    data = MarkovData(cfg.vocab_size, cfg.seq_len, cfg.data_seed)
    strategy = get_strategy(cfg, ctx)
    strategy.setup()
    n_params = sum(p.numel() for p in build_model(cfg).parameters())

    if ctx.is_main:
        print(f"[{cfg.strategy}] world={ctx.world} device={ctx.device} params={n_params:,} "
              f"floor={data.entropy_floor():.3f} nats")

    start_step, resumed_from = 0, None
    if cfg.resume:
        start_step, resumed_from = checkpoint.load(cfg.resume, strategy, cfg)
        if ctx.is_main:
            print(f"  resumed at step {start_step} from {cfg.resume} "
                  f"(saved by {resumed_from['strategy']} x{resumed_from['world']})")
    if start_step >= cfg.steps:
        raise SystemExit(f"checkpoint is already at step {start_step}; raise --steps to train further")

    losses, step_times = [], []
    for step in range(start_step, cfg.steps):
        tokens, targets = data.global_batch(step, cfg.global_batch)
        t0 = time.perf_counter()
        loss = strategy.train_step(tokens, targets)
        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        losses.append(loss)
        if ctx.is_main and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            print(f"  step {step:4d}  loss {loss:.4f}  {step_times[-1] * 1000:6.1f} ms")
        last = step == cfg.steps - 1
        if cfg.save and (last or (cfg.save_every and (step + 1) % cfg.save_every == 0)):
            checkpoint.save(cfg.save, strategy, step + 1, cfg)
            if ctx.is_main:
                print(f"  saved step {step + 1} to {cfg.save}")

    # Every rank reports what it holds and sends; rank 0 writes it all down.
    per_rank = gather_objects({
        "rank": ctx.rank,
        "memory": strategy.memory_report(),
        "comm_bytes_per_step": strategy.comm_bytes_per_step(n_params),
        "layers": list(getattr(strategy, "layer_range", (0, cfg.n_layers + 2))),
    }, ctx)

    timed = step_times[WARMUP_STEPS:] or step_times
    mean_step = sum(timed) / len(timed)
    result = {
        "config": cfg.to_dict(),
        "world": ctx.world,
        "device": str(ctx.device),
        "host": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "n_params": n_params,
        "entropy_floor": data.entropy_floor(),
        "uniform_loss": data.uniform_loss,
        "start_step": start_step,
        "resumed_from": resumed_from,
        "losses": losses,
        "final_loss": losses[-1],
        "mean_step_s": mean_step,
        "tokens_per_s": cfg.global_batch * cfg.seq_len / mean_step,
        "per_rank": per_rank,
    }
    if hasattr(strategy, "bubble_fraction"):
        result["bubble_fraction"] = strategy.bubble_fraction()

    if ctx.is_main:
        print(f"  done: final loss {losses[-1]:.4f}, {result['tokens_per_s']:,.0f} tokens/s")
        if cfg.out:
            os.makedirs(cfg.out, exist_ok=True)
            with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
                json.dump(result, f, indent=1)
    cleanup()
    return result


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
