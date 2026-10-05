"""Save and resume training from one portable file.

The file holds the whole model and AdamW state under plain TinyGPT names, so a run
saved with one strategy can resume with any other, at any process count — e.g.
save under DDP x4, resume under FSDP x2. Batches are rebuilt from the step
number, so the step is all the data pipeline needs to pick up where it left off.
"""

import os

import torch
import torch.distributed as dist

from .config import Config

FORMAT = 1
# These must match between the saved run and the resuming run; everything else may change.
MODEL_FIELDS = ("vocab_size", "d_model", "n_heads", "n_layers", "seq_len")


def save(path: str, strategy, next_step: int, cfg: Config):
    """Collective: every rank calls it; rank 0 writes. Written atomically (tmp file + rename)."""
    params, optim = strategy.full_state()
    if strategy.ctx.is_main:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save({
            "format": FORMAT,
            "step": next_step,
            "params": params,
            "optim": optim,
            "config": cfg.to_dict(),
            "saved_by": {"strategy": cfg.strategy, "world": strategy.ctx.world},
        }, tmp)
        os.replace(tmp, path)
    if dist.is_initialized():
        dist.barrier()   # nobody runs ahead (or exits) before the file exists


def load(path: str, strategy, cfg: Config):
    """Restore weights and optimizer state on every rank.

    Returns (step to continue from, {"strategy", "world"} that saved it).
    """
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("format") != FORMAT:
        raise SystemExit(f"{path}: unknown checkpoint format {ckpt.get('format')!r}")
    diff = [f for f in MODEL_FIELDS if ckpt["config"][f] != getattr(cfg, f)]
    if diff:
        raise SystemExit(f"{path}: model shape differs in {', '.join(diff)} — pass the same model flags")
    strategy.load_full_state(ckpt["params"], ckpt["optim"])
    return ckpt["step"], ckpt["saved_by"]
