"""Process-group setup that works the same under `python` and `torchrun`."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class Context:
    rank: int
    world: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def setup() -> Context:
    """Join the process group if launched by torchrun; otherwise run as a world of one.

    GPUs present -> NCCL and one GPU per process. Otherwise -> Gloo on CPU.
    """
    if "RANK" not in os.environ:
        return Context(0, 1, 0, torch.device("cpu"))
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device, backend = torch.device("cuda", local_rank), "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
        # Split the CPU cores between the processes on this machine so they don't fight.
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // local_world))
    dist.init_process_group(backend)
    return Context(rank, world, local_rank, device)


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def gather_objects(obj, ctx: Context):
    """Every rank's `obj`, in rank order (a list of one when not distributed)."""
    if ctx.world == 1:
        return [obj]
    out = [None] * ctx.world
    dist.all_gather_object(out, obj)
    return out
