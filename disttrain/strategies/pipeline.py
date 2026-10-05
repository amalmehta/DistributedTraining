"""Pipeline parallel (GPipe schedule), written out by hand with send/recv.

The layer list is cut into W consecutive stages, one per rank. The batch is cut
into M microbatches. Forward: each stage receives activations from the stage
before, runs its layers and sends the result on. Backward runs the same path in
reverse, passing activation gradients back. Gradients add up over the M
microbatches, then every stage steps its own optimizer.

Stages sit idle while the pipe fills and drains — the "bubble" — which is a
fraction (W-1)/(M+W-1) of each step. More microbatches, smaller bubble.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

from ..model import build_model, lm_loss
from .base import Strategy


def stage_bounds(n_blocks: int, world: int):
    """[lo, hi) indices into (embed, block_1..block_n, head) for each stage.

    Blocks are spread as evenly as possible; the embedding rides with the first
    stage and the head with the last.
    """
    if n_blocks < world:
        raise ValueError(f"pipeline needs at least one block per stage ({n_blocks} blocks, {world} stages)")
    sizes = [n_blocks // world + (1 if i < n_blocks % world else 0) for i in range(world)]
    bounds, start = [], 1
    for i, s in enumerate(sizes):
        lo = 0 if i == 0 else start
        hi = start + s + (1 if i == world - 1 else 0)
        bounds.append((lo, hi))
        start += s
    return bounds


class PipelineStrategy(Strategy):
    name = "pipeline"

    def setup(self):
        cfg, ctx = self.cfg, self.ctx
        if cfg.global_batch % cfg.microbatches:
            raise SystemExit("--global-batch must be divisible by --microbatches")
        full = build_model(cfg)   # build everything with the shared seed, keep only our slice
        lo, hi = stage_bounds(cfg.n_layers, ctx.world)[ctx.rank]
        self.layer_range = (lo, hi)
        self.model = nn.Sequential(*full.layers[lo:hi]).to(ctx.device)
        self.optimizer = self.make_optimizer(self.model.parameters())
        self.first, self.last = ctx.rank == 0, ctx.rank == ctx.world - 1

    def train_step(self, tokens, targets):
        cfg, ctx, dev = self.cfg, self.ctx, self.ctx.device
        M = cfg.microbatches
        act_shape = (cfg.global_batch // M, cfg.seq_len, cfg.d_model)
        tok_mbs, tgt_mbs = tokens.chunk(M), targets.chunk(M)
        self.optimizer.zero_grad(set_to_none=True)

        inputs, outputs, total = [], [], 0.0
        # Fill: forward every microbatch through this stage.
        for i in range(M):
            if self.first:
                x = tok_mbs[i].to(dev)
            else:
                x = torch.empty(act_shape, device=dev)
                dist.recv(x, src=ctx.rank - 1)
                x.requires_grad_()
            y = self.model(x)
            if self.last:
                y = lm_loss(y, tgt_mbs[i].to(dev)) / M   # mean over microbatches = mean over batch
                total += y.item()
            else:
                dist.send(y.detach(), dst=ctx.rank + 1)
            inputs.append(x)
            outputs.append(y)

        # Drain: backward every microbatch, passing input gradients upstream.
        for i in range(M):
            if self.last:
                outputs[i].backward()
            else:
                g = torch.empty(act_shape, device=dev)
                dist.recv(g, src=ctx.rank + 1)
                outputs[i].backward(g)
            if not self.first:
                dist.send(inputs[i].grad, dst=ctx.rank - 1)

        self.optimizer.step()
        report = torch.tensor([total], device=dev)
        dist.broadcast(report, src=ctx.world - 1)   # only the last stage knows the loss
        return report.item()

    def comm_bytes_per_step(self, n_params):
        act = self.cfg.global_batch * self.cfg.seq_len * self.cfg.d_model * 4
        return act * ((not self.last) + (not self.first))   # activations down, gradients up

    def bubble_fraction(self):
        w, m = self.ctx.world, self.cfg.microbatches
        return (w - 1) / (m + w - 1)
