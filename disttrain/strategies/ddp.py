"""Data parallel (DDP): every rank has the whole model and a slice of the batch.

After backward, DDP all-reduces (averages) the gradients across ranks, so every
replica takes the identical optimizer step and stays in sync.
"""

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..model import build_model, lm_loss
from .base import Strategy


class DDPStrategy(Strategy):
    name = "ddp"

    def setup(self):
        model = build_model(self.cfg).to(self.ctx.device)
        ids = [self.ctx.local_rank] if self.ctx.device.type == "cuda" else None
        self.model = DDP(model, device_ids=ids)
        self.optimizer = self.make_optimizer(self.model.parameters())

    def train_step(self, tokens, targets):
        tokens, targets = self.local_rows(tokens, targets)
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            loss = lm_loss(self.model(tokens), targets)
        self.backward(loss)             # gradient all-reduce overlaps with this
        self.optimizer_step()
        # Average the per-rank losses only for logging; training doesn't need it.
        report = loss.detach().clone()
        dist.all_reduce(report, op=dist.ReduceOp.SUM)
        return report.item() / self.ctx.world

    def comm_bytes_per_step(self, n_params):
        # Ring all-reduce of fp32 gradients: 2 (W-1)/W of the gradient size.
        w = self.ctx.world
        return int(2 * (w - 1) / w * n_params * 4)
