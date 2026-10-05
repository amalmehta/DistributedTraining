"""The one interface every strategy implements.

The trainer only ever calls `train_step(tokens, targets)` with the *global*
batch and gets back the global mean loss. How the work is split — by rows,
by parameter shards, or by layers — is entirely the strategy's business.
"""

import torch

from ..config import Config
from ..dist_utils import Context


def tensor_bytes(tensors):
    return int(sum(t.numel() * t.element_size() for t in tensors if t is not None))


class Strategy:
    name = "base"

    def __init__(self, cfg: Config, ctx: Context):
        self.cfg, self.ctx = cfg, ctx
        self.model = None
        self.optimizer = None

    # --- subclasses fill these in -------------------------------------------------
    def setup(self):
        """Build self.model and self.optimizer."""
        raise NotImplementedError

    def train_step(self, tokens, targets) -> float:
        """One optimizer step on the global batch; returns the global mean loss."""
        raise NotImplementedError

    def comm_bytes_per_step(self, n_params: int) -> int:
        """Bytes each rank sends per step (textbook estimate, fp32)."""
        return 0

    # --- shared helpers ------------------------------------------------------------
    def make_optimizer(self, params):
        return torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

    def local_rows(self, tokens, targets):
        """Data parallel: this rank's contiguous slice of the global batch."""
        per = self.cfg.global_batch // self.ctx.world
        lo = self.ctx.rank * per
        dev = self.ctx.device
        return tokens[lo:lo + per].to(dev), targets[lo:lo + per].to(dev)

    def memory_report(self):
        """What this rank actually holds: parameters, gradients, optimizer state."""
        params = list(self.model.parameters())
        opt_state = [v for s in self.optimizer.state.values() for v in s.values()
                     if torch.is_tensor(v) and v.dim() > 0]
        return {
            "params": tensor_bytes(params),
            "grads": tensor_bytes(p.grad for p in params),
            "optimizer": tensor_bytes(opt_state),
        }
