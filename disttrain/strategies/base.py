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

    # --- checkpointing ---------------------------------------------------------------
    # The portable format: parameters and AdamW state keyed by their names in a plain
    # TinyGPT ("layers.1.qkv.weight"), full-size tensors on CPU. Any strategy, at any
    # process count, can write it and read it back.

    def named_params(self):
        """(TinyGPT name, local parameter) for what this rank holds. Single and DDP hold everything."""
        model = getattr(self.model, "module", self.model)   # unwrap DDP
        return list(model.named_parameters())

    def full_state(self):
        """(params, optimizer state) for the whole model. Collective: every rank calls it."""
        return self.local_state()

    def load_full_state(self, params, optim):
        """Overwrite this rank's parameters and optimizer state from the portable format."""
        with torch.no_grad():
            for name, p in self.named_params():
                p.copy_(params[name])
                if name in optim:
                    self.optimizer.state[p] = {k: v.clone().to(p.device) if v.dim() else v.clone()
                                               for k, v in optim[name].items()}

    def local_state(self):
        params, optim = {}, {}
        for name, p in self.named_params():
            params[name] = p.detach().cpu().clone()
            if p in self.optimizer.state:
                optim[name] = {k: v.detach().cpu().clone() for k, v in self.optimizer.state[p].items()}
        return params, optim

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
