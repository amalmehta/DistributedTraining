"""Baseline: one process, whole model, whole batch. Every other strategy must match it."""

from ..model import build_model, lm_loss
from .base import Strategy


class SingleStrategy(Strategy):
    name = "single"

    def setup(self):
        if self.ctx.world != 1:
            raise SystemExit("--strategy single runs in one process; launch it with python, not torchrun")
        self.model = build_model(self.cfg).to(self.ctx.device)
        self.optimizer = self.make_optimizer(self.model.parameters())

    def train_step(self, tokens, targets):
        self.optimizer.zero_grad(set_to_none=True)
        with self.autocast():
            loss = lm_loss(self.model(tokens.to(self.ctx.device)), targets.to(self.ctx.device))
        self.backward(loss)
        self.optimizer_step()
        return loss.item()
