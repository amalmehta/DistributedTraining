"""Fully sharded data parallel (FSDP): like DDP, but no rank keeps a full copy.

Each Block's parameters, gradients and optimizer state are split into W
shards, one per rank. Right before a Block runs, its weights are all-gathered;
right after backward, its gradients are reduce-scattered back into shards.
Memory per rank drops roughly W-fold for the price of extra communication.
"""

import torch.distributed as dist
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from ..model import Block, build_model, lm_loss
from .base import Strategy


class FSDPStrategy(Strategy):
    name = "fsdp"

    def setup(self):
        model = build_model(self.cfg).to(self.ctx.device)
        self.model = FSDP(
            model,
            auto_wrap_policy=ModuleWrapPolicy({Block}),   # one shard group per Block (+ the rest)
            device_id=self.ctx.device,
            use_orig_params=False,                         # parameters() then yields the local shards
        )
        # Built after wrapping, so the optimizer only ever sees (and stores state for) shards.
        self.optimizer = self.make_optimizer(self.model.parameters())

    def train_step(self, tokens, targets):
        tokens, targets = self.local_rows(tokens, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss = lm_loss(self.model(tokens), targets)
        loss.backward()
        self.optimizer.step()
        report = loss.detach().clone()
        dist.all_reduce(report, op=dist.ReduceOp.SUM)
        return report.item() / self.ctx.world

    def _full_state_dict_type(self, rank0_only):
        # Offload only from GPU: on a CPU device, offload_to_cpu frees the very storage it reads.
        offload = self.ctx.device.type == "cuda"
        return FSDP.state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=offload, rank0_only=rank0_only),
            FullOptimStateDictConfig(offload_to_cpu=offload, rank0_only=rank0_only))

    def full_state(self):
        # FSDP all-gathers the shards and hands back plain TinyGPT names on rank 0.
        with self._full_state_dict_type(rank0_only=True):
            params = self.model.state_dict()
            optim = FSDP.optim_state_dict(self.model, self.optimizer)
        return params, optim.get("state", {})

    def load_full_state(self, params, optim):
        # Every rank loads the full tensors and FSDP keeps only its own shard of each.
        group = {k: v for k, v in self.optimizer.param_groups[0].items() if k != "params"}
        full_osd = {"state": optim, "param_groups": [{**group, "params": list(params)}]}
        with self._full_state_dict_type(rank0_only=False):
            self.model.load_state_dict(params)
            osd = FSDP.optim_state_dict_to_load(self.model, self.optimizer, full_osd)
        self.optimizer.load_state_dict(osd)

    def comm_bytes_per_step(self, n_params):
        # All-gather for forward, all-gather again for backward, reduce-scatter grads:
        # 3 x (W-1)/W of the model size.
        w = self.ctx.world
        return int(3 * (w - 1) / w * n_params * 4)
