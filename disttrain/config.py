"""Run configuration: one dataclass, filled from the command line."""

import argparse
from dataclasses import dataclass, asdict, fields

STRATEGIES = ("single", "ddp", "fsdp", "pipeline")


@dataclass
class Config:
    strategy: str = "single"
    # model
    vocab_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    seq_len: int = 64
    # training
    global_batch: int = 32       # sequences per optimizer step, summed over all ranks
    microbatches: int = 4        # pipeline only: how many pieces each batch is cut into
    steps: int = 150
    lr: float = 3e-3
    weight_decay: float = 0.01
    seed: int = 0                # model init
    data_seed: int = 1234        # the Markov chain and the batches drawn from it
    log_every: int = 10
    out: str = ""                # folder for metrics.json; empty = don't write

    def to_dict(self):
        return asdict(self)


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(
        description="Train the same small transformer with any distributed strategy.")
    for f in fields(Config):
        kw = {"default": f.default, "type": type(f.default)}
        if f.name == "strategy":
            kw["choices"] = STRATEGIES
        p.add_argument("--" + f.name.replace("_", "-"), dest=f.name, **kw)
    return Config(**vars(p.parse_args(argv)))
