from .ddp import DDPStrategy
from .fsdp import FSDPStrategy
from .pipeline import PipelineStrategy
from .single import SingleStrategy

STRATEGY_CLASSES = {
    "single": SingleStrategy,
    "ddp": DDPStrategy,
    "fsdp": FSDPStrategy,
    "pipeline": PipelineStrategy,
}


def get_strategy(cfg, ctx):
    return STRATEGY_CLASSES[cfg.strategy](cfg, ctx)
