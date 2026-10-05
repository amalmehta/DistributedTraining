# Distributed Training Demo

One PyTorch training loop, four ways to split it across processes — single, DDP, FSDP and pipeline — and proof that they all train the exact same model.

[![Distributed Training Demo site](docs/screenshot.png)](https://amalmehta.github.io/DistributedTraining/)

```mermaid
flowchart LR
  B["Global batch<br/>(same on every rank)"] --> S{"--strategy"}
  S -->|single| A["1 process<br/>whole model, whole batch"]
  S -->|ddp| D["W full copies<br/>B/W rows each<br/>all-reduce gradients"]
  S -->|fsdp| F["W shards of every layer<br/>B/W rows each<br/>all-gather + reduce-scatter"]
  S -->|pipeline| P["W stages of layers<br/>M microbatches<br/>send/recv activations"]
  A & D & F & P --> L["Identical loss curve"]
```

- **Live site:** https://amalmehta.github.io/DistributedTraining/
- [Instructions](docs/INSTRUCTIONS.md) — setup, run, test, benchmark
- [System design](docs/SYSTEM-DESIGN.md) — architecture, flows, decisions, limits
- [File structure](docs/FILE-STRUCTURE.md) — what's where
