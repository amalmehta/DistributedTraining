"""Synthetic data: token sequences drawn from a fixed random Markov chain.

Why this instead of a real corpus: no downloads, every rank can rebuild any
batch from (data_seed, step) alone, and the best possible loss is known
exactly — the chain's entropy rate — so the charts have a true floor.
"""

import math

import torch


class MarkovData:
    def __init__(self, vocab_size: int, seq_len: int, data_seed: int, concentration: float = 0.1):
        g = torch.Generator().manual_seed(data_seed)
        # Low Dirichlet concentration -> each token has a few likely successors.
        alpha = torch.full((vocab_size,), concentration)
        gamma = torch._standard_gamma(alpha.expand(vocab_size, vocab_size).contiguous(), generator=g)
        self.P = gamma / gamma.sum(dim=1, keepdim=True)
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data_seed = data_seed

    def global_batch(self, step: int, batch_size: int):
        """The full batch for one step. Identical on every rank that asks."""
        g = torch.Generator().manual_seed(self.data_seed * 1_000_003 + step)
        seq = torch.empty(batch_size, self.seq_len + 1, dtype=torch.long)
        seq[:, 0] = torch.randint(self.vocab_size, (batch_size,), generator=g)
        for t in range(self.seq_len):
            seq[:, t + 1] = torch.multinomial(self.P[seq[:, t]], 1, generator=g).squeeze(1)
        return seq[:, :-1], seq[:, 1:]

    def entropy_floor(self) -> float:
        """Entropy rate of the chain in nats: the lowest achievable mean loss."""
        pi = torch.full((self.vocab_size,), 1.0 / self.vocab_size, dtype=torch.float64)
        P = self.P.double()
        for _ in range(1000):
            pi = pi @ P
        row_h = -(P * torch.log(P.clamp_min(1e-300))).sum(dim=1)
        return float((pi * row_h).sum())

    @property
    def uniform_loss(self) -> float:
        return math.log(self.vocab_size)
