"""A small GPT-style transformer, built as a flat list of layers.

The flat list (embed, block, block, ..., head) is what makes every strategy
possible with one model: DDP and FSDP wrap the whole thing, FSDP shards each
Block separately, and the pipeline cuts the list into consecutive stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)

    def forward(self, idx):
        pos = torch.arange(idx.shape[1], device=idx.device)
        return self.tok(idx) + self.pos(pos)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class Head(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, x):
        return self.out(self.ln(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Embed(cfg)] + [Block(cfg) for _ in range(cfg.n_layers)] + [Head(cfg)])

    def forward(self, idx):
        x = idx
        for layer in self.layers:
            x = layer(x)
        return x


def build_model(cfg: Config) -> TinyGPT:
    """Same seed on every rank -> identical starting weights everywhere."""
    torch.manual_seed(cfg.seed)
    return TinyGPT(cfg)


def lm_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
