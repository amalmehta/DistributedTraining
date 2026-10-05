"""The core claim: every strategy trains the exact same model as one process would.

Distributed runs go through real `torchrun` subprocesses, just as a user would launch them.
"""

import json
import math
import subprocess
import sys

import pytest

from disttrain.config import Config
from disttrain.data import MarkovData
from disttrain.strategies.pipeline import stage_bounds
from disttrain.train import train

SMALL = ["--d-model", "32", "--n-heads", "2", "--n-layers", "4", "--seq-len", "16",
         "--global-batch", "8", "--microbatches", "4", "--steps", "6", "--log-every", "100"]


def small_config(**kw):
    base = dict(d_model=32, n_heads=2, n_layers=4, seq_len=16, global_batch=8,
                microbatches=4, steps=6, log_every=100)
    return Config(**{**base, **kw})


def run_torchrun(strategy, world, out, *extra):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}",
           "-m", "disttrain.train", "--strategy", strategy, "--out", str(out), *SMALL, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((out / "metrics.json").read_text())


@pytest.fixture(scope="module")
def baseline():
    return train(small_config(strategy="single"))


def test_batches_are_deterministic():
    d = MarkovData(16, 8, data_seed=7)
    a, b = d.global_batch(3, 4), d.global_batch(3, 4)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert (a[0][:, 1:] == a[1][:, :-1]).all()          # targets are inputs shifted by one
    assert not (d.global_batch(4, 4)[0] == a[0]).all()


def test_entropy_floor_is_below_uniform():
    d = MarkovData(64, 8, data_seed=1234)
    assert 0 < d.entropy_floor() < d.uniform_loss == pytest.approx(math.log(64))


@pytest.mark.parametrize("n_blocks,world", [(4, 1), (4, 2), (4, 4), (5, 2), (7, 3)])
def test_stage_bounds_cover_every_layer_once(n_blocks, world):
    bounds = stage_bounds(n_blocks, world)
    assert bounds[0][0] == 0 and bounds[-1][1] == n_blocks + 2
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    blocks = [min(hi, n_blocks + 1) - max(lo, 1) for lo, hi in bounds]
    assert max(blocks) - min(blocks) <= 1


def test_stage_bounds_rejects_too_many_stages():
    with pytest.raises(ValueError):
        stage_bounds(2, 3)


def test_single_process_learns(baseline):
    assert baseline["losses"][-1] < baseline["losses"][0]


@pytest.mark.parametrize("strategy,world", [("ddp", 2), ("fsdp", 2), ("pipeline", 2), ("pipeline", 4)])
def test_strategy_matches_single_process(strategy, world, baseline, tmp_path):
    result = run_torchrun(strategy, world, tmp_path)
    assert result["world"] == world
    for got, want in zip(result["losses"], baseline["losses"]):
        assert got == pytest.approx(want, abs=1e-4)


def test_fsdp_shards_memory(tmp_path):
    result = run_torchrun("fsdp", 2, tmp_path)
    full_bytes = result["n_params"] * 4
    for r in result["per_rank"]:
        assert r["memory"]["params"] <= full_bytes / 2 + 1024   # half, plus padding


# ---------------------------------------------------------------- checkpoints

def test_resume_continues_exactly(baseline, tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    train(small_config(strategy="single", steps=3, save=ckpt))
    resumed = train(small_config(strategy="single", resume=ckpt))
    assert resumed["start_step"] == 3
    assert resumed["losses"] == baseline["losses"][3:]          # bit-for-bit


@pytest.mark.parametrize("saver,saver_world,resumer,resumer_world", [
    ("ddp", 2, "fsdp", 2),
    ("fsdp", 2, "pipeline", 2),
    ("pipeline", 2, "single", 1),
])
def test_resume_under_a_different_strategy(saver, saver_world, resumer, resumer_world, baseline, tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    run_torchrun(saver, saver_world, tmp_path / "a", "--steps", "3", "--save", ckpt)
    if resumer_world == 1:
        resumed = train(small_config(strategy=resumer, resume=ckpt))
    else:
        resumed = run_torchrun(resumer, resumer_world, tmp_path / "b", "--resume", ckpt)
    assert resumed["start_step"] == 3
    assert resumed["resumed_from"] == {"strategy": saver, "world": saver_world}
    for got, want in zip(resumed["losses"], baseline["losses"][3:]):
        assert got == pytest.approx(want, abs=1e-4)


def test_resume_rejects_a_different_model_shape(tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    train(small_config(strategy="single", steps=2, save=ckpt))
    with pytest.raises(SystemExit, match="d_model"):
        train(Config(strategy="single", d_model=64, n_heads=2, n_layers=4, seq_len=16,
                     global_batch=8, steps=4, resume=ckpt))


def test_resume_needs_steps_left(tmp_path):
    ckpt = str(tmp_path / "ckpt.pt")
    train(small_config(strategy="single", steps=2, save=ckpt))
    with pytest.raises(SystemExit, match="already at step 2"):
        train(small_config(strategy="single", steps=2, resume=ckpt))
