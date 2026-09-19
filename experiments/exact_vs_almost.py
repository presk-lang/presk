#!/usr/bin/env python3
"""Exact versus almost: two experiments on an exported training run.

Run from the repository root after `presk export … --conformance-entry train:K`
(`make experiment` does it for the tutorial's tower and the tiny MiniGPT):

    .venv/bin/python experiments/exact_vs_almost.py out/tutorial_train

1. Shapes as types: the RMSNorm division `x / sqrt(ms)` that Presk aligns by
   axis name is, in plain PyTorch, a trailing-axis broadcast that runs without
   error and returns garbage whenever the batch equals the model width.
2. The realization gap: the K interpreter steps recorded by the export
   (`θ₀`, the batches, `θ_K` at 256 bits) are replayed in float32 and float64
   with `train.py`'s optimizer and schedule; the worst relative gap of every
   parameter to the exact reference is the error of that arithmetic.
"""
import argparse
import os
import sys

import torch


def broadcasting_demo():
    torch.manual_seed(0)
    x = torch.randn(4, 4)                                  # batch 4, model 4
    ms = (x ** 2).mean(dim=-1)                             # one value per row
    wrong = x / torch.sqrt(ms + 1e-6)                      # trailing broadcast: column j / row j
    right = x / torch.sqrt(ms + 1e-6).unsqueeze(-1)        # what Presk means by ℝ[..., model] / ℝ[...]
    gap = (wrong - right).abs().max().item()
    print("1. shapes as types")
    print(f"   plain PyTorch, batch == model width: `x / sqrt(ms)` runs without error, max |wrong - right| = {gap:.3f}")
    print("   Presk: `x / sqrt(ms)` with x : ℝ[..., model: D], ms : ℝ[...] aligns by name; the export inserts unsqueeze(-1)")


def replay(dtype):
    import test_conformance as tc
    from pskd import PskdDataset, collate
    import train as tr

    inputs = {k: tc.to_value(v) for k, v in tc.load("inputs.json").items()}
    expected = tc.to_value(tc.load("expected.json"))
    batches = tc.load("batches.json")
    model = tr.make_model().to(dtype)
    model.load_state_dict({k: v.to(dtype) for k, v in tc.flatten_params(inputs["θ"], model).items()})
    theta = dict(model.named_parameters())
    dataset = PskdDataset(tr.DATA, dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.LR, betas=tr.BETAS, eps=tr.EPS, weight_decay=tr.WEIGHT_DECAY)
    for t, idxs in enumerate(batches, start=1):
        for group in opt.param_groups:
            group["lr"] = tr.LR * tr.schedule(t, len(batches))
        opt.zero_grad()
        tr.OBJECTIVE(theta, collate([dataset[i] for i in idxs])).backward()
        if tr.GRAD_CLIP is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.GRAD_CLIP)
        opt.step()
    exp = tc.flatten_params(expected, model)
    # The largest absolute gap over all parameters, and the model's scale (the
    # largest exact parameter): an elementwise relative gap would be dominated by
    # parameters that both arithmetics keep near zero.
    abs_gap = max((p.detach().double() - exp[n]).abs().max().item() for n, p in model.named_parameters())
    scale = max(exp[n].abs().max().item() for n in exp)
    return len(batches), abs_gap, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export_dir", help="an output directory of `presk export --conformance-entry train:K`")
    args = ap.parse_args()
    broadcasting_demo()
    os.chdir(args.export_dir)
    sys.path.insert(0, os.getcwd())
    print(f"2. the realization gap ({args.export_dir})")
    for dtype in (torch.float32, torch.float64):
        k, abs_gap, scale = replay(dtype)
        print(f"   {str(dtype).removeprefix('torch.'):8s} after {k} steps: max |θ − θ_exact| = {abs_gap:.3e}  (parameters up to {scale:.3f} in magnitude, gap/scale {abs_gap / scale:.1e})")


if __name__ == "__main__":
    main()
