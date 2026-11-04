#!/usr/bin/env python3
"""rw2.py — statement RW2 of plan 0.17.0 (presk D180): the converted
checkpoint, run by presk under `f64`, against Hugging Face `transformers`'
`LlamaForCausalLM` in `float64` on the same ids.

    .venv-hf/bin/python experiments/llama/rw2.py CHECKPOINT_DIR WEIGHTS.psk.d PROMPTS.txt
        [--presk target/release/presk] [--program coverage/llama3.psk]
        [--const NAME=N]... [--backend native] [--out DIR]

PROMPTS.txt holds one prompt per line, as whitespace-separated token ids,
all of one length. For each, presk runs `Llama3.apply` on the converted
weights (`--weights`) and the ids (`--data`), and `transformers` runs the
checkpoint upcast to `float64`. RW2 holds when, for every prompt, the
largest elementwise gap of the logits is at most 10⁻⁸ of their largest
magnitude.

Hugging Face's Llama computes in `float32` in three places whatever the
model's type (transformers 5.21, `modeling_llama.py`): the mixed precision
of a `bf16` model. Each alone misses the `float64` model by about 10⁻⁷.
- **The rotary angles.** `inv_freq` is built with `dtype=torch.float`, and
  the rotary `forward` casts with `.float()`.
- **`LlamaRMSNorm`** casts its input `.to(torch.float32)`.
- **The eager attention's softmax** has `dtype=torch.float32`.

The harness therefore:
- recomputes `inv_freq` by Hugging Face's own function (`ROPE_INIT_FUNCTIONS`)
  with `torch.float` standing for `torch.float64` while it runs, and checks
  that it agrees with Hugging Face's `float32` frequencies to 10⁻⁶;
- replaces the rotary `forward` by the same computation in `float64`;
- replaces `LlamaRMSNorm.forward` by the same lines without the cast;
- takes the SDPA attention, which computes in its operands' type.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
from transformers import LlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def float64_rope(model):
    """Hugging Face's rotary frequencies and angles, in float64."""
    rot = model.model.rotary_emb
    cfg = model.config
    kind = cfg.rope_parameters["rope_type"]
    init = ROPE_INIT_FUNCTIONS[kind] if kind != "default" else rot.compute_default_rope_parameters
    saved = torch.float
    torch.float = torch.float64
    try:
        inv64, scale = init(cfg)
    finally:
        torch.float = saved
    assert inv64.dtype == torch.float64, inv64.dtype
    inv32 = rot.inv_freq.to(torch.float64)
    rel = ((inv64 - inv32).abs() / inv64.abs()).max().item()
    assert rel < 1e-6, f"float64 frequencies {rel:.2e} from Hugging Face's"
    rot.inv_freq = inv64
    rot.attention_scaling = scale

    def forward(x, position_ids):
        freqs = position_ids[..., None].to(torch.float64) * rot.inv_freq.to(torch.float64)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * rot.attention_scaling).to(x.dtype), (emb.sin() * rot.attention_scaling).to(x.dtype)

    rot.forward = forward
    return rel


def float64_rmsnorm():
    """`LlamaRMSNorm.forward` as Hugging Face writes it, without its cast to float32."""
    from transformers.models.llama import modeling_llama

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

    modeling_llama.LlamaRMSNorm.forward = forward


def numbers(v):
    if isinstance(v, list):
        return [x for y in v for x in numbers(y)]
    return [float(v)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("weights")
    ap.add_argument("prompts")
    ap.add_argument("--presk", default="target/release/presk")
    ap.add_argument("--program", default="coverage/llama3.psk")
    ap.add_argument("--const", action="append", default=[])
    ap.add_argument("--backend", default="native")
    ap.add_argument("--out", default="out/llama3/rw2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    torch.set_default_dtype(torch.float64)
    float64_rmsnorm()
    model = LlamaForCausalLM.from_pretrained(a.checkpoint, dtype=torch.float64, attn_implementation="sdpa")
    model.eval()
    rel = float64_rope(model)
    vocab = model.config.vocab_size
    consts = [x for c in a.const for x in ("--const", c)]
    prompts = [[int(t) for t in line.split()] for line in open(a.prompts) if line.strip()]
    rows, worst = [], 0.0
    for k, ids in enumerate(prompts):
        n = len(ids)
        text = os.path.join(a.out, f"ids{k}.txt")
        data = os.path.join(a.out, f"ids{k}.psk.d")
        with open(text, "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        subprocess.run([a.presk, "data", "pack", "--type", f"idx[{vocab}][batch: 1, seq: {n}]", "-o", data, text], check=True, capture_output=True)
        res = os.path.join(a.out, f"presk{k}.json")
        with open(res, "w") as f:
            subprocess.run([a.presk, "run", a.program, "--entry", "Llama3.apply", *consts, "--dim", "B=1", "--dim", f"T={n}",
                            "--weights", a.weights, "--data", data, "--backend", a.backend, "--realize", "f64", "--json", "--result-only", "--digits", "17"],
                           check=True, stdout=f)
        with open(res) as f:
            got = np.array(numbers(json.load(f)["result"]["data"]), dtype=np.float64).reshape(1, n, vocab)
        with torch.no_grad():
            want = model(torch.tensor([ids])).logits.to(torch.float64).numpy()
        gap = float(np.abs(got - want).max())
        mag = float(np.abs(want).max())
        argmax_same = int((got.argmax(-1) == want.argmax(-1)).sum())
        rows.append({"prompt": k, "tokens": n, "gap": gap, "magnitude": mag, "relative": gap / mag, "argmax_same": argmax_same})
        worst = max(worst, gap / mag)
        print(f"prompt {k}: {n} tokens, max |presk − HF| {gap:.3e}, max |logit| {mag:.3e}, relative {gap / mag:.3e}, argmax the same at {argmax_same} of {n}")
    holds = worst <= 1e-8
    summary = {"frequencies_f64_vs_f32": rel, "prompts": rows, "worst_relative": worst, "rw2": "holds" if holds else "fails"}
    with open(os.path.join(a.out, "rw2.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"RW2 {'holds' if holds else 'FAILS'}: the worst relative gap {worst:.3e} (threshold 1e-8); Hugging Face's float32 frequencies {rel:.1e} from float64")
    sys.exit(0 if holds else 1)


if __name__ == "__main__":
    main()
