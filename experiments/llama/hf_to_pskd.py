#!/usr/bin/env python3
"""hf_to_pskd.py — a Hugging Face Llama checkpoint as the `.psk.d` parameter
file of `coverage/llama3.psk` (plan 0.17.0 step 7, presk D180).

    hf_to_pskd.py CHECKPOINT_DIR -o OUT.psk.d [--program coverage/llama3.psk]
        [--model Llama3] [--presk target/release/presk] [--const NAME=N]...
        [--encoding bf16|f32|f64]

`safetensors` is read by hand, with the standard library and numpy: an
8-byte little-endian length, a JSON header naming each tensor's dtype, shape
and byte range, then the tensors' raw little-endian bytes. A sharded
checkpoint is read through its `model.safetensors.index.json`.

The leaves are presk's, in the order `presk dump --stage elab` lists the
model's parameter tree:
- each mapped to a Hugging Face tensor by its name;
- a projection's `weight`, which a `Linear` stores as ℝ[out, in],
  transposed to presk's ℝ[in, out];
- the layers stacked along a leading axis.

Nothing is permuted: the program's RoPE pairs dimension r of a head with
r + HD/2, as Hugging Face's checkpoints have it (plan §10). The numbers are
the checkpoint's own. By default they are written in its type, `bf16`, as a
version 2 file; `--encoding f32` or `f64` widens them exactly, the latter as
a version 1 file. The script checks every shape against the program's and
refuses a number that the encoding does not hold.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys

import numpy as np

# presk's path (in `presk dump --stage elab`'s listing) -> (the Hugging Face
# name, with {i} the layer, and whether the tensor is transposed).
MAP = {
    "tok": ("model.embed_tokens.weight", False),
    "blocks.attn.wq": ("model.layers.{i}.self_attn.q_proj.weight", True),
    "blocks.attn.wk": ("model.layers.{i}.self_attn.k_proj.weight", True),
    "blocks.attn.wv": ("model.layers.{i}.self_attn.v_proj.weight", True),
    "blocks.attn.wo": ("model.layers.{i}.self_attn.o_proj.weight", True),
    "blocks.ff.gate": ("model.layers.{i}.mlp.gate_proj.weight", True),
    "blocks.ff.up": ("model.layers.{i}.mlp.up_proj.weight", True),
    "blocks.ff.down": ("model.layers.{i}.mlp.down_proj.weight", True),
    "blocks.n1.γ": ("model.layers.{i}.input_layernorm.weight", False),
    "blocks.n2.γ": ("model.layers.{i}.post_attention_layernorm.weight", False),
    "norm.γ": ("model.norm.weight", False),
}

ENCODING = {"f64": 0, "f32": 1, "bf16": 2}


def presk_leaves(presk, program, model, consts):
    """The model's leaves `(path, shape)` in storage order, from presk."""
    cmd = [presk, "dump", "--stage", "elab", program] + [a for c in consts for a in ("--const", c)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    lines = out.splitlines()
    try:
        start = lines.index(f"{model}:")
    except ValueError:
        raise SystemExit(f"{program}: no instance `{model}` in `presk dump --stage elab`")
    leaves = []
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        m = re.match(r"\s+(\S+) : ℝ\[(.*)\]$", line)
        if not m:
            raise SystemExit(f"not a real leaf: {line!r}")
        shape = [int(a.rpartition(":")[2]) for a in m.group(2).split(",") if a.strip()]
        leaves.append((m.group(1), shape))
    return leaves


class Checkpoint:
    """The tensors of a safetensors checkpoint, read on demand from a memory map."""

    DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}

    def __init__(self, path):
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.where = {}
        for name in files:
            full = os.path.join(path, name)
            with open(full, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            data = np.memmap(full, dtype=np.uint8, mode="r", offset=8 + n)
            for key, meta in header.items():
                if key != "__metadata__":
                    self.where[key] = (data, meta)

    def bf16_bits(self, name):
        """The tensor's elements as bf16 bit patterns (uint16), exactly."""
        if name not in self.where:
            raise SystemExit(f"the checkpoint has no tensor `{name}`")
        data, meta = self.where[name]
        a, b = meta["data_offsets"]
        dt = self.DTYPES[meta["dtype"]]
        arr = np.frombuffer(data[a:b], dtype=dt).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            return arr
        f32 = arr.astype(np.float32)
        if not np.array_equal(f32.astype(arr.dtype), arr):
            raise SystemExit(f"`{name}`: {meta['dtype']} values that are no f32")
        bits = f32.view(np.uint32)
        if np.any(bits & 0xFFFF):
            raise SystemExit(f"`{name}`: values that are no bf16; convert with --encoding f32 or f64")
        return (bits >> 16).astype(np.uint16)

    def values(self, name):
        """The tensor's elements as float64, exactly."""
        data, meta = self.where[name]
        a, b = meta["data_offsets"]
        arr = np.frombuffer(data[a:b], dtype=self.DTYPES[meta["dtype"]]).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            return (arr.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        return arr.astype(np.float64)


def leaf_array(ck, path, shape, encoding):
    """The leaf `path` of presk's shape, from the checkpoint, in `encoding`."""
    if path not in MAP:
        raise SystemExit(f"no Hugging Face tensor for presk's `{path}`")
    name, transpose = MAP[path]
    get = ck.bf16_bits if encoding == "bf16" else ck.values

    def one(i=None):
        t = get(name.format(i=i))
        return t.T if transpose else t

    if "{i}" in name:
        arr = np.stack([one(i) for i in range(shape[0])])
    else:
        arr = one()
    if list(arr.shape) != list(shape):
        raise SystemExit(f"`{path}`: the checkpoint gives {list(arr.shape)}, the program {shape}")
    if encoding == "f32":
        f32 = arr.astype(np.float32)
        if not np.array_equal(f32.astype(np.float64), arr):
            raise SystemExit(f"`{path}`: values that are no f32; convert with --encoding f64")
        return np.ascontiguousarray(f32).astype("<f4")
    if encoding == "f64":
        return np.ascontiguousarray(arr).astype("<f8")
    return np.ascontiguousarray(arr).astype("<u2")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--program", default="coverage/llama3.psk")
    ap.add_argument("--model", default="Llama3")
    ap.add_argument("--presk", default="target/release/presk")
    ap.add_argument("--const", action="append", default=[], help="NAME=N, as presk takes it")
    ap.add_argument("--encoding", choices=list(ENCODING), default="bf16")
    a = ap.parse_args()

    leaves = presk_leaves(a.presk, a.program, a.model, a.const)
    ck = Checkpoint(a.checkpoint)
    ty = f"params {a.model}".encode()
    code = ENCODING[a.encoding]
    with open(a.out, "wb") as f:
        f.write(b"PSKD")
        f.write(struct.pack("<H", 2 if code else 1))
        f.write(struct.pack("<I", len(ty)) + ty)
        f.write(struct.pack("<Q", 1))
        if code:
            f.write(struct.pack("<I", len(leaves)) + bytes([code] * len(leaves)))
        total = 0
        for path, shape in leaves:
            arr = leaf_array(ck, path, shape, a.encoding)
            f.write(arr.tobytes())
            total += arr.size
    print(f"{a.out}: {len(leaves)} leaves, {total} parameters in {a.encoding}, from {a.checkpoint}", file=sys.stderr)


if __name__ == "__main__":
    main()
