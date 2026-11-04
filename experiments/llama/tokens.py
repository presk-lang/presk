#!/usr/bin/env python3
"""tokens.py — text to token ids and back, by a Hugging Face checkpoint's
own tokenizer (plan 0.17.0 step 7, presk D180). The tokenizer stays outside
the compiler: `presk serve` and `generate.c` take ids and print ids.

    tokens.py encode CHECKPOINT_DIR "text" [--length N] [--no-bos]
    tokens.py decode CHECKPOINT_DIR "id id id ..."

`encode` prints the ids on one line, the beginning-of-text token first
(Llama 3's `<|begin_of_text|>`, 128000) unless `--no-bos`; with `--length
N` it keeps the first N, and refuses a text shorter than that. `decode`
prints the text of the ids. It reads `tokenizer.json` with the `tokenizers`
library (`presk/.venv`, the manual's `dependencies` chapter).
"""

import argparse
import json
import os
import sys

from tokenizers import Tokenizer


def bos_id(path):
    """The beginning-of-text id, from the checkpoint's configuration."""
    for name in ("generation_config.json", "config.json"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            with open(p) as f:
                b = json.load(f).get("bos_token_id")
            if b is not None:
                return int(b)
    raise SystemExit(f"{path}: no bos_token_id in its configuration")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["encode", "decode"])
    ap.add_argument("checkpoint")
    ap.add_argument("text")
    ap.add_argument("--length", type=int)
    ap.add_argument("--no-bos", action="store_true")
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.checkpoint, "tokenizer.json"))
    if a.command == "encode":
        ids = tok.encode(a.text, add_special_tokens=False).ids
        if not a.no_bos:
            ids = [bos_id(a.checkpoint)] + ids
        if a.length is not None:
            if len(ids) < a.length:
                raise SystemExit(f"{len(ids)} tokens, fewer than --length {a.length}")
            ids = ids[: a.length]
        print(" ".join(map(str, ids)))
    else:
        ids = [int(t) for t in a.text.split()]
        sys.stdout.write(tok.decode(ids) + "\n")


if __name__ == "__main__":
    main()
