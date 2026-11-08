#!/usr/bin/env python3
"""texts.py — the three samples of plan 0.17.1 on WikiText-2 (raw), by the
checkpoint's own tokenizer (presk-specs `implementation/presk-plan-0.17.1.md`
§3.1).

    texts.py WIKITEXT_DIR CHECKPOINT_DIR OUT_DIR

WIKITEXT_DIR holds `wikitext-2-raw-v1/{train,test}-00000-of-00001.parquet`
(`hf download Salesforce/wikitext --repo-type dataset --include
'wikitext-2-raw-v1/*'`, CC BY-SA 4.0, the user's download). Each split's
`text` column is joined by "\\n\\n", as the field's perplexity scripts join
it (GPTQ's `datautils.py`, Hugging Face's perplexity guide), and tokenized
whole by `tokenizer.json`, without special tokens. Each window is the
beginning-of-text token followed by the split's next tokens, in order, the
windows contiguous and disjoint:

- calib.txt     4 windows of 16 (15 tokens of the train split each), the
                envelope's and every solve's;
- heldout.txt   32 windows of 32 (31 tokens of the test split each, from
                its first), the certificate's: 1 024 positions;
- quality.txt   128 windows of 129 (the next 128 tokens of the test split
                each): a window's first 128 ids are the context and its last
                128 the targets of the perplexity, 16 384 of them.

One window a line, ids separated by spaces. `texts.json` records the
parquet files' sha256, the rule, the splits' token counts, where each
sample starts and ends in its split, and every window decoded.
It needs `pyarrow` and `tokenizers` (`presk/.venv-hf`).
"""

import hashlib
import json
import os
import sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

CALIB = (4, 16)
HELDOUT = (32, 32)
QUALITY = (128, 129)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bos_id(checkpoint):
    for name in ("generation_config.json", "config.json"):
        p = os.path.join(checkpoint, name)
        if os.path.exists(p):
            with open(p) as f:
                b = json.load(f).get("bos_token_id")
            if b is not None:
                return int(b)
    raise SystemExit(f"{checkpoint}: no bos_token_id in its configuration")


def windows(tokens, start, count, length, bos):
    """`count` windows of `length` ids, each the beginning-of-text token and
    the next `length − 1` tokens from `start`; and where they end."""
    out, at = [], start
    for _ in range(count):
        body = tokens[at : at + length - 1]
        if len(body) < length - 1:
            raise SystemExit(f"the split ends at token {len(tokens)}, before {count} windows of {length}")
        out.append([bos] + body)
        at += length - 1
    return out, at


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    wiki, checkpoint, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(checkpoint, "tokenizer.json"))
    bos = bos_id(checkpoint)
    record = {"rule": "each split's `text` column joined by \"\\n\\n\", tokenized whole without special tokens; "
                      "every window the beginning-of-text token and the split's next tokens, contiguous and disjoint",
              "bos": bos, "sources": {}, "splits": {}, "samples": {}}
    tokens = {}
    for split in ("train", "test"):
        path = os.path.join(wiki, "wikitext-2-raw-v1", f"{split}-00000-of-00001.parquet")
        lines = pq.read_table(path).column("text").to_pylist()
        text = "\n\n".join(lines)
        tokens[split] = tok.encode(text, add_special_tokens=False).ids
        record["sources"][split] = {"file": os.path.relpath(path, wiki), "sha256": sha256(path)}
        record["splits"][split] = {"lines": len(lines), "characters": len(text), "tokens": len(tokens[split])}

    plan = (("calib", "train", 0, CALIB), ("heldout", "test", 0, HELDOUT), ("quality", "test", None, QUALITY))
    at_test = 0
    for name, split, start, (count, length) in plan:
        if start is None:
            start = at_test
        ws, end = windows(tokens[split], start, count, length, bos)
        if split == "test":
            at_test = end
        with open(os.path.join(out, f"{name}.txt"), "w") as f:
            for w in ws:
                f.write(" ".join(map(str, w)) + "\n")
        record["samples"][name] = {"split": split, "windows": count, "length": length, "from_token": start,
                                   "to_token": end, "text": [tok.decode(w[1:]) for w in ws]}
        print(f"{name}: {count} windows of {length} from the {split} split's tokens {start}..{end}"
              f" (of {len(tokens[split])})")
    with open(os.path.join(out, "texts.json"), "w") as f:
        json.dump(record, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
