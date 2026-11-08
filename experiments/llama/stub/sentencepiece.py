"""A stand-in for `sentencepiece`, never called (plan 0.17.1, presk D185).

llama.cpp's convert_hf_to_gguf.py (release b12132, conversion/base.py,
`_create_vocab_sentencepiece`) imports `SentencePieceProcessor` before it
checks for a `tokenizer.model`, which Llama 3's checkpoint does not have. The
check then raises FileNotFoundError, and the converter takes Llama 3's
byte-level BPE vocabulary from `tokenizer.json` (`_set_vocab_gpt2`). With the
module absent the import raises ModuleNotFoundError first, which the
converter does not catch. This file only lets the import succeed: an
instance would refuse to load anything.
"""


class SentencePieceProcessor:
    def __init__(self, *a, **k):
        raise RuntimeError("the sentencepiece stand-in of experiments/llama/stub is not a tokenizer")
