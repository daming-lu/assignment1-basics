import json
import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict
import time
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


class DiyTokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        import pdb;pdb.set_trace()
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        toy_vocab = { "0": " ", "1": "a", "2": "c", "3": "e", "4": "h", "5": "t", "6": "th", "7": " c", "8": " a", "9": "the", "10": " at" }
        toy_merges = [(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'),(b' a', b't')]
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass
    
    def decode(self, ids: list[int]) -> str:
        pass
