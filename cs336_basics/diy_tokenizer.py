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
        import pdb;pdb.set_trace()
        toy_vocab = { "0": " ", "1": "a", "2": "c", "3": "e", "4": "h", "5": "t", "6": "th", "7": " c", "8": " a", "9": "the", "10": " at" }
        toy_vocab_rev = {v: int(k) for k, v in toy_vocab.items()}
        toy_merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'),(b' a', b't')]
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
        if pattern:
            pattern = f"({pattern})"        
        
        chunk = text
        segments = re.split(pattern, chunk) if pattern else [chunk]        
        pre_tokens_bytes: list[list[bytes]] = []
        for segment in segments:
            # if not segment:
            #     continue
            if segment in self.special_tokens:
                # Treat the whole special token as a single token
                token_bytes = [segment.encode("utf-8")]
                pre_tokens_bytes.append(token_bytes)
            else:
                # Standard tokenization
                tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
                for token in tokens:
                    token_bytes = [bytes([b]) for b in token]
                    pre_tokens_bytes.append(token_bytes)        
        print('pre_tokens_bytes', pre_tokens_bytes)    
        # 
        token_ids = []
        for one_pre_token in pre_tokens_bytes:
            while True:
                merge_happened = False
                for i in range(0, len(one_pre_token) - 1):
                    cur_pair = (one_pre_token[i], one_pre_token[i+1])
                    print('cur_pair')
                    if cur_pair in toy_merges:
                        one_pre_token[i] = cur_pair[0] + cur_pair[1]
                        del one_pre_token[i+1]
                        merge_happened = True
                        break
                if not merge_happened:
                    break
            print('one_pre_token', one_pre_token)    
            for one_merged_token in one_pre_token:
                token_ids.append(toy_vocab_rev[one_merged_token.decode('utf-8')])
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        pass
    
    def decode(self, ids: list[int]) -> str:
        toy_vocab = { "0": " ", "1": "a", "2": "c", "3": "e", "4": "h", "5": "t", "6": "th", "7": " c", "8": " a", "9": "the", "10": " at" }
        result = []
        for id in ids:
            result.append(toy_vocab[str(id)])
        return "".join(result)
        
