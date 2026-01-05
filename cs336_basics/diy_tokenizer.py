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
        self.vocab = vocab
        self.merges = merges
        self.merge_ranks = {
            pair:rank for rank, pair in enumerate(merges)
        }
        self.special_tokens = special_tokens
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # self.vocab_rev = {v: (int(k) if isinstance(k, str) else k) for k, v in vocab.items()}
        self.vocab_rev = {v: int(k) for k, v in vocab.items()}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        # toy_vocab = { "0": " ", "1": "a", "2": "c", "3": "e", "4": "h", "5": "t", "6": "th", "7": " c", "8": " a", "9": "the", "10": " at" }
        # toy_vocab_rev = {v: int(k) for k, v in toy_vocab.items()}
        # toy_merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'),(b' a', b't')]
        # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pattern = None
        if self.special_tokens:
            specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(tok) for tok in specials)
            if pattern:
                pattern = f"({pattern})"        
        
        chunk = text
        segments = re.split(pattern, chunk) if pattern else [chunk]        
        pre_tokens_bytes: list[list[bytes]] = []
        for segment in segments:
            # if not segment:
            #     continue
            if self.special_tokens and segment in self.special_tokens:
                # Treat the whole special token as a single token
                token_bytes = [segment.encode("utf-8")]
                pre_tokens_bytes.append(token_bytes)
            else:
                # Standard tokenization
                tokens = [match.group(0).encode("utf-8") for match in re.finditer(self.PAT, segment)]
                for token in tokens:
                    token_bytes = [bytes([b]) for b in token]
                    pre_tokens_bytes.append(token_bytes)        
        # print('pre_tokens_bytes', pre_tokens_bytes)    

        token_ids = []
        for one_pre_token in pre_tokens_bytes:
            while True:
                best_pair = None
                best_rank = None
                best_pair_idx = None
                for i in range(0, len(one_pre_token) - 1):
                    cur_pair = (one_pre_token[i], one_pre_token[i+1])
                    # print('cur_pair')
                    if cur_pair in self.merges:
                        cur_pair_rank = self.merge_ranks[cur_pair]
                        if best_rank is None or best_rank > cur_pair_rank:
                            best_rank = cur_pair_rank
                            best_pair = cur_pair                         
                            best_pair_idx = i
                if best_pair_idx is None:
                    break
                else:
                    one_pre_token[best_pair_idx] = best_pair[0] + best_pair[1]
                    del one_pre_token[best_pair_idx+1]
            # print('one_pre_token', one_pre_token)    
            for one_merged_token in one_pre_token:
                token_ids.append(self.vocab_rev[one_merged_token])
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            text_enc = self.encode(text)
            # yield from text_enc
            for one_enc in text_enc:
                yield one_enc
    
    def decode(self, ids: list[int]) -> str:
        # toy_vocab = { "0": " ", "1": "a", "2": "c", "3": "e", "4": "h", "5": "t", "6": "th", "7": " c", "8": " a", "9": "the", "10": " at" }
        # print('ids: ', ids)
        # tokens = bytes()
        # vocab_size = len(self.vocab)
        # replacement_bytes = "\uFFFD".encode("utf-8")

        # for token_id in ids:
        #     if token_id < vocab_size:
        #         token = self.vocab.get(token_id) # if isinstance(self.vocab, dict) else self.vocab[token_id]
        #         print('append: ', token)
        #         if token is None:
        #             token = replacement_bytes
        #     else:
        #         token = replacement_bytes
        #     tokens += token
        # text = tokens.decode("utf-8", errors="replace")
        # print('text: ', text)
        # return text
        
        # import pdb;pdb.set_trace()
        # print('ids: ', ids)
        result = []
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"        
        for id in ids:
            if id < vocab_size:
                # print('append: ', self.vocab[id])
                result.append(self.vocab[id])
            else:
                result.append(replacement_char)
        text = b"".join(result)
        text = text.decode("utf-8", errors='replace')
        # print('text: ', text)

        return text