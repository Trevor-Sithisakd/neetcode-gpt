import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        combined = positive + negative
        # 2. Encode each sentence by replacing words with their IDs
        # generate ids
        vocab = sorted({word for sentence in combined for word in sentence.split()})
        # make it into a dictionary 
        idxs = {word: idx + 1 for idx, word in enumerate(vocab)}
        # 3. Combine positive + negative into one list of tensors
        encoded = [torch.tensor([idxs[w] for w in s.split()]) for s in combined]
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)
