import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16,1)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embeddings = self.embedding_layer(x) # this is what gives the sentiment that I was confused about this gives the score of how positive or negative the thing is that is what gets measure
        averaged = torch.mean(embeddings, dim=1)# need average for BoW
        projected = self.linear_layer(averaged) # this gives the output for the the sentiment as a scalar 
        activated = self.sigmoid_layer(projected) # this gives the squased output between 0 and 1 which is how the sentiment is calculated
        return torch.round(activated, decimals = 4) # 
        # Return a B, 1 tensor and round to 4 decimal places
       
