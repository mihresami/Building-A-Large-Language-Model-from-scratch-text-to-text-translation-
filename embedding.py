import torch.nn as nn
import math
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Embedding layer is initialized with vocab_size and d_model as input parameters and returns the embedding output. 
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, input):
        # Embedding layer takes in input and returns the embedding output after scaling it by sqrt(d_model) as per the transformer paper. 
        embedding_output = self.embedding(input) * math.sqrt(self.d_model)
        return embedding_output