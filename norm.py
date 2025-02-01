import torch.nn as nn
import torch
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
    
    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)      
        std = input.std(dim=-1, keepdim=True)      
        return self.gamma * ((input - mean)/(std + self.eps)) + self.beta
        