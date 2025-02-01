import torch.nn as nn
from torch.nn import LayerNorm
class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = LayerNorm(d_model)
    def forward(self, input, sub_layer):
        return input + self.dropout(sub_layer(self.layer_norm(input)))