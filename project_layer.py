import torch.nn as nn
import torch
class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)
    def forward(self, decoder_output):
        output = self.projection_layer(decoder_output)
        return torch.log_softmax(output, dim=-1)