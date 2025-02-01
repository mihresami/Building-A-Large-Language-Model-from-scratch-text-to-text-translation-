import torch.nn as nn
from torch.nn import LayerNorm
class Encoder(nn.Module):
    def __init__(self, d_model,encoderblocklist: nn.ModuleList):
        super().__init__()
        self.encoderblocklist = encoderblocklist
        self.layer_norm = LayerNorm(d_model)
    def forward(self, encoder_input, encoder_mask):
        for encoderblock in self.encoderblocklist:
            encoder_input = encoderblock(encoder_input, encoder_mask)
        encoder_output = self.layer_norm(encoder_input)
        return encoder_output