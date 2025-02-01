import torch.nn as nn
import torch
from torch.nn import LayerNorm
class Decoder(nn.Module):
    def __init__(self,d_model,decoderblocklist: nn.ModuleList):
        super().__init__()
        self.decoderblocklist = decoderblocklist
        self.layer_norm = LayerNorm(d_model)
    def forward(self, decoder_input, decoder_mask, encoder_output, encoder_mask):
        for decoderblock in self.decoderblocklist:
            decoder_input = decoderblock(decoder_input, decoder_mask, encoder_output, encoder_mask)
        decoder_output = self.layer_norm(decoder_input)
        return decoder_output
class ProjectionLayer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.projection_layer = nn.Linear(d_model, vocab_size)
    def forward(self, decoder_output):
        # Linear layer to project the decoder output to the vocabulary size 
        output = self.projection_layer(decoder_output)
        return torch.log_softmax(output, dim=-1)