import torch.nn as nn
from add_norm import AddAndNorm
from multi_head import MultiHeadAttention
from feed_forward import FeedForward
class EncoderBlock(nn.Module):
    def __init__(self, d_model,multihead_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float):
        super().__init__()
        # EncoderBlock is initialized with multihead_attention, feed_forward and dropout_rate as input parameters. 
        self.multihead_attention = multihead_attention
        self.feed_forward = feed_forward
        self.add_and_norm_list = nn.ModuleList([AddAndNorm(d_model,dropout_rate) for _ in range(2)])
    def forward(self, encoder_input, encoder_mask):
        # EncoderBlock is called with encoder_input and encoder_mask as input parameters.
        encoder_input = self.add_and_norm_list[0](encoder_input, lambda encoder_input: self.multihead_attention(encoder_input, encoder_input, encoder_input, encoder_mask))
        encoder_input = self.add_and_norm_list[1](encoder_input, self.feed_forward)
        return encoder_input