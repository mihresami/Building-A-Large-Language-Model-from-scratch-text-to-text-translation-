import torch.nn as nn
from multi_head import MultiHeadAttention
from feed_forward import FeedForward
from add_norm import AddAndNorm
class DecoderBlock(nn.Module):
    def __init__(self, d_model,masked_multihead_attention: MultiHeadAttention,multihead_attention: MultiHeadAttention, feed_forward: FeedForward, dropout_rate: float):
        super().__init__()
        # DecoderBlock is initialized with masked_multihead_attention, multihead_attention, feed_forward and dropout_rate as input parameters.

        self.masked_multihead_attention = masked_multihead_attention
        self.multihead_attention = multihead_attention
        self.feed_forward = feed_forward
        self.add_and_norm_list = nn.ModuleList([AddAndNorm(d_model,dropout_rate) for _ in range(3)])
    def forward(self, decoder_input, decoder_mask, encoder_output, encoder_mask):
        # DecoderBlock is called with decoder_input, decoder_mask, encoder_output and encoder_mask as input parameters.
        decoder_input = self.add_and_norm_list[0](decoder_input, lambda decoder_input: self.masked_multihead_attention(decoder_input, decoder_input, decoder_input, decoder_mask))
        decoder_input = self.add_and_norm_list[1](decoder_input, lambda decoder_input: self.multihead_attention(decoder_input,encoder_output, encoder_output, encoder_mask)) 
        decoder_input = self.add_and_norm_list[2](decoder_input, self.feed_forward)
        return decoder_input