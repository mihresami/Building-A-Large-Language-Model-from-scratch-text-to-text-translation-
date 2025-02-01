import torch.nn as nn
from transformer import Transformer
from model_train import arg_parse
import torch
class LL_model(nn.Module):
    def __init__(self,source_vocab_size, max_len, d_ff, d_model=512, num_heads=8, num_blocks=6, dropout_rate=0.1):
        super(LL_model,self).__init__()
        self.model = Transformer(source_vocab_size, d_model,d_ff, num_heads, num_blocks, max_len, dropout_rate)
        for param in self.model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    def forward(self, encoder_input,decoder_input,encoder_mask, decoder_mask):
        return self.model(encoder_input, decoder_input, encoder_mask, decoder_mask)
    

   