import torch.nn as nn
import math
import torch
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by number of heads"
        self.d_k = d_model // self.num_heads
    def forward(self, q, k, v, encoder_mask=None):
        self.encoder_mask = encoder_mask
        batch_size = q.shape[0]
        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)
        q = q.view(batch_size, q.shape[1], self.num_heads,self.d_k).transpose(1,2)
        k = k.view(batch_size, k.shape[1], self.num_heads,self.d_k).transpose(1,2)
        v = v.view(batch_size, v.shape[1], self.num_heads,self.d_k).transpose(1,2)
        attention_score = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if self.encoder_mask is not None:
            attention_score = attention_score.masked_fill(encoder_mask==0, -1e9)
        attention_weight = torch.softmax(attention_score,dim=-1)
        if self.dropout is not None:
            attention_weight = self.dropout(attention_weight)
        attention_ouput = attention_weight @ v
        attention_ouput = attention_ouput.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads * self.d_k)
        multihead_ouput = self.W_o(attention_ouput)
        return multihead_ouput