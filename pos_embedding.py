import torch.nn as nn
import math
import torch
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int, dropout_rate: float):
        super().__init__()
        # PositionalEncoding is initialized with max_seq_len, d_model and dropout_rate as input parameters. 
        self.dropout = nn.Dropout(dropout_rate)
        self.pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # print(f"Position Nan: {torch.isnan(pos).any()}")
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * (math.log(10000)/d_model)
        div_term = torch.clamp(div_term,min=1e-9, max=1e9)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pe[:, 0::2] = torch.sin(pos * div_term)
        self.pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = self.pe.unsqueeze(0).to(device)  
    
    def forward(self, input_embdding):
       
        input_embdding = input_embdding + (self.pe[:, :input_embdding.shape[1], :]).requires_grad_(False)  
        return self.dropout(input_embdding)