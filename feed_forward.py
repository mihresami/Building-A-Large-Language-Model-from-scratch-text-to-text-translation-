import torch.nn as nn
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float):
        super().__init__()
        self.layer_1 = nn.Linear(d_model, d_ff)
        self.activation_1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, input):
        return self.layer_2(self.dropout(self.activation_1(self.layer_1(input))))