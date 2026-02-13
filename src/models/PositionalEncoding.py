import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self,d_model:int,context_len:int):
        super().__init__()
        self.d_model = d_model
        self.context_len = context_len
        self.embedding = nn.Embedding(context_len,d_model)

    def forward(self,x):
        B,T,C = x.shape
        pos = torch.arange(0, self.context_len, device=x.device)
        pos = pos.unsqueeze(0).expand(B,T)
        return self.embedding(pos)