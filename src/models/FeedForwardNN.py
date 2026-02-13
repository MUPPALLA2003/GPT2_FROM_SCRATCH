import torch.nn as nn

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.linlayer = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.activation = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linlayer(x)
        x = self.activation(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
