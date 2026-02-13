import torch.nn as nn

class TokenEmbeddings(nn.Module):

    def __init__(self,d_model:int,vocabulary_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size,d_model)

    def forward(self,x):
        return self.embedding(x)