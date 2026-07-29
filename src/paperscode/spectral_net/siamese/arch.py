import torch
import torch.nn as nn

class SiameseTwin(nn.Module):
    def __init__(self, input_dim = 784, d_embed=10):
        super().__init__()
        self.d_embed = d_embed
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024, bias=True), 
            nn.ReLU(),

            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),

            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),

            nn.Linear(512, d_embed, bias=True)
        )

    def forward(self, x):
        # (B, d)
        return self.model(x) 

