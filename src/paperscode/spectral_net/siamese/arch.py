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

    def contrastive_loss(self, x1, x2, y, margin=2.0):
        # y=0 means same class, and y=1 mean different class
        # x1 -> (B, d), x2 -> (B, d), y -> (B, )
        dist = torch.norm(x1-x2, p=2, dim=1, keepdim=False) # (B, )
        return torch.mean(
            (1-y)*(dist**2) + y*((torch.clamp(margin-dist, min=0))**2)
        )

