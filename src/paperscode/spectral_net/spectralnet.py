import torch 
import torch.nn as nn

class SpectralNet(nn.Module):
    def __init__(self, input_dim=784, siamese=None, output_clusters=10, eps=1e-6):
        super().__init__()
        self.input_dim = input_dim
        self.output_clusters = output_clusters
        self.eps = eps

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),

            nn.Linear(1024, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, output_clusters),
        )


    def forward(self, x):
        # x-> (B, 784)
        op = self.model(x) # (B, k)
        S = op.T@op + self.eps*torch.eye(self.output_clusters, device=op.device, dtype=op.dtype) # (k, k)
        L = torch.linalg.cholesky(S) # (k, k)
        op_orth = torch.linalg.solve_triangular(L, op.T, upper=False).T # (B, k)

        return op_orth