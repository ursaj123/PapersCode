import torch 
import torch.nn as nn

from paperscode.spectral_net.siamese.arch import SiameseTwin

class SpectralNet(nn.Module):
    def __init__(self, input_dim=784, siamese_path=None, affinity_matrix_clusters=10, output_clusters=10, eps=1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.output_clusters = output_clusters
        self.affinity_matrix_clusters = affinity_matrix_clusters
        self.eps = eps
        self.siamese = None


        if siamese_path is not None:
            self.siamese = SiameseTwin()
            ckpt = torch.load(siamese_path, map_location='cpu')
            self.siamese.load_state_dict(ckpt["model"])

            self.siamese.eval()
            for param in self.siamese.parameters():
                param.requires_grad = False


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
        return self.model(x) # (B, k)
        
    def orthogonlize(self, op):
        S = op.T@op + self.eps*torch.eye(self.output_clusters, device=op.device, dtype=op.dtype) # (k, k)
        L = torch.linalg.cholesky(S) # (k, k)
        op_orth = torch.linalg.solve_triangular(L, op.T, upper=False).T # (B, k)

        return op_orth
        
    def pairwise_dist(self, x):
        # calculate ||x_i-x-j||^2
        y = x
        norms = torch.sum(y**2, dim=-1) # (B, )
        cdist = norms[:, None] - 2*y@y.T +  norms[None, :]   # (B, B)
        cdist = cdist.clamp_min(0.0)
        return cdist

    def spectral_loss(self, x_inp, op):
        y = x_inp
        if self.siamese is not None:
            with torch.no_grad():
                y = self.siamese(y)

        # y-> (B, k) 
        cdist = self.pairwise_dist(y)

        sorted_indices = torch.argsort(cdist, dim=-1)#[:, affinity_matrix_clusters] # (B, k)
        sorted_indices = sorted_indices[:, 1:self.affinity_matrix_clusters+1] # removing the self loop, as distance would always be zero
        mask = torch.zeros_like(cdist) # (B, B)
        mask = mask.scatter_(1, sorted_indices, 1.0)


        mean_dist = torch.sum(torch.sqrt(cdist)*mask, dim=-1)/self.affinity_matrix_clusters # (B, ) kepping top k neighbors only
        sigma = (mean_dist[:, None])@(mean_dist[None, :]) + 1e-8 # (B, B)
        

        W = torch.exp(-cdist/(2*sigma))*mask # (B, B)
        W = (W + W.T)/2

        spec_op = self.orthogonlize(op)
        cdist_spec = self.pairwise_dist(spec_op)
        return (1/spec_op.shape[0])*torch.sum(W*cdist_spec)







