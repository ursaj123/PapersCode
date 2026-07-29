import torch

def pairwise_dist(x):
    # calculate ||x_i-x-j||^2
    y = x.clone()
    norms = torch.sum(y**2, dim=-1) # (B, )
    cdist = norms[:, None] - 2*y@y.T +  norms[None, :]   # (B, B)
    cdist = cdist.clamp_min(1e-8)
    return cdist

def spectral_loss(x_inp, spec_op, input_dim=784, d_embed=10, affinity_matrix_clusters=10, siamese=None):
    """
    """
    y = x_inp.clone()
    if siamese is not None:
        y = siamese(y)

    # y-> (B, k) 
    cdist = pairwise_dist(y)

    sorted_indices = torch.argsort(cdist, dim=-1)#[:, affinity_matrix_clusters] # (B, k)
    sorted_indices = sorted_indices[:, 1:affinity_matrix_clusters+1] # removing the self loop, as distance would always be zero
    mask = torch.zeros_like(cdist) # (B, B)
    mask = mask.scatter_(1, sorted_indices, 1.0)


    mean_dist = torch.sum(torch.sqrt(cdist)*mask, dim=-1)/affinity_matrix_clusters # (B, ) kepping top k neighbors only
    sigma = (mean_dist[:, None])@(mean_dist[None, :]) + 1e-8 # (B, B)
    

    W = torch.exp(-cdist/sigma)*mask # (B, B)
    W = (W + W.T)/2

    cdist_spec = pairwise_dist(spec_op)
    return (1/spec_op.shape[0])*torch.sum(W*cdist_spec)







