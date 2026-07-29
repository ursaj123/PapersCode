import torch
def contrastive_loss(x1, x2, y, margin=2.0):
    # y=0 means same class, and y=1 mean different class
    # x1 -> (B, d), x2 -> (B, d), y -> (B, )
    dist = torch.norm(x1-x2, p=2, dim=1, keepdim=False) # (B, )
    return torch.mean(
        (1-y)*(dist**2) + y*((torch.clamp(margin-dist, min=0))**2)
    )


