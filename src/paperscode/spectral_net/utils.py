import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans

def kmeans_eval(loader, model, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
)):
    # model must be loaded and be on device and in eval mode
    # make large batches

    for idx, batch in tqdm(enumerate(loader)):
        inp = batch.to(device) # (B, d)
        op = model(inp)

        inp = inp/inp.sum(dim=-1) # (B, d)

        KMeans(n_clusters=op.shape[-1], )



        pass

    pass