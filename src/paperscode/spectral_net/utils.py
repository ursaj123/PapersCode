import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from sklearn.cluster import KMeans
from paperscode.spectral_net.spectralnet import SpectralNet


def analyze_training_metrics(run_dir_name):
    if not os.path.exists(run_dir_name):
        return
    if not 'history.json' in os.listdir(run_dir_name):
        return
    
    with open(run_dir_name + '/history.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    if len(data)==0:
        return
    
    epochs = list(range(len(data)))
    metrics = []
    for m in data[0].keys():
        if 'train_' in m:
            metrics.append(m.replace('train_', ''))
    metrics = list(sorted(metrics))


    num_cols = 3
    if len(metrics)==2:
        num_cols = 2
    elif len(metrics)==1:
        num_cols = 1


    fig = plt.figure(figsize=(15, 15))
    for idx, m in enumerate(metrics):
        train_m = [i["train_" + m] for i in data]
        val_m = [i["val_" + m] for i in data]

        plt.subplot((len(metrics)-1)//3+1, num_cols, idx+1)
        plt.plot(epochs, train_m, label='train_'+m)
        plt.plot(epochs, val_m, label='val_'+m)
        plt.title(m)
        plt.legend()
        plt.grid()

    plt.tight_layout()
    output_path = os.path.join(run_dir_name, "metrics.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)
    

    


def kmeans_eval(model, loader, device=torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
)):
    # model must be loaded and be on device and in eval mode
    # make large batches

    embeddings, labels = [], []
    for idx, batch in tqdm(enumerate(loader)):
        x, y = batch
        x = x.reshape(x.shape[0], -1).to(device)
        with torch.no_grad():
            op = model(x) # (B, k)
            op = model.orthogonlize(op) # (B, k)

        embeddings.append(op)
        labels.append(y)

    embeddings = torch.vstack(embeddings) # (data_shape, 10)
    embeddings = embeddings.detach().cpu().numpy()
    labels = torch.cat(labels).numpy()

    kmeans = KMeans(
        n_clusters=10,
        n_init=100,
        random_state=0
    ).fit(embeddings)
    preds = kmeans.labels_

    # Hungarian Algo
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / labels.shape[0]


    return acc






