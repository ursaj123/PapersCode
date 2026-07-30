import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import argparse



from paperscode.spectral_net.siamese.arch import SiameseTwin
from paperscode.common.trainer import Trainer, TrainerConfig
from paperscode.common.cli import add_trainer_args, trainer_config_from_args

class SiameseTwinTrainer(Trainer):
    def compute_loss(self, batch):
        x, y = batch
        x = x.to(self.device)
        x = x.reshape(x.shape[0], -1)
        y = y.to(self.device)

        bs = x.shape[0]

        # (B, B): True if same class
        same_class = y[:, None] == y[None, :]
        same_class.fill_diagonal_(False)

        # Keep only upper triangle to avoid (i,j) and (j,i)
        upper = torch.triu(torch.ones_like(same_class, dtype=torch.bool), diagonal=1)

        positive_mask = same_class & upper
        negative_mask = (~same_class) & upper

        positive_pairs = torch.where(positive_mask)
        negative_pairs = torch.where(negative_mask)

        # Ensure enough pairs exist
        num_pos = min(bs // 2, len(positive_pairs[0]))
        num_neg = min(bs // 2, len(negative_pairs[0]))

        # Sample WITHOUT replacement
        pos_idx = torch.randperm(len(positive_pairs[0]), device=self.device)[:num_pos]
        neg_idx = torch.randperm(len(negative_pairs[0]), device=self.device)[:num_neg]

        # Positive pairs
        pos_x1 = x[positive_pairs[0][pos_idx]]
        pos_x2 = x[positive_pairs[1][pos_idx]]

        # Negative pairs
        neg_x1 = x[negative_pairs[0][neg_idx]]
        neg_x2 = x[negative_pairs[1][neg_idx]]

        # Combine
        x1 = torch.cat([pos_x1, neg_x1], dim=0)
        x2 = torch.cat([pos_x2, neg_x2], dim=0)

        # Pass through Siamese network
        z1 = self.model(x1)
        z2 = self.model(x2)

        # Labels: 0 = positive, 1 = negative
        labels = torch.cat([
            torch.zeros(num_pos, device=self.device),
            torch.ones(num_neg, device=self.device)
        ])

        model = self.model.module if hasattr(self.model, "module") else self.model
        loss = model.contrastive_loss(z1, z2, labels) 
        dist = torch.norm(z1 - z2, p=2, dim=1)

        metrics = {
            "loss": loss.item(),
            "pos_dist": dist[labels == 0].mean().item(),
            "neg_dist": dist[labels == 1].mean().item(),
        }

        return loss, metrics

if __name__=='__main__':
    '''
    we can add more args specific to siamese twin
    but since I'm using a specific dataset, with everything fixed
    I would not take any arguments related to siamesetwins
    '''

    parser = argparse.ArgumentParser()
    parser = add_trainer_args(parser)
    
    # group = parser.add_argument_group('SiameseTwin')
    # group.add_argument("--k")

    args = parser.parse_args()
    cfg = trainer_config_from_args(args)


    
    # loading dataset and dataloaders
    tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=4)

    

    model = SiameseTwin()
    trainer = SiameseTwinTrainer(model, cfg)
    trainer.fit(train_loader, val_loader)
    # trainer.load_best()
    print("Best val_loss:", trainer._best_metric)




