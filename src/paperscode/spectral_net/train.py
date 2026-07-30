import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import argparse



from paperscode.spectral_net.spectralnet import SpectralNet
from paperscode.spectral_net.utils import *
from paperscode.common.trainer import Trainer, TrainerConfig
from paperscode.common.cli import add_trainer_args, trainer_config_from_args
from torch.nn.parallel import DistributedDataParallel as DDP

class SpectralNetTrainer(Trainer):
    def compute_loss(self, batch):
        x, _ = batch
        x = x.reshape(x.shape[0], -1).to(self.device)
        spec_op = self.model(x)

        loss = self.model.module.spectral_loss(x, spec_op) if hasattr(self.model, "module") else self.model.spectral_loss(x, spec_op)
        metrics = {
            "loss": loss.item(),
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

    group = parser.add_argument_group("SpectralNet")

    group.add_argument("--siamese-path", type=str)
    group.add_argument("--affinity-matrix-clusters", type=int, default=10)
    group.add_argument("--output-clusters", type=int, default=10)

    args = parser.parse_args()
    cfg = trainer_config_from_args(args)


    
    # loading dataset and dataloaders
    tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=4)

    

    model = SpectralNet(
        siamese_path=args.siamese_path,
        affinity_matrix_clusters = args.affinity_matrix_clusters,
        output_clusters = args.output_clusters
    )
    trainer = SpectralNetTrainer(model, cfg)
    trainer.fit(train_loader, val_loader)
    # trainer.load_best()
    print("Best val_loss:", trainer._best_metric)




