import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader



from paperscode.spectral_net.siamese.arch import SiameseTwin
from paperscode.spectral_net.siamese.loss import contrastive_loss
from paperscode.trainer import Trainer, TrainerConfig

class SiameseTwinTrainer(Trainer):
    def compute_loss(self, batch):
        x, y = batch
        x = x.to(self.device)
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

        return contrastive_loss(z1, z2, labels)

if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--lr", type=float, default=0.001
    # )
    # parser.add_argument("--epochs", type=float, default=20)
    # parser.add_argument("--bs", type=int, default=32, help="Batch Size")
    # parser.add_argument("--inputdim", type=int, default=784, help='Data input size')
    # args = parser.parse_args()

    tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST("data", train=True,  download=True, transform=tf)
    val_ds   = datasets.MNIST("data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=4)

    cfg = TrainerConfig(
        output_dir="runs",
        run_name="mnist_test",
        max_epochs=5,
        lr=1e-3,
        optimizer="adamw",
        precision="bfloat16",           # bfloat16 AMP
        grad_clip_norm=1.0,
        scheduler="cosine",
        early_stopping=True,
        early_stopping_monitor="val_loss",
        early_stopping_patience=3,
        save_best=True,
        ema=True,
        ema_decay=0.999,
        seed=42,
        # Logging
        use_log_file=True,              # → runs/mnist_test/trainer.log
        log_level="INFO",
        # Progress bars (requires: pip install tqdm)
        progress_bar=True,              # per-batch bar with live loss
        progress_bar_epochs=True,       # outer epoch bar
    )


    model = SiameseTwin()
    trainer = SiameseTwinTrainer(model, cfg)
    trainer.fit(train_loader, val_loader)
    # trainer.load_best()
    print("Best val_loss:", trainer._best_metric)




