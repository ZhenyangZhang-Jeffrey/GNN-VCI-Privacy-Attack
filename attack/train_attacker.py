"""
Train Attacker MLP for Model Inversion

Train network to reconstruct gene expressions (Y) from latent codes (Z).
Minimize reconstruction error ||Y_pred - Y_true||².
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vci.model.attacker import create_attacker_mlp


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train attacker model")
    parser.add_argument("--attacker_dataset", type=str, default="../artifact/marson_attacker_dataset.pt")
    parser.add_argument("--architecture", type=str, default="default", choices=["small", "default", "large", "deep"])
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--output_activation", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--artifact_path", type=str, default="../artifact")
    parser.add_argument("--exp_name", type=str, default="marson_attack")
    parser.add_argument("--device", type=str, default="cuda")
    
    return parser.parse_args()


class AttackerTrainer:
    """Train and evaluate attacker model."""
    
    def __init__(self, model, args, device="cuda"):
        self.model = model.to(device)
        self.args = args
        self.device = device
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_z, batch_y in tqdm(train_loader, desc="Training"):
            batch_z = batch_z.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            y_pred = self.model(batch_z)
            loss = self.criterion(y_pred, batch_y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Compute validation loss."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_z, batch_y in tqdm(val_loader, desc="Validating"):
                batch_z = batch_z.to(self.device)
                batch_y = batch_y.to(self.device)
                
                y_pred = self.model(batch_z)
                loss = self.criterion(y_pred, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def should_stop(self, val_loss):
        """Check early stopping criterion."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.args.early_stopping_patience
    
    def train(self, train_loader, val_loader):
        """Full training loop with early stopping."""
        for epoch in range(self.args.max_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            if self.should_stop(val_loss):
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model


def train_attacker(args):
    """Load data, create model, train, and save results."""
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    attacker_dataset = torch.load(args.attacker_dataset)
    z_train = torch.tensor(attacker_dataset["z_train"], dtype=torch.float32)
    y_train = torch.tensor(attacker_dataset["y_train"], dtype=torch.float32)
    z_test = torch.tensor(attacker_dataset["z_test"], dtype=torch.float32)
    y_test = torch.tensor(attacker_dataset["y_test"], dtype=torch.float32)
    
    # Split train/val
    n_train = z_train.shape[0]
    n_val = int(n_train * args.validation_ratio)
    n_train = n_train - n_val
    
    indices = np.random.permutation(z_train.shape[0])
    z_train = z_train[indices]
    y_train = y_train[indices]
    
    z_train_split, z_val = z_train[:n_train], z_train[n_train:]
    y_train_split, y_val = y_train[:n_train], y_train[n_train:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_train_split, y_train_split),
        batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_val, y_val),
        batch_size=args.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z_test, y_test),
        batch_size=args.batch_size, shuffle=False
    )
    

    model = create_attacker_mlp(
        latent_dim=z_train.shape[1],
        gene_dim=y_train.shape[1],
        architecture=args.architecture,
        dropout_rate=args.dropout_rate,
        output_activation=args.output_activation
    )
    
    # Train
    trainer = AttackerTrainer(model, args, device=device)
    trained_model = trainer.train(train_loader, val_loader)
    
    # Test
    trained_model.eval()
    test_losses = []
    with torch.no_grad():
        for batch_z, batch_y in tqdm(test_loader, desc="Testing"):
            batch_z = batch_z.to(device)
            batch_y = batch_y.to(device)
            y_pred = trained_model(batch_z)
            loss = nn.MSELoss()(y_pred, batch_y)
            test_losses.append(loss.item())
    
    test_loss = np.mean(test_losses)
    
    # Save
    os.makedirs(args.artifact_path, exist_ok=True)
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_dir = os.path.join(args.artifact_path, f"{args.exp_name}_{dt}")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(trained_model.state_dict(), os.path.join(save_dir, "attacker_model.pt"))
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    train_attacker(args)
