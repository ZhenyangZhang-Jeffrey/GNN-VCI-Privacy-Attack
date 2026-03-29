"""
Build Attacker Dataset

Extract latent representations (Z) from frozen VCI encoder
for all cells. Pair with original gene expressions (Y) and donor labels.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
import anndata as h5ad_module
import scanpy as sc
from tqdm import tqdm

from vci.model.model import load_VCI
from vci.dataset.dataset import load_dataset_splits
from vci.utils.data_utils import move_tensors


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build attacker dataset")
    parser.add_argument("--checkpoint_vci", type=str, required=True, help="VCI model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Marson dataset")
    parser.add_argument("--data_name", type=str, default="gene", help="Dataset name")
    parser.add_argument("--artifact_path", type=str, default="../artifact", help="Output directory")
    parser.add_argument("--output_filename", type=str, default="marson_attacker_dataset.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    
    return parser.parse_args()


def extract_latent_features(vci_model, dataset, device="cuda", batch_size=128):
    """
    Extract latent codes Z from frozen VCI encoder.
    
    Args:
        vci_model: Trained VCI model
        dataset: PyTorch dataset
        device: Compute device
        batch_size: Batch size
    
    Returns:
        z_features: Latent codes (n_samples, latent_dim)
        original_outcomes: Gene expressions (n_samples, n_genes)
        donor_indicators: Donor labels
    """
    
    def custom_collate_fn(batch):
        """Custom collate function to handle None values in batch elements."""
        # Filter out items that are None (entire entries)
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        
        # Separate each element of the tuple
        outcomes = []
        treatments = []
        covariates_list = []
        donor_ids = []
        extras = []
        
        for item in batch:
            if len(item) >= 5:
                o, t, c, d, e = item[0], item[1], item[2], item[3], item[4]
            elif len(item) >= 4:
                o, t, c, d = item[0], item[1], item[2], item[3]
                e = None
            else:
                continue  # Skip malformed items
            
            # Skip items with None values in critical fields
            if o is None or t is None or c is None or d is None:
                continue
            
            outcomes.append(o)
            treatments.append(t)
            covariates_list.append(c)
            donor_ids.append(d)
            extras.append(e)
        
        if len(outcomes) == 0:
            return None
        
        # Stack outcomes and treatments
        outcomes = torch.stack(outcomes) if isinstance(outcomes[0], torch.Tensor) else torch.from_numpy(np.array(outcomes))
        treatments = torch.stack(treatments) if isinstance(treatments[0], torch.Tensor) else torch.from_numpy(np.array(treatments))
        
        # Handle covariates (list of tensors)
        if isinstance(covariates_list[0], list):
            num_covariates = len(covariates_list[0])
            covariates = []
            for i in range(num_covariates):
                cov_i = [c[i] for c in covariates_list]
                if all(cv is not None for cv in cov_i):
                    if isinstance(cov_i[0], torch.Tensor):
                        covariates.append(torch.stack(cov_i))
                    else:
                        covariates.append(torch.from_numpy(np.array(cov_i)))
        else:
            covariates = covariates_list
        
        # Stack donor_ids
        donor_ids_tensor = torch.tensor(donor_ids) if isinstance(donor_ids[0], (int, np.integer)) else torch.stack(donor_ids)
        
        return (outcomes, treatments, covariates, donor_ids_tensor, extras)
    
    vci_model.eval()
    
    z_list = []
    y_list = []
    donor_list = []
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Z"):
            if batch is None:
                continue
                
            # Handle flexible batch structure
            if len(batch) >= 5:
                outcomes, treatments, covariates, donor_ids, _ = batch[:5]
            else:
                outcomes, treatments, covariates, donor_ids = batch[:4]
            
            # Move to device
            outcomes = outcomes.to(device)
            treatments = treatments.to(device)
            covariates = [c.to(device) if isinstance(c, torch.Tensor) else c 
                         for c in covariates]
            
            # Extract latent features from frozen encoder
            latents_dist = vci_model.encode(outcomes, treatments, covariates)
            z = latents_dist.mean  # Use mean of latent distribution
            
            # Collect results
            z_list.append(z.cpu().numpy())
            y_list.append(outcomes.cpu().numpy())
            donor_list.append(donor_ids.cpu().numpy() if torch.is_tensor(donor_ids) 
                            else donor_ids)
    
    # Concatenate all batches
    z_features = np.concatenate(z_list, axis=0)
    original_outcomes = np.concatenate(y_list, axis=0)
    donor_indicators = np.concatenate(donor_list, axis=0)
    

    return z_features, original_outcomes, donor_indicators


def build_attacker_dataset(args):
    """
    Build attacker dataset from frozen VCI encoder outputs.
    
    Dataset structure:
    - z_features: Latent codes from VCI encoder
    - y_original: Original gene expressions
    - donor_indicator: Donor identity labels
    """
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    vci_checkpoint = torch.load(args.checkpoint_vci, map_location="cpu")
    vci_state_dict, vci_args = vci_checkpoint
    vci_model = load_VCI(vci_args, state_dict=vci_state_dict, device=device)
    
    datasets = load_dataset_splits(args.data_name, args.data_path, label_names=None, sample_cf=False)
    z_train, y_train, donor_train = extract_latent_features(vci_model, datasets["train"], device=device, batch_size=args.batch_size)
    z_test, y_test, donor_test = extract_latent_features(vci_model, datasets["test"], device=device, batch_size=args.batch_size)
    
    # Build dataset
    attacker_dataset = {
        "z_train": z_train,
        "y_train": y_train,
        "donor_train": donor_train,
        "z_test": z_test,
        "y_test": y_test,
        "donor_test": donor_test,
        "latent_dim": z_train.shape[1],
        "n_genes": y_train.shape[1],
        "n_donors": len(np.unique(np.concatenate([donor_train, donor_test]))),
    }
    
    # Save
    os.makedirs(args.artifact_path, exist_ok=True)
    output_path = os.path.join(args.artifact_path, args.output_filename)
    torch.save(attacker_dataset, output_path)
    
    return attacker_dataset


if __name__ == "__main__":
    args = parse_arguments()
    build_attacker_dataset(args)
