"""
Evaluate Counterfactuals using trained VAE and Classifier.

This script generates counterfactual samples and evaluates their quality using:
- Validity: Whether predictions match the target
- Proximity: Distance to original samples
- Diversity: Diversity of generated counterfactuals
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from vci.train.prepare import prepare
from vci.utils.data_utils import move_tensors


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate counterfactual explanations"
    )
    
    # Model checkpoint arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--checkpoint_classifier", type=str, default=None,
                        help="Path to trained Classifier checkpoint")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Dataset name: gene, celebA, morphoMNIST")
    parser.add_argument("--label_names", type=str, default=None)
    
    # Model arguments
    parser.add_argument("--hparams", type=str, default=None)
    parser.add_argument("--artifact_path", type=str, default="./artifact",
                        help="Path to save results")
    
    # Evaluation arguments
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_cf_samples", type=int, default=10,
                        help="Number of counterfactuals to generate per sample")
    
    # Output arguments  
    parser.add_argument("--save_cf", action="store_true",
                       help="Save generated counterfactuals")
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model state dict and arguments from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, tuple):
        state_dict, args = checkpoint
    else:
        state_dict = checkpoint
        args = {}
    return state_dict, args


def generate_counterfactuals(model, x, num_samples=10, device="cuda"):
    """
    Generate counterfactual samples for input x.
    
    Args:
        model: Trained VAE model
        x: Input sample(s)
        num_samples: Number of counterfactuals to generate
        device: Device to use
        
    Returns:
        torch.Tensor: Generated counterfactuals of shape (B, num_samples, *x.shape[1:])
    """
    model.eval()
    
    with torch.no_grad():
        x = move_tensors(x, device=device)
        
        # Encode input
        if hasattr(model, 'encode'):
            z = model.encode(x)
        else:
            # Fallback: try to get latents from model
            z = model.cov_emb if hasattr(model, 'cov_emb') else None
        
        if z is None:
            print("Warning: Could not extract latent representation")
            return None
        
        # Generate counterfactuals by sampling from latent space
        batch_size = x.shape[0]
        counterfactuals = []
        
        for _ in range(num_samples):
            # Sample noise
            noise = torch.randn_like(z)
            z_perturbed = z + 0.1 * noise  # Small perturbation
            
            # Decode
            if hasattr(model, 'decode'):
                cf = model.decode(z_perturbed)
            else:
                cf = z_perturbed
            
            counterfactuals.append(cf)
        
        # Stack counterfactuals
        counterfactuals = torch.stack(counterfactuals, dim=1)  # (B, num_samples, ...)
        
    return counterfactuals


def evaluate_counterfactuals(model, datasets, classifier=None, 
                             num_cf_samples=10, batch_size=128, device="cuda"):
    """
    Evaluate quality of generated counterfactuals.
    
    Metrics:
    - Validity: Whether model predictions for CF match intended changes
    - Proximity: L2 distance between original and counterfactual
    - Diversity: Variance among generated counterfactuals
    """
    model.eval()
    
    metrics = {
        "proximity_l2": [],
        "proximity_l1": [],
        "diversity": [],
        "validity": [] if classifier else None,
    }
    
    test_loader = datasets.get("test_loader", None)
    if test_loader is None:
        print("Warning: No test loader found in datasets")
        return metrics
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), 
                                      desc="Evaluating counterfactuals",
                                      total=len(test_loader)):
            x = batch[0]  # Assuming x is first element
            x = move_tensors(x, device=device)
            
            # Generate counterfactuals
            cf = generate_counterfactuals(model, x, num_cf_samples, device)
            
            if cf is None:
                continue
            
            batch_size_actual = x.shape[0]
            x_expanded = x.unsqueeze(1).expand_as(cf)
            
            # Compute proximity metrics
            l2_distance = torch.sqrt(((cf - x_expanded) ** 2).sum(dim=tuple(range(2, cf.ndim))))
            l1_distance = torch.abs(cf - x_expanded).sum(dim=tuple(range(2, cf.ndim)))
            
            metrics["proximity_l2"].extend(l2_distance.mean(dim=1).cpu().numpy())
            metrics["proximity_l1"].extend(l1_distance.mean(dim=1).cpu().numpy())
            
            # Compute diversity (variance across samples)
            cf_std = cf.std(dim=1)
            diversity = cf_std.mean()
            metrics["diversity"].append(diversity.item())
            
            # Compute validity if classifier available
            if classifier is not None and metrics["validity"] is not None:
                # This would require predictions from classifier
                # Placeholder for now
                pass
    
    # Aggregate metrics
    results = {
        "avg_proximity_l2": np.mean(metrics["proximity_l2"]) if metrics["proximity_l2"] else np.nan,
        "avg_proximity_l1": np.mean(metrics["proximity_l1"]) if metrics["proximity_l1"] else np.nan,
        "avg_diversity": np.mean(metrics["diversity"]) if metrics["diversity"] else np.nan,
    }
    
    return results


def main():
    args = parse_arguments()
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load VAE checkpoint
    print(f"[1/3] Loading VAE checkpoint from {args.checkpoint}")
    state_dict, checkpoint_args = load_checkpoint(args.checkpoint, device=device)
    
    # Merge checkpoint args with command line args
    if checkpoint_args:
        for key, val in checkpoint_args.items():
            if key not in args.__dict__ or args.__dict__[key] is None:
                setattr(args, key, val)
    
    # Prepare model and datasets
    print(f"[2/3] Preparing model and datasets...")
    model, datasets = prepare(
        vars(args),
        state_dict=state_dict,
        device=device
    )
    
    # Load classifier if provided
    classifier = None
    if args.checkpoint_classifier:
        print(f"[3/3] Loading Classifier checkpoint...")
        classifier_state_dict, _ = load_checkpoint(args.checkpoint_classifier, device=device)
        print("  [Note] Classifier loaded but full integration pending")
    
    # Evaluate counterfactuals
    print(f"\n[4/4] Evaluating counterfactuals...")
    cf_metrics = evaluate_counterfactuals(
        model,
        datasets,
        classifier=classifier,
        num_cf_samples=args.num_cf_samples,
        batch_size=args.batch_size,
        device=device
    )
    
    # Display results
    print("\n" + "="*60)
    print("COUNTERFACTUAL EVALUATION RESULTS")
    print("="*60)
    
    for metric_name, metric_value in cf_metrics.items():
        if not np.isnan(metric_value):
            print(f"{metric_name:.<40} {metric_value:.6f}")
    
    # Save results
    os.makedirs(args.artifact_path, exist_ok=True)
    json_path = os.path.join(args.artifact_path, "counterfactual_metrics.json")
    with open(json_path, "w") as f:
        json.dump(cf_metrics, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    return cf_metrics


if __name__ == "__main__":
    main()
