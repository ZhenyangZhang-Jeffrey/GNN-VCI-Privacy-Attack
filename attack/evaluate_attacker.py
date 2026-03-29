"""
Evaluate Attacker Model

Assess if the attack network effectively reconstructs gene expression
through model inversion. Measure performance against baseline and evaluate
reconstruction quality on highly variable genes.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import paired_cosine_distances
from tqdm import tqdm

from vci.model.attacker import create_attacker_mlp


def compute_naive_baseline(y_train, y_test):
    """
    Compute baseline: mean profile of training set.
    Used as comparison point for attack performance.
    
    Args:
        y_train: Training gene expression (n_train, n_genes)
        y_test: Test gene expression (n_test, n_genes)
    
    Returns:
        baseline_pred: Mean profile repeated for test samples
        naive_mse: MSE of baseline
        naive_cosine_sim: Cosine similarity of baseline
        naive_r2: R² score of baseline
    """
    
    # Compute mean expression profile from training set
    mean_profile = np.mean(y_train, axis=0)
    baseline_pred = np.tile(mean_profile, (y_test.shape[0], 1))
    
    # Compute metrics
    naive_mse = mean_squared_error(y_test, baseline_pred)
    cosine_distances = paired_cosine_distances(y_test, baseline_pred)
    naive_cosine_sim = np.mean(1 - cosine_distances)
    
    try:
        naive_r2 = r2_score(y_test, baseline_pred)
    except:
        naive_r2 = 0.0
    
    return baseline_pred, naive_mse, naive_cosine_sim, naive_r2


def compute_cosine_similarity(y_true, y_pred):
    """
    Vectorized cosine similarity computation using sklearn.
    Better captures profile shape similarity than MSE alone.
    
    Args:
        y_true: True gene expression (n_samples, n_genes)
        y_pred: Predicted gene expression (n_samples, n_genes)
    
    Returns:
        mean_cosine_sim: Mean cosine similarity
        cosine_sims: Per-sample cosine similarity
    """
    
    # Vectorized computation: distance -> similarity
    cosine_distances = paired_cosine_distances(y_true, y_pred)
    cosine_sims = 1 - cosine_distances
    
    return np.mean(cosine_sims), cosine_sims


def compute_r2_score(y_true, y_pred):
    """
    Compute R² score averaged per gene.
    Standard approach for evaluating gene expression reconstruction.
    
    Args:
        y_true: True gene expression (n_samples, n_genes)
        y_pred: Predicted gene expression (n_samples, n_genes)
    
    Returns:
        r2: Mean R² across genes
    """
    
    try:
        # Gene-wise average: compute R² per gene, then average
        return r2_score(y_true, y_pred, multioutput='uniform_average')
    except:
        return 0.0


def identify_hvg(y_train, y_test=None, n_hvg=50):
    """
    Select highly variable genes by variance from training set.
    
    Args:
        y_train: Training gene expression (n_train, n_genes)
        y_test: Test gene expression (optional)
        n_hvg: Number of HVG genes
    
    Returns:
        hvg_idx: Indices of HVG genes
        hvg_genes_var: Variance of HVG genes
    """
    
    # Rank genes by variance
    gene_variance = np.var(y_train, axis=0)
    hvg_idx = np.argsort(gene_variance)[-n_hvg:]
    hvg_genes_var = gene_variance[hvg_idx]
    
    return hvg_idx, hvg_genes_var


def evaluate_attacker_model(args):
    """Evaluate attack model performance on test set."""
    
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Load data and model
    attacker_dataset = torch.load(args.attacker_dataset)
    y_train = attacker_dataset["y_train"]
    y_test = attacker_dataset["y_test"]
    z_test = torch.tensor(attacker_dataset["z_test"], dtype=torch.float32)
    
    model = create_attacker_mlp(
        latent_dim=z_test.shape[1],
        gene_dim=y_test.shape[1],
        architecture=args.architecture,
        dropout_rate=args.dropout_rate,
        output_activation=args.output_activation
    )
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Generate predictions
    y_pred_list = []
    with torch.no_grad():
        for i in tqdm(range(0, z_test.shape[0], args.batch_size), desc="Predicting"):
            batch_z = z_test[i:i+args.batch_size].to(device)
            batch_pred = model(batch_z).cpu().numpy()
            y_pred_list.append(batch_pred)
    
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # Objective 1: Beat naive baseline
    print("\n" + "="*70)
    print("Objective 1: Quantitative - Beat Naive Baseline")
    print("="*70)
    
    baseline_pred, naive_mse, naive_cosine_sim, naive_r2 = compute_naive_baseline(y_train, y_test)
    
    attacker_mse = mean_squared_error(y_test, y_pred)
    attacker_cosine_sim, _ = compute_cosine_similarity(y_test, y_pred)
    attacker_r2 = compute_r2_score(y_test, y_pred)
    
    print(f"\n📊 Baseline: {naive_mse:.6f} MSE, {naive_cosine_sim:.6f} cosine sim")
    print(f"🎯 Attack:   {attacker_mse:.6f} MSE, {attacker_cosine_sim:.6f} cosine sim")
    
    mse_improvement = (naive_mse - attacker_mse) / naive_mse * 100
    r2_improvement = attacker_r2 - naive_r2
    print(f"\n📈 Improvements: {mse_improvement:+.2f}% MSE, R² {r2_improvement:+.6f}")
    
    # Objective 2: Strong performance metrics
    print("\n" + "="*70)
    print("Objective 2: Strong Metrics")
    print("="*70)
    
    _, cosine_sims = compute_cosine_similarity(y_test, y_pred)
    print(f"\nCosine Similarity: {np.mean(cosine_sims):.6f} ± {np.std(cosine_sims):.6f}")
    print(f"  {np.sum(np.array(cosine_sims) > 0.9) / len(cosine_sims) * 100:.1f}% samples > 0.9")
    print(f"\nR² Score: {attacker_r2:.6f} (target: > 0.5)")
    
    # Objective 3: Reconstruction on highly variable genes
    print("\n" + "="*70)
    print("Objective 3: HVG Reconstruction Quality")
    print("="*70)
    
    n_hvg_genes = 50
    hvg_idx, _ = identify_hvg(y_train, y_test, n_hvg=n_hvg_genes)
    
    hvg_mse = mean_squared_error(y_test[:, hvg_idx], y_pred[:, hvg_idx])
    hvg_cosine_sim, _ = compute_cosine_similarity(y_test[:, hvg_idx], y_pred[:, hvg_idx])
    hvg_r2 = compute_r2_score(y_test[:, hvg_idx], y_pred[:, hvg_idx])
    
    print(f"\nHVG (n={n_hvg_genes}): MSE={hvg_mse:.6f}, Cosine={hvg_cosine_sim:.6f}, R²={hvg_r2:.6f}")
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    summary = {
        "baseline": {"mse": float(naive_mse), "cosine_sim": float(naive_cosine_sim), "r2": float(naive_r2)},
        "attacker": {"mse": float(attacker_mse), "cosine_sim": float(attacker_cosine_sim), "r2": float(attacker_r2)},
        "hvg": {"mse": float(hvg_mse), "cosine_sim": float(hvg_cosine_sim), "r2": float(hvg_r2)},
        "mse_improvement_pct": float(mse_improvement)
    }
    
    print(json.dumps(summary, indent=2))
    
    # Save results
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(summary, f, indent=2)
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate attacker model")
    parser.add_argument("--attacker_dataset", type=str, default="../artifact/marson_attacker_dataset.pt",
                        help="Path to attacker dataset")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                        help="Path to trained attacker model checkpoint")
    parser.add_argument("--architecture", type=str, default="default",
                        choices=["small", "default", "large", "deep"])
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--output_activation", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default="../artifact/attack_evaluation_results.json",
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    evaluate_attacker_model(args)
