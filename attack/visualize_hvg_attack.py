"""
Visualize HVG Gene Reconstruction

Generate comparison plots between true and reconstructed
gene expression for highly variable genes.
"""

import os
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress
from sklearn.metrics import r2_score
from tqdm import tqdm

from vci.model.attacker import create_attacker_mlp


def identify_hvg(y_train, n_hvg=50):
    """Select top variable genes by variance."""
    gene_variance = np.var(y_train, axis=0)
    hvg_idx = np.argsort(gene_variance)[-n_hvg:]
    return hvg_idx[::-1]


def load_data(attacker_dataset_path, model_checkpoint_path, device="cuda"):
    """Load dataset and trained model, generate predictions."""
    dataset = torch.load(attacker_dataset_path)
    z_test = torch.tensor(dataset['z_test'], dtype=torch.float32)
    y_test = torch.tensor(dataset['y_test'], dtype=torch.float32)
    y_train = torch.tensor(dataset['y_train'], dtype=torch.float32)
    
    model = create_attacker_mlp(
        latent_dim=z_test.shape[1],
        gene_dim=y_test.shape[1],
        architecture="default",
        dropout_rate=0.1
    )
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model = model.to(device).eval()
    
    y_pred_list = []
    with torch.no_grad():
        for i in tqdm(range(0, z_test.shape[0], 128), desc="Predicting"):
            batch_z = z_test[i:i+128].to(device)
            y_pred_list.append(model(batch_z).cpu().numpy())
    
    y_pred = np.concatenate(y_pred_list, axis=0)
    return y_train.numpy(), y_test.numpy(), y_pred


def plot_single_cell_hvg(y_true_hvg, y_pred_hvg, cell_idx, ax, gene_names=None):
    """Plot single cell HVG reconstruction."""
    n_hvg = len(y_true_hvg)
    
    ax.scatter(y_true_hvg, y_pred_hvg, alpha=0.6, s=50, color='steelblue', edgecolors='navy')
    
    min_val = min(y_true_hvg.min(), y_pred_hvg.min())
    max_val = max(y_true_hvg.max(), y_pred_hvg.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x', alpha=0.7)
    
    slope, intercept, r_value, _, _ = linregress(y_true_hvg, y_pred_hvg)
    x_fit = np.array([min_val, max_val])
    ax.plot(x_fit, slope * x_fit + intercept, 'g-', lw=2.5, label=f'Fit R²={r_value**2:.3f}', alpha=0.8)
    
    mae = np.mean(np.abs(y_true_hvg - y_pred_hvg))
    ax.set_xlabel('True Expression', fontsize=10)
    ax.set_ylabel('Reconstructed', fontsize=10)
    ax.set_title(f'Cell #{cell_idx} ({n_hvg} genes), R²={r_value**2:.4f}, MAE={mae:.4f}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_aspect('equal', adjustable='box')


def plot_aggregated_hvg(y_true_hvg, y_pred_hvg, ax):
    """Plot aggregated HVG reconstruction across all cells."""
    y_true_flat = y_true_hvg.flatten()
    y_pred_flat = y_pred_hvg.flatten()
    
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=20, color='steelblue')
    
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='y=x', alpha=0.8)
    
    slope, intercept, r_value, _, _ = linregress(y_true_flat, y_pred_flat)
    x_fit = np.array([min_val, max_val])
    ax.plot(x_fit, slope * x_fit + intercept, 'g-', lw=3, label=f'Fit R²={r_value**2:.3f}', alpha=0.85)
    
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    ax.set_xlabel('True Expression', fontsize=12)
    ax.set_ylabel('Reconstructed', fontsize=12)
    ax.set_title(f'All Cells - HVG Quality\nR²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_aspect('equal', adjustable='box')


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    
    y_train, y_test, y_pred = load_data(args.attacker_dataset, args.model_checkpoint, device=device)
    
    hvg_idx = identify_hvg(y_train, n_hvg=50)
    y_test_hvg = y_test[:, hvg_idx]
    y_pred_hvg = y_pred[:, hvg_idx]
    
    # Single cells
    n_cells_to_plot = 6
    cell_indices = np.random.choice(y_test.shape[0], n_cells_to_plot, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for plot_idx, cell_idx in enumerate(cell_indices):
        plot_single_cell_hvg(y_test_hvg[cell_idx], y_pred_hvg[cell_idx], cell_idx, axes[plot_idx])
    
    fig.suptitle('HVG Gene Reconstruction (Single Cells)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    output_path_1 = os.path.join(args.output_dir, "hvg_single_cells.png")
    plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aggregated
    fig, ax = plt.subplots(figsize=(10, 9))
    plot_aggregated_hvg(y_test_hvg, y_pred_hvg, ax)
    fig.suptitle(f'HVG Reconstruction (All Cells, n={y_test.shape[0]})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path_2 = os.path.join(args.output_dir, "hvg_aggregated.png")
    plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-gene errors
    # ============================================================
    gene_errors = np.abs(y_test_hvg - y_pred_hvg).mean(axis=0)
    gene_r2_scores = []
    
    for gene_idx in range(y_test_hvg.shape[1]):
        r2 = r2_score(y_test_hvg[:, gene_idx], y_pred_hvg[:, gene_idx])
        gene_r2_scores.append(r2)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top errors
    top_genes_idx = np.argsort(gene_errors)[::-1][:10]
    axes[0].barh(range(len(top_genes_idx)), gene_errors[top_genes_idx], color='coral', edgecolor='darkred', alpha=0.7)
    axes[0].set_yticks(range(len(top_genes_idx)))
    axes[0].set_yticklabels([f'Gene {hvg_idx[i]}' for i in top_genes_idx])
    axes[0].set_xlabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    axes[0].set_title('Top 10 Genes by Error', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Top R² scores
    top_genes_r2_idx = np.argsort(gene_r2_scores)[::-1][:10]
    colors = ['green' if score > 0.3 else 'orange' if score > 0 else 'red' for score in np.array(gene_r2_scores)[top_genes_r2_idx]]
    axes[1].barh(range(len(top_genes_r2_idx)), np.array(gene_r2_scores)[top_genes_r2_idx], color=colors, edgecolor='darkgreen', alpha=0.7)
    axes[1].set_yticks(range(len(top_genes_r2_idx)))
    axes[1].set_yticklabels([f'Gene {hvg_idx[i]}' for i in top_genes_r2_idx])
    axes[1].set_xlabel('R² Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Top 10 Genes by R² Score', fontsize=12, fontweight='bold')
    axes[1].axvline(x=0.3, color='red', linestyle='--', lw=2, alpha=0.7, label='Target (R²>0.3)')
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    output_path_3 = os.path.join(args.output_dir, "hvg_gene_quality.png")
    plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save stats
    stats = {
        "n_cells": int(y_test.shape[0]),
        "n_hvg_genes": len(hvg_idx),
        "hvg_r2": float(r2_score(y_test_hvg.flatten(), y_pred_hvg.flatten())),
        "hvg_mse": float(np.mean(np.square(y_test_hvg - y_pred_hvg))),
        "hvg_mae": float(np.mean(np.abs(y_test_hvg - y_pred_hvg))),
        "gene_r2_mean": float(np.mean(gene_r2_scores)),
        "genes_r2_gt_03": int(np.sum(np.array(gene_r2_scores) > 0.3))
    }
    
    output_stats_path = os.path.join(args.output_dir, "hvg_visualization_stats.json")
    with open(output_stats_path, 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HVG reconstruction")
    parser.add_argument("--attacker_dataset", type=str, default="../artifact/marson_attacker_dataset.pt")
    parser.add_argument("--model_checkpoint", type=str, default="../artifact/marson_attack_2026.03.25_14:31:40/attacker_model.pt")
    parser.add_argument("--output_dir", type=str, default="../artifact")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
