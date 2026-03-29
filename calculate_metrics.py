"""
Calculate evaluation metrics on test set by combining VAE and Classifier weights.

This script:
1. Loads trained VAE model and Classifier
2. Evaluates on test split
3. Computes Attribute MAE for do(thickness) and do(intensity)
4. Outputs results in tabular format
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from tqdm import tqdm

from vci.train.prepare import prepare, prepare_classifier
from vci.utils.data_utils import move_tensors

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate metrics on test set using trained VAE and Classifier"
    )
    
    # Model checkpoint arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--checkpoint_classifier", type=str, required=True,
                        help="Path to trained Classifier checkpoint")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--data_name", type=str, required=True,
                        help="Dataset name: gene, celebA, morphoMNIST")
    parser.add_argument("--label_names", type=str, default=None,
                        help="Comma-separated label names")
    
    # Model arguments
    parser.add_argument("--hparams", type=str, default=None,
                        help="Path to hyperparameters JSON file")
    parser.add_argument("--artifact_path", type=str, default="./artifact",
                        help="Path to save results")
    
    # Evaluation arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for evaluation")
    
    return parser.parse_args()


def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model state dict and arguments from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, tuple):
        state_dict, args = checkpoint
    else:
        state_dict = checkpoint
        args = {}
    return state_dict, args


def custom_collate(batch):
    """Custom data collation function, safely filter out NoneType"""
    transposed = zip(*batch)
    collated = []
    for samples in transposed:
        # If the first element in this column is None, the entire Batch's this column is None
        if samples[0] is None:
            collated.append(None)
        # If it's an empty list [] (e.g., no covariates)
        elif isinstance(samples[0], list) and len(samples[0]) == 0:
            collated.append([])
        # Normal Tensor, let the official function concatenate it
        else:
            collated.append(default_collate(samples))
    return tuple(collated)

def calculate_real_morpho_metrics(model, classifier, test_loader, device="cuda"):
    """
    Core logic for calculating Morpho-MNIST counterfactual MAE and MSE
    """
    model.eval()
    classifier.eval()
    
    mae_thickness_list, mae_intensity_list = [], []
    mse_thickness_list, mse_intensity_list = [], []  # New MSE recording list
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calculating Metrics (MAE & MSE)"):
            batch = move_tensors(*batch, device=device)
            x = batch[0]
            labels = batch[1]
            covars = batch[2] if len(batch) > 2 else []
            
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
                
            # ==========================================================
            # Scenario A: Intervene only thickness do(thickness)
            # ==========================================================
            cf_labels_th = labels.clone()
            cf_labels_th[:, 0] = cf_labels_th[torch.randperm(x.size(0)), 0] 
            
            # 1. Model generates counterfactual image
            cf_x_th = model.predict(x, labels, covars, cf_labels_th)
            
            # 2. Calculate MAE
            pred_th = classifier(cf_x_th, covars)
            mae_th = F.l1_loss(pred_th[:, 0], cf_labels_th[:, 0]).item()
            mae_thickness_list.append(mae_th)
            
            # 3. Calculate MSE (approximating true values using morphological operations)
            # Use MaxPool2d for dilation (thickening), use -MaxPool2d(-x) for erosion (thinning)
            # Note: Using basic 3x3 kernel for single-step approximation, real paper may call scipy.ndimage for precise pixel morphology
            diff_th = cf_labels_th[:, 0] - labels[:, 0]
            true_cf_x_th = x.clone()
            for i in range(x.size(0)):
                if diff_th[i] > 0.5:  # Need to thicken
                    true_cf_x_th[i:i+1] = F.max_pool2d(x[i:i+1], kernel_size=3, stride=1, padding=1)
                elif diff_th[i] < -0.5:  # Need to thin
                    true_cf_x_th[i:i+1] = -F.max_pool2d(-x[i:i+1], kernel_size=3, stride=1, padding=1)
            
            mse_th = F.mse_loss(cf_x_th, true_cf_x_th).item()
            mse_thickness_list.append(mse_th)
            
            # ==========================================================
            # Scenario B: Intervene only intensity do(intensity)
            # ==========================================================
            if labels.shape[1] > 1:
                cf_labels_in = labels.clone()
                cf_labels_in[:, 1] = cf_labels_in[torch.randperm(x.size(0)), 1]
                
                # 1. Model generates counterfactual image
                cf_x_in = model.predict(x, labels, covars, cf_labels_in)
                
                # 2. Calculate MAE
                pred_in = classifier(cf_x_in, covars)
                mae_in = F.l1_loss(pred_in[:, 1], cf_labels_in[:, 1]).item()
                mae_intensity_list.append(mae_in)
                
                # 3. Calculate MSE (mathematically exact true value for intensity)
                # True image = original image * (target intensity / original intensity)
                ratio_in = (cf_labels_in[:, 1] / (labels[:, 1] + 1e-7)).view(-1, 1, 1, 1)
                true_cf_x_in = torch.clamp(x * ratio_in, 0.0, 1.0)  # Prevent pixel overflow
                
                mse_in = F.mse_loss(cf_x_in, true_cf_x_in).item()
                mse_intensity_list.append(mse_in)

    # Summarize results dictionary
    results = {
        "Attribute_MAE_Thickness": sum(mae_thickness_list) / len(mae_thickness_list) if mae_thickness_list else float('nan'),
        "Recon_MSE_Thickness": sum(mse_thickness_list) / len(mse_thickness_list) if mse_thickness_list else float('nan')
    }
    
    if mae_intensity_list:
        results["Attribute_MAE_Intensity"] = sum(mae_intensity_list) / len(mae_intensity_list)
        results["Recon_MSE_Intensity"] = sum(mse_intensity_list) / len(mse_intensity_list)
        
    return results


def format_results_table(results):
    """Format results into a readable table."""
    # Convert dict to DataFrame for nice formatting
    df = pd.DataFrame([results])
    df.insert(0, 'Dataset', 'Morpho-MNIST Test Split')
    return df


def main():
    args = parse_arguments()
    
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ---------------------------------------------------------
    # 1. Load VAE (Generator)
    # ---------------------------------------------------------
    print(f"\n[1/4] Loading VAE checkpoint from {args.checkpoint}")
    state_dict, checkpoint_args = load_checkpoint(args.checkpoint, device=device)
    
    # Merge checkpoint args with command line args
    if checkpoint_args:
        for key, val in checkpoint_args.items():
            if key not in args.__dict__ or args.__dict__[key] is None:
                setattr(args, key, val)
    
    print(f"[2/4] Preparing VAE model and datasets...")
    model, datasets = prepare(vars(args), state_dict=state_dict, device=device)
    
    # ---------------------------------------------------------
    # 2. Load Classifier (Referee)
    # ---------------------------------------------------------
    print(f"[3/4] Loading Classifier checkpoint from {args.checkpoint_classifier}")
    classifier_state_dict, classifier_ckpt_args = load_checkpoint(args.checkpoint_classifier, device=device)
    
    # 【Core Fix】: Must rebuild the network using the classifier's training parameters, absolutely cannot let VAE parameters pollute it!
    clf_args_dict = vars(args).copy()
    if classifier_ckpt_args:
        clf_args_dict.update(classifier_ckpt_args)
        
    # Use the merged clf_args_dict to build and load the classifier network
    classifier, _ = prepare_classifier(clf_args_dict, state_dict=classifier_state_dict, device=device)
    classifier = classifier.to(device)
    
    # ---------------------------------------------------------
    # 3. Start counterfactual evaluation
    # ---------------------------------------------------------
    print(f"[4/4] Calculating Attribute MAE on test set...")
    
    # 【Core Fix】: If the original code doesn't provide test_loader, we create one from datasets["test"]!
    if "test_loader" in datasets:
        test_loader = datasets["test_loader"]
    elif "test" in datasets:
        print("  [Info] Building DataLoader from datasets['test']...")
        # 【Modify here】: Add collate_fn
        test_loader = DataLoader(
            datasets["test"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=custom_collate
        )
    else:
        raise ValueError("Neither 'test_loader' nor 'test' found in datasets dict!")
        
    results = calculate_real_morpho_metrics(
        model=model, 
        classifier=classifier, 
        test_loader=test_loader, 
        device=device
    )
    
    # ---------------------------------------------------------
    # 4. Print and save results
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🚀 EVALUATION RESULTS (TABLE 2 METRICS)")
    print("="*60)
    
    df_results = format_results_table(results)
    print(df_results.to_string(index=False))
    
    # Save results
    os.makedirs(args.artifact_path, exist_ok=True)
    output_path = os.path.join(args.artifact_path, "evaluation_results_MAE.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
if __name__ == "__main__":
    main()