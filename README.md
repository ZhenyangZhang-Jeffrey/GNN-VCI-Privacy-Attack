# GNN Final Project: VCI Model Privacy Attack Analysis

This project is a final project for the Graph Neural Networks (GNN) course, based on the implementation of the [**Variational Causal Inference**](https://arxiv.org/abs/2410.12730) original paper code.

**For detailed information about the original paper code, please refer to:** https://github.com/yulun-rayn/variational-causal-inference

## 📋 Project Structure

```
├── vci/                  # VCI original paper framework code
├── attack/               # Final project: Model privacy attack analysis
├── artifact/             # Training results and attack outputs
└── datasets/             # Datasets
```

## 🚀 VCI Model Privacy Attack: Complete Pipeline

This project demonstrates privacy threats from model inversion attacks through frozen encoder latent features, evaluated on the Marson scRNA-seq dataset.

### 📁 Attack Folder Contents

- `prepare_attacker_dataset.py` - Extract latent features Z from VCI encoder
- `train_attacker.py` - Train MLP attacker model (Z → Y)
- `evaluate_attacker.py` - Evaluate attack success on 3 objectives
- `visualize_hvg_attack.py` - Visualize HVG reconstruction (Chinese labels)
- `visualize_hvg_attack_en.py` - Visualize HVG reconstruction (English labels)
- Shell scripts for executing each step

### 🔧 Complete Execution Steps

#### Step 1: Activate Environment
```bash
eval "$(conda shell.bash hook)"
conda activate VCI
cd /path/to/VCI_Project
```

#### Step 2: Prepare Attacker Dataset (Extract Z from frozen encoder)
```bash
python attack/prepare_attacker_dataset.py \
    --checkpoint_vci artifact/saves/3.1_marson_baseline_2026.03.15_15:36:47/model_seed=None_epoch=99.pt \
    --data_path datasets/marson_prepped.h5ad \
    --data_name gene \
    --artifact_path artifact \
    --device cuda
```

**Output**: `artifact/marson_attacker_dataset.pt` 
- Z features: (29374 train, 7344 test) × 128-dim
- Y labels: (36718 total) × 2013-dim

#### Step 3: Train Attacker Model (MLP)
```bash
python attack/train_attacker.py \
    --attacker_dataset artifact/marson_attacker_dataset.pt \
    --artifact_path artifact \
    --device cuda
```

**Output**: `artifact/marson_attack_YYYY.MM.DD_HH:MM:SS/attacker_model.pt`
- Architecture: 128 → 512 → 1024 → 2013

#### Step 4: Evaluate Attack Success
```bash
python attack/evaluate_attacker.py \
    --attacker_dataset artifact/marson_attacker_dataset.pt \
    --model_checkpoint artifact/marson_attack_2026.03.25_14:31:40/attacker_model.pt \
    --device cuda
```

**Output**: `artifact/attack_evaluation_results.json`

Key Results:
- ✅ Beats naive baseline: MSE -15.37%, Cosine Similarity +17.85%
- 🔬 HVG genes exposed: R²=0.336 (target>0.3), Cosine=0.839 (target>0.7)

#### Step 5: Visualize Results (Optional, English version)
```bash
python attack/visualize_hvg_attack_en.py \
    --attacker_dataset artifact/marson_attacker_dataset.pt \
    --model_checkpoint artifact/marson_attack_2026.03.25_14:31:40/attacker_model.pt \
    --output_dir artifact \
    --device cuda
```

**Output**:
- `artifact/hvg_single_cells_en.png` - 6 example cells
- `artifact/hvg_aggregated_en.png` - All 7344 cells × 50 HVG genes
- `artifact/hvg_gene_quality_en.png` - Per-gene error metrics

## 🔍 Key Research Findings

**VCI Model Inversion Attack (Marson Dataset):**
- **Attack Type**: Frozen encoder → Reconstruct gene expression  
- **Input**: 128-dim latent features Z
- **Target**: 2013-dim gene expression Y
- **Privacy Impact**: 50 HVG genes successfully reconstructed with R²=0.336 ✅

## ⚙️ Parameter Configuration

All scripts support the following key parameters:
- `--device cuda` or `--device cpu`
- `--batch_size` (default: 128)
- `--artifact_path` (default: ../artifact)
- `--attacker_dataset` (default: ../artifact/marson_attacker_dataset.pt)
- `--model_checkpoint` (obtained from latest run)

## 📚 Original Paper Framework

For complete documentation on the VCI framework, installation steps, data preparation, and original experiments, please visit the official repository:

→ **[yulun-rayn/variational-causal-inference](https://github.com/yulun-rayn/variational-causal-inference)**

## 📄 License

MIT License
