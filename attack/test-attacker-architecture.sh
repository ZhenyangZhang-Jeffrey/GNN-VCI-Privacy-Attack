#!/bin/bash

# Task 1.4 Demo: Test AttackerMLP Architecture
# 验证攻击者网络的架构设计和前向传播

eval "$(conda shell.bash hook)"
conda activate VCI
DATA=$(cd -P -- "$(dirname -- "$0")/../" && pwd -P)

echo "=========================================================="
echo "🧪 Testing AttackerMLP Architecture"
echo "=========================================================="

python -c "
import torch
from vci.model.attacker import create_attacker_mlp, AttackerMLP

print('\n' + '='*60)
print('AttackerMLP Architecture Test')
print('='*60 + '\n')

# Create model with different architectures
architectures = ['small', 'default', 'large', 'deep']

for arch in architectures:
    print(f'\\n🏗️ Architecture: {arch.upper()}')
    print('-'*60)
    
    model = create_attacker_mlp(
        latent_dim=128,
        gene_dim=2000,
        architecture=arch,
        dropout_rate=0.1,
        output_activation=None  # Identity activation
    )
    
    # Test forward pass
    z = torch.randn(32, 128)  # Batch of 32 samples
    y_hat = model(z)
    
    print(f'Input shape:  {z.shape}')
    print(f'Output shape: {y_hat.shape}')
    print(f'Parameters:   {sum(p.numel() for p in model.parameters()):,}')
    
    assert y_hat.shape == (32, 2000), f'Unexpected output shape: {y_hat.shape}'
    print(f'✅ Forward pass successful')

print('\\n' + '='*60)
print('🎉 All tests passed!')
print('='*60)
"

echo ""
echo "✅ Model architecture validation complete!"
