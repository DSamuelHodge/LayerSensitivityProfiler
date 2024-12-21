import torch
import torch.nn as nn
from layer_sensitivity_profiler import LayerSensitivityProfiler
import matplotlib.pyplot as plt

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Single attention layer for testing
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        # Linear layer for testing
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        # Apply attention
        attn_out, _ = self.attention(x, x, x)
        # Apply linear
        return self.linear(attn_out)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a simple transformer model
    model = SimpleTransformer(d_model=512, nhead=8, num_layers=2).to(device)
    print("Created SimpleTransformer model")
    
    # Create sample input
    batch_size = 32
    seq_length = 100
    d_model = 512
    sample_input = torch.randn(batch_size, seq_length, d_model).to(device)
    print(f"Created sample input tensor of shape: {sample_input.shape}")
    
    # Initialize the profiler with different bond dimensions to test
    bond_dims = [2, 4, 8, 16, 32]
    profiler = LayerSensitivityProfiler(model, bond_dims, device=device)
    print(f"\nInitialized LayerSensitivityProfiler with bond dimensions: {bond_dims}\n")
    
    print("Profiling model layers...")
    results = profiler.profile_model(sample_input)
    
    # Print results
    print("\nProfiling Results:")
    for layer_name, layer_metrics in results.items():
        print(f"\nLayer: {layer_name}")
        print("Bond Dimension | Sensitivity Score")
        print("-" * 35)
        for bond_dim in bond_dims:
            sensitivity = layer_metrics[bond_dim]['sensitivity']
            print(f"{bond_dim:^13} | {sensitivity:.4f}")
        
        # Get the best compression ratio
        compression_ratios = [metrics['compression_ratio'] for metrics in layer_metrics.values()]
        sensitivities = [metrics['sensitivity'] for metrics in layer_metrics.values()]
        best_idx = min(range(len(sensitivities)), key=lambda i: sensitivities[i])
        best_bond_dim = bond_dims[best_idx]
        best_compression = compression_ratios[best_idx]
        
        print(f"\nOptimal bond dimension: {best_bond_dim}")
        print(f"Compression ratio: {best_compression:.2f}x")
    
    # Plot sensitivity curves
    profiler.plot_sensitivity_curves('sensitivity_curves.png')
    print("\nSensitivity curves have been saved to 'sensitivity_curves.png'")

if __name__ == "__main__":
    main()
