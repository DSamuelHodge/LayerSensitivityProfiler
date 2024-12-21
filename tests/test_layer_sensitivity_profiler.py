import unittest
import torch
import torch.nn as nn
from layer_sensitivity_profiler import LayerSensitivityProfiler

class SimpleModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(1)  # Add sequence length dimension
        x, _ = self.attention(x, x, x)
        return x

class TestLayerSensitivityProfiler(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.input_dim = 64
        self.hidden_dim = 32
        self.batch_size = 16
        self.device = 'cpu'
        self.bond_dims = [2, 4, 8]
        
        # Create model
        self.model = SimpleModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        # Create profiler
        self.profiler = LayerSensitivityProfiler(
            model=self.model,
            bond_dimensions=self.bond_dims,
            device=self.device
        )
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.input_dim)
    
    def test_initialization(self):
        """Test profiler initialization."""
        # Check device and bond dimensions
        self.assertEqual(self.profiler.device, self.device)
        self.assertEqual(self.profiler.bond_dimensions, self.bond_dims)
        
        # Check compression handlers
        self.assertIn(nn.Linear, self.profiler.compression_handlers)
        self.assertIn(nn.MultiheadAttention, self.profiler.compression_handlers)
    
    def test_linear_compression(self):
        """Test compression of linear layer."""
        # Get linear layer
        linear = self.model.linear
        
        # Test compression with different bond dimensions
        for bond_dim in self.bond_dims:
            compressed = self.profiler._compress_linear_layer(linear, bond_dim)
            
            # Check output shape
            test_input = torch.randn(self.batch_size, linear.in_features)
            output = compressed(test_input)
            self.assertEqual(output.shape, (self.batch_size, linear.out_features))
    
    def test_attention_compression(self):
        """Test compression of attention layer."""
        # Get attention layer
        attention = self.model.attention
        
        # Test compression with different bond dimensions
        for bond_dim in self.bond_dims:
            compressed = self.profiler._compress_attention_layer(attention, bond_dim)
            
            # Check output shape
            seq_len = 4
            test_input = torch.randn(self.batch_size, seq_len, self.hidden_dim)
            output, _ = compressed(test_input, test_input, test_input)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.hidden_dim))
    
    def test_sensitivity_computation(self):
        """Test sensitivity score computation."""
        # Create test tensors
        original = torch.randn(10, 5)
        compressed = original + 0.1 * torch.randn_like(original)  # Small perturbation
        
        # Compute sensitivity
        sensitivity = self.profiler._compute_sensitivity(original, compressed)
        
        # Check sensitivity is reasonable
        self.assertGreater(sensitivity, 0)
        self.assertLess(sensitivity, 2)  # Should be close to 1 for small perturbations
    
    def test_model_profiling(self):
        """Test full model profiling."""
        # Profile model
        results = self.profiler.profile_model(self.sample_input)
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('linear', results)
        self.assertIn('attention', results)
        
        # Check each layer's results
        for layer_name, layer_results in results.items():
            self.assertIn('sensitivity_scores', layer_results)
            self.assertIn('compression_ratios', layer_results)
            self.assertIn('optimal_bond_dim', layer_results)
            
            # Check lengths match bond dimensions
            self.assertEqual(len(layer_results['sensitivity_scores']), len(self.bond_dims))
            self.assertEqual(len(layer_results['compression_ratios']), len(self.bond_dims))
            
            # Check optimal bond dimension is valid
            self.assertIn(layer_results['optimal_bond_dim'], self.bond_dims)

if __name__ == '__main__':
    unittest.main()
