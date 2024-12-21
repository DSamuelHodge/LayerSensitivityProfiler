import unittest
import torch
import torch.nn as nn
from tdvp import TDVP

class TestTDVP(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.in_features = 64
        self.out_features = 32
        self.bond_dim = 8
        self.batch_size = 16
        self.device = 'cpu'
        
        # Create TDVP instance
        self.tdvp = TDVP(
            in_features=self.in_features,
            out_features=self.out_features,
            bond_dim=self.bond_dim,
            device=self.device
        )
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.in_features)
    
    def test_initialization(self):
        """Test TDVP initialization."""
        # Check dimensions
        self.assertEqual(self.tdvp.in_features, self.in_features)
        self.assertEqual(self.tdvp.out_features, self.out_features)
        self.assertEqual(self.tdvp.bond_dim, self.bond_dim)
        
        # Check tensor shapes
        self.assertEqual(self.tdvp.core_matrix.shape, (self.out_features, self.bond_dim))
        self.assertEqual(self.tdvp.transform_matrix.shape, (self.bond_dim, self.in_features))
    
    def test_forward_pass(self):
        """Test forward pass through TDVP."""
        # Run forward pass
        output = self.tdvp(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.out_features))
        
        # Check output is not all zeros or NaN
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.any(torch.isnan(output)))
    
    def test_effective_weight(self):
        """Test getting effective weight matrix."""
        # Get effective weight
        effective_weight = self.tdvp.get_effective_weight()
        
        # Check shape
        self.assertEqual(effective_weight.shape, (self.out_features, self.in_features))
        
        # Check it matches manual computation
        manual_weight = torch.matmul(self.tdvp.core_matrix, self.tdvp.transform_matrix)
        self.assertTrue(torch.allclose(effective_weight, manual_weight))
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # Calculate number of parameters
        tdvp_params = (self.out_features * self.bond_dim) + (self.bond_dim * self.in_features)
        full_params = self.in_features * self.out_features
        
        # Compression ratio should be > 1 for small bond dimensions
        compression_ratio = full_params / tdvp_params
        self.assertGreater(compression_ratio, 1)

if __name__ == '__main__':
    unittest.main()
