import unittest
import torch
import torch.nn as nn
from mps import MPS

class TestMPS(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.input_dim = 64
        self.output_dim = 32
        self.bond_dim = 8
        self.batch_size = 16
        self.device = 'cpu'
        
        # Create MPS instance
        self.mps = MPS(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            bond_dimension=self.bond_dim,
            device=self.device
        )
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.input_dim)
    
    def test_initialization(self):
        """Test MPS initialization."""
        # Check dimensions
        self.assertEqual(self.mps.input_dim, self.input_dim)
        self.assertEqual(self.mps.output_dim, self.output_dim)
        self.assertEqual(self.mps.bond_dimension, self.bond_dim)
        
        # Check tensor shapes
        self.assertEqual(self.mps.left_tensor.shape, (self.input_dim, self.bond_dim))
        self.assertEqual(self.mps.right_tensor.shape, (self.bond_dim, self.output_dim))
        self.assertEqual(self.mps.bias.shape, (self.output_dim,))
    
    def test_forward_pass(self):
        """Test forward pass through MPS."""
        # Run forward pass
        output = self.mps(self.sample_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Check output is not all zeros or NaN
        self.assertFalse(torch.all(output == 0))
        self.assertFalse(torch.any(torch.isnan(output)))
    
    def test_effective_matrix(self):
        """Test getting effective weight matrix."""
        # Get effective matrix
        effective_matrix = self.mps.get_effective_matrix()
        
        # Check shape
        self.assertEqual(effective_matrix.shape, (self.input_dim, self.output_dim))
        
        # Check it matches manual computation
        manual_matrix = torch.matmul(self.mps.left_tensor, self.mps.right_tensor)
        self.assertTrue(torch.allclose(effective_matrix, manual_matrix))
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        # Calculate number of parameters
        mps_params = (self.input_dim * self.bond_dim) + (self.bond_dim * self.output_dim) + self.output_dim
        full_params = self.input_dim * self.output_dim + self.output_dim
        
        # Compression ratio should be > 1 for small bond dimensions
        compression_ratio = full_params / mps_params
        self.assertGreater(compression_ratio, 1)

if __name__ == '__main__':
    unittest.main()
