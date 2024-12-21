import torch
import torch.nn as nn
import numpy as np

class MPS(nn.Module):
    def __init__(self, input_dim, output_dim, bond_dimension, device='cuda'):
        """
        Initialize Matrix Product State for tensor decomposition.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            bond_dimension (int): Maximum bond dimension
            device (str): Device to store tensors on ('cuda' or 'cpu')
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dimension = bond_dimension
        self.device = device
        
        # Initialize core tensors
        self.left_tensor = nn.Parameter(
            torch.randn(input_dim, bond_dimension, device=device) / np.sqrt(bond_dimension)
        )
        self.right_tensor = nn.Parameter(
            torch.randn(bond_dimension, output_dim, device=device) / np.sqrt(bond_dimension)
        )
        
        # Initialize bias
        self.bias = nn.Parameter(torch.zeros(output_dim, device=device))
    
    def forward(self, x):
        """
        Forward pass through the MPS.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Contract input with left tensor
        h = torch.matmul(x, self.left_tensor)
        # Contract with right tensor
        out = torch.matmul(h, self.right_tensor)
        # Add bias
        return out + self.bias
    
    def get_effective_matrix(self):
        """
        Get the effective weight matrix from the MPS.
        
        Returns:
            torch.Tensor: Effective weight matrix
        """
        return torch.matmul(self.left_tensor, self.right_tensor)