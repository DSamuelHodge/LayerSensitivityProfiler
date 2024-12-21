import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mps import MPS
from tdvp import TDVP

class LayerSensitivityProfiler:
    def __init__(self, model, bond_dimensions=[2, 4, 8, 16], device='cuda'):
        """
        Initialize the Layer Sensitivity Profiler.
        
        Args:
            model (nn.Module): The neural network model to analyze
            bond_dimensions (list): List of bond dimensions to test for compression
            device (str): Device to run computations on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.bond_dimensions = bond_dimensions
        self.device = device
        self.sensitivity_scores = {}
        self.layer_metrics = defaultdict(dict)
        
        # Map layer types to their compression handlers
        self.compression_handlers = {
            nn.Linear: self._compress_linear_layer,
            nn.MultiheadAttention: self._compress_attention_layer
        }
    
    def profile_layer(self, layer, input_tensor, bond_dim):
        """
        Profile a single layer's sensitivity to compression.
        
        Args:
            layer (nn.Module): Layer to profile
            input_tensor (torch.Tensor): Input tensor for the layer
            bond_dim (int): Bond dimension to test
            
        Returns:
            float: Sensitivity score for this layer and bond dimension
        """
        # Get original output
        if isinstance(layer, nn.MultiheadAttention):
            # Reshape input tensor for attention layer
            batch_size, seq_len = input_tensor.shape[0], 4
            hidden_dim = layer.embed_dim
            attention_input = torch.randn(batch_size, seq_len, hidden_dim, device=self.device)
            original_output, _ = layer(attention_input, attention_input, attention_input)
            
            # Compress layer and get output
            compressed_layer = self.compression_handlers[type(layer)](layer, bond_dim)
            compressed_output, _ = compressed_layer(attention_input, attention_input, attention_input)
        else:
            original_output = layer(input_tensor)
            
            # Compress layer and get output
            compressed_layer = self.compression_handlers[type(layer)](layer, bond_dim)
            compressed_output = compressed_layer(input_tensor)
        
        # Compute sensitivity
        return self._compute_sensitivity(original_output, compressed_output)
    
    def _compress_linear_layer(self, layer, bond_dim):
        """
        Compress a linear layer using MPS.
        
        Args:
            layer (nn.Linear): Linear layer to compress
            bond_dim (int): Target bond dimension
            
        Returns:
            nn.Module: Compressed layer
        """
        # Create MPS with correct dimensions
        mps = MPS(
            input_dim=layer.in_features,
            output_dim=layer.out_features,
            bond_dimension=bond_dim,
            device=self.device
        )
        
        # Initialize with original weights
        with torch.no_grad():
            # Get weight matrix
            weight = layer.weight.data.float()
            
            # Perform SVD
            U, S, V = torch.svd(weight)
            
            # Truncate to bond dimension
            bond_dim = min(bond_dim, len(S))
            U = U[:, :bond_dim]
            S = S[:bond_dim]
            V = V[:, :bond_dim]
            
            # Create core and transform matrices
            sqrt_S = torch.sqrt(S)
            mps.left_tensor.data = V[:, :bond_dim] * sqrt_S.unsqueeze(0)  # shape: (in_features, bond_dim)
            mps.right_tensor.data = U[:, :bond_dim] * sqrt_S.unsqueeze(0)  # shape: (out_features, bond_dim)
            mps.right_tensor.data = mps.right_tensor.data.t()  # shape: (bond_dim, out_features)
            
            # Copy bias if present
            if layer.bias is not None:
                mps.bias.data = layer.bias.data.clone()
        
        return mps
    
    def _compress_attention_layer(self, layer, bond_dim):
        """
        Compress a multi-head attention layer using TDVP.
        
        Args:
            layer (nn.MultiheadAttention): Attention layer to compress
            bond_dim (int): Target bond dimension
            
        Returns:
            nn.Module: Compressed layer
        """
        embed_dim = layer.embed_dim
        num_heads = layer.num_heads
        head_dim = embed_dim // num_heads
        
        # Create separate TDVP modules for Q, K, V projections
        q_proj = TDVP(embed_dim, embed_dim, bond_dim, device=self.device)
        k_proj = TDVP(embed_dim, embed_dim, bond_dim, device=self.device)
        v_proj = TDVP(embed_dim, embed_dim, bond_dim, device=self.device)
        
        # Initialize with original weights
        with torch.no_grad():
            # Handle both separate and combined weight matrices
            if hasattr(layer, 'in_proj_weight'):
                # Combined weights case
                qkv_weights = layer.in_proj_weight.chunk(3, dim=0)
                q_proj.core_matrix.data, q_proj.transform_matrix.data = self._init_from_matrix(
                    qkv_weights[0].float(), bond_dim
                )
                k_proj.core_matrix.data, k_proj.transform_matrix.data = self._init_from_matrix(
                    qkv_weights[1].float(), bond_dim
                )
                v_proj.core_matrix.data, v_proj.transform_matrix.data = self._init_from_matrix(
                    qkv_weights[2].float(), bond_dim
                )
            elif hasattr(layer, 'q_proj_weight') and layer.q_proj_weight is not None:
                # Separate weights case
                q_proj.core_matrix.data, q_proj.transform_matrix.data = self._init_from_matrix(
                    layer.q_proj_weight.float(), bond_dim
                )
                k_proj.core_matrix.data, k_proj.transform_matrix.data = self._init_from_matrix(
                    layer.k_proj_weight.float(), bond_dim
                )
                v_proj.core_matrix.data, v_proj.transform_matrix.data = self._init_from_matrix(
                    layer.v_proj_weight.float(), bond_dim
                )
        
        # Create a wrapper module
        class CompressedAttention(nn.Module):
            def __init__(self, q_proj, k_proj, v_proj, num_heads, embed_dim, device):
                super().__init__()
                self.q_proj = q_proj
                self.k_proj = k_proj
                self.v_proj = v_proj
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.scaling = float(self.head_dim) ** -0.5
                self.device = device
            
            def forward(self, query, key=None, value=None, key_padding_mask=None, need_weights=True, attn_mask=None):
                if key is None:
                    key = query
                if value is None:
                    value = key
                
                batch_size = query.size(0)
                seq_len = query.size(1)
                
                # Apply projections
                q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
                k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
                v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
                
                # Transpose for attention calculation
                q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
                k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
                v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
                
                # Calculate attention
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                
                if attn_mask is not None:
                    scores = scores.masked_fill(attn_mask == 0, float('-inf'))
                
                if key_padding_mask is not None:
                    scores = scores.masked_fill(
                        key_padding_mask.unsqueeze(1).unsqueeze(2),
                        float('-inf'),
                    )
                
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v)
                
                # Reshape output
                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
                
                return out, attn if need_weights else None
        
        return CompressedAttention(q_proj, k_proj, v_proj, num_heads, embed_dim, self.device)
    
    def _init_from_matrix(self, matrix, bond_dim):
        """
        Initialize core and transform matrices from a weight matrix using SVD.
        
        Args:
            matrix (torch.Tensor): Input weight matrix
            bond_dim (int): Target bond dimension
            
        Returns:
            tuple: (core_matrix, transform_matrix)
        """
        U, S, V = torch.svd(matrix)
        
        # Truncate to bond dimension
        bond_dim = min(bond_dim, len(S))
        U = U[:, :bond_dim]
        S = S[:bond_dim]
        V = V[:, :bond_dim]
        
        # Create core and transform matrices
        sqrt_S = torch.sqrt(S)
        core_matrix = torch.matmul(U[:, :bond_dim], torch.diag(sqrt_S))  # shape: (out_features, bond_dim)
        transform_matrix = torch.matmul(V[:, :bond_dim], torch.diag(sqrt_S)).t()  # shape: (bond_dim, in_features)
        
        return core_matrix.contiguous(), transform_matrix.contiguous()
    
    def _compute_sensitivity(self, original_output, compressed_output):
        """
        Compute sensitivity score between original and compressed outputs.
        
        Args:
            original_output (torch.Tensor): Output from original layer
            compressed_output (torch.Tensor): Output from compressed layer
        
        Returns:
            float: Sensitivity score
        """
        # Compute relative error
        error = torch.norm(original_output - compressed_output) / torch.norm(original_output)
        return error.item()
    
    def profile_model(self, sample_input):
        """
        Profile the entire model's layer sensitivities.
        
        Args:
            sample_input (torch.Tensor): Sample input tensor for the model
            
        Returns:
            dict: Dictionary containing sensitivity scores and compression ratios for each layer
        """
        results = {}
        
        # Get all layers to profile
        layers_to_profile = []
        for name, module in self.model.named_modules():
            if type(module) in self.compression_handlers:
                layers_to_profile.append((name, module))
        
        print("\nProfiling layers:\n")
        # Profile each layer
        for name, layer in layers_to_profile:
            print(f"Analyzing layer: {name}")
            layer_scores = []
            
            for bond_dim in self.bond_dimensions:
                print(f"  Testing bond dimension: {bond_dim}")
                
                # Profile layer
                sensitivity = self.profile_layer(layer, sample_input, bond_dim)
                compression_ratio = self._calculate_compression_ratio(layer, bond_dim)
                
                print(f"    Sensitivity: {sensitivity:.4f}")
                print(f"    Compression ratio: {compression_ratio:.2f}x")
                
                # Store results
                self.layer_metrics[name][bond_dim] = {
                    'sensitivity': sensitivity,
                    'compression_ratio': compression_ratio
                }
                layer_scores.append((bond_dim, sensitivity))
            
            # Store sensitivity scores for plotting
            self.sensitivity_scores[name] = layer_scores
            print()
        
        return self.layer_metrics
    
    def _calculate_compression_ratio(self, layer, bond_dim):
        """
        Calculate compression ratio for a layer.
        
        Args:
            layer (nn.Module): Layer to analyze
            bond_dim (int): Bond dimension used for compression
        
        Returns:
            float: Compression ratio
        """
        if isinstance(layer, nn.Linear):
            original_params = layer.in_features * layer.out_features
            compressed_params = bond_dim * (layer.in_features + layer.out_features)
        else:  # MultiheadAttention
            embed_dim = layer.embed_dim
            original_params = 3 * embed_dim * embed_dim  # Q, K, V projections
            compressed_params = 3 * bond_dim * (2 * embed_dim)  # Compressed Q, K, V
        
        return original_params / compressed_params
    
    def plot_sensitivity_curves(self, save_path=None):
        """
        Plot sensitivity curves for all profiled layers.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        plt.clf()  # Clear any existing plots
        plt.figure(figsize=(12, 8))
        
        for name, scores in self.sensitivity_scores.items():
            bond_dims, sensitivities = zip(*scores)
            plt.plot(bond_dims, sensitivities, 'o-', label=name, linewidth=2, markersize=8)
        
        plt.xlabel('Bond Dimension', fontsize=12)
        plt.ylabel('Sensitivity Score', fontsize=12)
        plt.title('Layer Sensitivity vs Bond Dimension', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True)
        plt.xscale('log', base=2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nSensitivity curves have been saved to '{save_path}'")
        
        plt.close()
    
    def get_optimal_compression(self, sensitivity_threshold=0.1):
        """
        Get optimal compression settings for each layer.
        
        Args:
            sensitivity_threshold (float): Maximum acceptable sensitivity score
        
        Returns:
            dict: Optimal bond dimensions for each layer
        """
        optimal_settings = {}
        for name, metrics in self.layer_metrics.items():
            # Find highest compression (lowest bond dim) that meets sensitivity threshold
            valid_bonds = [
                bond_dim for bond_dim, data in metrics.items()
                if data['sensitivity'] <= sensitivity_threshold
            ]
            if valid_bonds:
                optimal_settings[name] = {
                    'bond_dimension': min(valid_bonds),
                    'compression_ratio': metrics[min(valid_bonds)]['compression_ratio'],
                    'sensitivity': metrics[min(valid_bonds)]['sensitivity']
                }
        
        return optimal_settings
