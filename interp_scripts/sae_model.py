# sae_model.py
import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """L1-regularized Sparse Autoencoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, l1_coeff: float = 1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.pre_bias
        latent = self.encoder(x_centered)
        latent = torch.relu(latent)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent) + self.pre_bias
    
    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon, latent
    
    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)


class TopKSparseAutoencoder(nn.Module):
    """TopK Sparse Autoencoder with guaranteed sparsity."""
    
    def __init__(self, input_dim: int, hidden_dim: int, k: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.pre_bias
        latent = self.encoder(x_centered)
        latent = torch.relu(latent)
        
        # TopK selection
        topk_values, topk_indices = torch.topk(latent, self.k, dim=-1)
        sparse_latent = torch.zeros_like(latent)
        sparse_latent.scatter_(-1, topk_indices, topk_values)
        
        return sparse_latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent) + self.pre_bias
    
    def forward(self, x: torch.Tensor):
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon, latent
    
    def normalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(self.decoder.weight.data, dim=0)
