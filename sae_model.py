# sae_model.py - Complete SAE with all improvements
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=12288, l1_coeff=1e-3):
        """
        Sparse Autoencoder for interpretability (Anthropic-style)
        
        Args:
            input_dim: Size of input activations (3072 for FLUX)
            hidden_dim: Number of SAE features (expansion factor * input_dim)
            l1_coeff: L1 sparsity penalty coefficient
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_coeff = l1_coeff
        
        # Pre-encoder bias (learned, centers the input distribution)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        
        # Encoder: input -> latent
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        
        # Decoder: latent -> reconstruction (no bias, we use pre_bias)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        # Kaiming initialization for encoder (good for ReLU)
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        # Initialize decoder as normalized transpose of encoder
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            self.normalize_decoder()
    
    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder columns to unit norm - call after each optimizer step"""
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x):
        """Encode input to sparse latent representation"""
        x_centered = x - self.pre_bias
        latent = self.encoder(x_centered)
        latent = torch.relu(latent)
        return latent
    
    def decode(self, latent):
        """Decode latent back to input space"""
        return self.decoder(latent) + self.pre_bias
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def loss(self, x):
        """Compute SAE loss: reconstruction + L1 sparsity"""
        reconstruction, latent = self.forward(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # L1 sparsity penalty on latent activations
        l1_loss = latent.abs().mean()
        
        # Total loss
        total_loss = recon_loss + self.l1_coeff * l1_loss
        
        # Compute additional metrics for logging
        with torch.no_grad():
            l0_sparsity = (latent > 0).float().sum(dim=1).mean()  # Avg active features
            frac_active = (latent > 0).float().mean()  # Fraction of all activations
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'l1_loss': l1_loss,
            'l0_sparsity': l0_sparsity,
            'frac_active': frac_active,
            'latent': latent
        }
    
    @torch.no_grad()
    def get_feature_density(self, dataloader, device='cuda'):
        """Compute how often each feature activates across the dataset"""
        feature_counts = torch.zeros(self.hidden_dim, device=device)
        total_samples = 0
        
        self.eval()
        for batch in dataloader:
            batch = batch.to(device)
            latent = self.encode(batch)
            feature_counts += (latent > 0).float().sum(dim=0)
            total_samples += batch.shape[0]
        self.train()
        
        return feature_counts / total_samples
    
    @torch.no_grad()
    def get_dead_features(self, dataloader, device='cuda', threshold=1e-4):
        """Find features that rarely/never activate"""
        density = self.get_feature_density(dataloader, device)
        dead_mask = density < threshold
        return dead_mask, density
    
    @torch.no_grad()
    def resample_dead_features(self, dead_mask, dataloader, device='cuda', 
                                num_samples=10000):
        """
        Reinitialize dead features using high reconstruction error examples.
        This is crucial for SAE training - dead features waste capacity.
        """
        if not dead_mask.any():
            return 0
        
        n_dead = dead_mask.sum().item()
        dead_indices = dead_mask.nonzero().squeeze(-1)
        
        # Collect samples with high reconstruction error
        self.eval()
        all_samples = []
        all_losses = []
        
        for batch in dataloader:
            batch = batch.to(device)
            recon, _ = self.forward(batch)
            losses = (recon - batch).pow(2).mean(dim=1)
            all_samples.append(batch)
            all_losses.append(losses)
            
            if sum(s.shape[0] for s in all_samples) >= num_samples:
                break
        
        all_samples = torch.cat(all_samples)[:num_samples]
        all_losses = torch.cat(all_losses)[:num_samples]
        
        # Sample proportionally to reconstruction loss
        probs = F.softmax(all_losses * 10, dim=0)  # Temperature scaling
        indices = torch.multinomial(probs, n_dead, replacement=True)
        replacement_vectors = all_samples[indices]
        
        # Center the replacement vectors
        replacement_vectors = replacement_vectors - self.pre_bias
        
        # Reinitialize dead encoder rows with scaled replacement vectors
        scale = 0.2  # Small scale to avoid disrupting training
        self.encoder.weight.data[dead_indices] = replacement_vectors * scale
        self.encoder.bias.data[dead_indices] = 0.0
        
        # Reinitialize corresponding decoder columns (normalized)
        self.decoder.weight.data[:, dead_indices] = F.normalize(
            replacement_vectors.T, dim=0
        )
        
        self.train()
        return n_dead
    
    @torch.no_grad()
    def compute_metrics(self, dataloader, device='cuda'):
        """Compute comprehensive evaluation metrics"""
        self.eval()
        
        total_recon_loss = 0
        total_l1 = 0
        total_l0 = 0
        total_ss = 0
        residual_ss = 0
        feature_activations = torch.zeros(self.hidden_dim, device=device)
        n_samples = 0
        
        all_means = []
        
        for batch in dataloader:
            batch = batch.to(device)
            recon, latent = self.forward(batch)
            
            # Reconstruction metrics
            batch_recon_loss = F.mse_loss(recon, batch, reduction='sum')
            total_recon_loss += batch_recon_loss.item()
            
            # R² computation
            batch_mean = batch.mean()
            all_means.append(batch_mean * batch.shape[0])
            total_ss += ((batch - batch_mean) ** 2).sum().item()
            residual_ss += ((batch - recon) ** 2).sum().item()
            
            # Sparsity metrics
            total_l1 += latent.abs().sum().item()
            total_l0 += (latent > 0).float().sum(dim=1).sum().item()
            
            # Feature activation counts
            feature_activations += (latent > 0).float().sum(dim=0)
            
            n_samples += batch.shape[0]
        
        # Compute feature utilization
        density = feature_activations / n_samples
        
        metrics = {
            'recon_loss': total_recon_loss / n_samples / self.input_dim,
            'r2_score': 1 - (residual_ss / total_ss) if total_ss > 0 else 0,
            'avg_l1': total_l1 / n_samples / self.hidden_dim,
            'avg_l0': total_l0 / n_samples,  # Avg active features per sample
            'frac_active': (total_l0 / n_samples) / self.hidden_dim,
            'dead_features': (density < 1e-4).sum().item(),
            'rare_features': ((density >= 1e-4) & (density < 0.01)).sum().item(),
            'common_features': ((density >= 0.01) & (density < 0.1)).sum().item(),
            'very_common_features': (density >= 0.1).sum().item(),
            'feature_density': density,
        }
        
        self.train()
        return metrics


class TopKSparseAutoencoder(nn.Module):
    """
    TopK SAE that guarantees exactly k features active per input.
    More predictable sparsity than L1 penalty - useful for guaranteed sparse codes.
    """
    def __init__(self, input_dim=3072, hidden_dim=12288, k=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.l1_coeff = 0  # Not used, but kept for compatibility
        
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()
            self.normalize_decoder()
    
    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    def encode(self, x):
        x_centered = x - self.pre_bias
        latent = self.encoder(x_centered)
        latent = torch.relu(latent)
        
        # Keep only top-k activations per sample
        if self.k < self.hidden_dim:
            topk_values, topk_indices = torch.topk(latent, self.k, dim=-1)
            sparse_latent = torch.zeros_like(latent)
            sparse_latent.scatter_(-1, topk_indices, topk_values)
            return sparse_latent
        return latent
    
    def decode(self, latent):
        return self.decoder(latent) + self.pre_bias
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def loss(self, x):
        reconstruction, latent = self.forward(x)
        recon_loss = F.mse_loss(reconstruction, x)
        
        with torch.no_grad():
            l0_sparsity = (latent > 0).float().sum(dim=1).mean()
            frac_active = (latent > 0).float().mean()
        
        return {
            'total_loss': recon_loss,
            'recon_loss': recon_loss,
            'l1_loss': torch.tensor(0.0),
            'l0_sparsity': l0_sparsity,
            'frac_active': frac_active,
            'latent': latent
        }
    
    # Inherit other methods from parent or reimplement
    get_feature_density = SparseAutoencoder.get_feature_density
    get_dead_features = SparseAutoencoder.get_dead_features
    resample_dead_features = SparseAutoencoder.resample_dead_features
    compute_metrics = SparseAutoencoder.compute_metrics