# sae_utils.py
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

from sae_model import SparseAutoencoder, TopKSparseAutoencoder


def load_sae(ckpt_path: str, device: str = "cuda") -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load SAE checkpoint with automatic TopK vs L1 detection.
    
    Returns:
        sae: The loaded model
        config: The config dict from checkpoint
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    
    input_dim = int(config.get("input_dim", 3072))
    hidden_dim = int(config.get("hidden_dim", 12288))
    use_topk = config.get("use_topk", False)
    topk_k = int(config.get("topk_k", 64))
    l1_coeff = float(config.get("l1_coeff", 0.0))
    
    if use_topk:
        print(f"Loading TopK SAE (k={topk_k}, hidden_dim={hidden_dim})")
        sae = TopKSparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, k=topk_k)
    else:
        print(f"Loading L1 SAE (l1_coeff={l1_coeff}, hidden_dim={hidden_dim})")
        sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, l1_coeff=l1_coeff)
    
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    
    if device == "cuda" and torch.cuda.is_available():
        sae = sae.to("cuda")
    else:
        sae = sae.to("cpu")
    
    return sae, config


def extract_image_tokens(
    activation: torch.Tensor,
    n_image_tokens: int,
    image_tokens_first: bool = True
) -> torch.Tensor:
    """
    Extract only image tokens from FLUX activation tensor.
    
    FLUX concatenates image tokens and text tokens. For interpretability,
    we typically want only the image tokens.
    
    Args:
        activation: Shape (1, T, D) or (T, D) where T includes both image and text tokens
        n_image_tokens: Number of image tokens (e.g., 256 for 256x256 with 16x16 patches)
        image_tokens_first: If True, image tokens are first; if False, they're last
    
    Returns:
        Tensor of shape (n_image_tokens, D)
    """
    if activation.dim() == 3 and activation.shape[0] == 1:
        activation = activation[0]
    
    if activation.dim() != 2:
        activation = activation.reshape(-1, activation.shape[-1])
    
    T = activation.shape[0]
    
    if T == n_image_tokens:
        # Already just image tokens
        return activation
    
    if T < n_image_tokens:
        raise ValueError(f"Activation has {T} tokens, expected at least {n_image_tokens}")
    
    if image_tokens_first:
        return activation[:n_image_tokens]
    else:
        return activation[-n_image_tokens:]


@torch.no_grad()
def compute_feature_activations(
    activation: torch.Tensor,
    sae: torch.nn.Module,
    n_image_tokens: Optional[int] = None,
    token_chunk_size: int = 256,
) -> np.ndarray:
    """
    Compute max feature activation across all tokens.
    
    Args:
        activation: Raw activation tensor from FLUX
        sae: Loaded SAE model
        n_image_tokens: If provided, extract only image tokens
        token_chunk_size: Process tokens in chunks to save memory
    
    Returns:
        s_row: Shape (hidden_dim,) - max activation per feature
    """
    device = next(sae.parameters()).device
    hidden_dim = sae.hidden_dim
    
    # Extract image tokens if specified
    if n_image_tokens is not None:
        x = extract_image_tokens(activation, n_image_tokens)
    else:
        if activation.dim() == 3 and activation.shape[0] == 1:
            x = activation[0]
        else:
            x = activation.reshape(-1, activation.shape[-1])
    
    x = x.float()
    T = x.shape[0]
    
    s_max = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
    
    for start in range(0, T, token_chunk_size):
        end = min(start + token_chunk_size, T)
        chunk = x[start:end].to(device, non_blocking=True)
        
        z = sae.encode(chunk)
        chunk_max = z.max(dim=0).values
        s_max = torch.maximum(s_max, chunk_max)
    
    return s_max.cpu().numpy().astype(np.float32)


@torch.no_grad()
def compute_feature_stats(
    activation: torch.Tensor,
    sae: torch.nn.Module,
    n_image_tokens: Optional[int] = None,
    token_chunk_size: int = 256,
    eps: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive feature activation statistics.
    
    Returns dict with:
        - max: max activation per feature
        - mean: mean activation per feature
        - std: std of activations per feature  
        - sparsity: fraction of tokens where each feature is active
        - l0_per_token: average number of active features per token
    """
    device = next(sae.parameters()).device
    hidden_dim = sae.hidden_dim
    
    if n_image_tokens is not None:
        x = extract_image_tokens(activation, n_image_tokens)
    else:
        if activation.dim() == 3 and activation.shape[0] == 1:
            x = activation[0]
        else:
            x = activation.reshape(-1, activation.shape[-1])
    
    x = x.float()
    T = x.shape[0]
    
    # Accumulate statistics
    s_max = torch.zeros(hidden_dim, device=device)
    s_sum = torch.zeros(hidden_dim, device=device)
    s_sq_sum = torch.zeros(hidden_dim, device=device)
    s_active_count = torch.zeros(hidden_dim, device=device)
    total_l0 = 0.0
    
    for start in range(0, T, token_chunk_size):
        end = min(start + token_chunk_size, T)
        chunk = x[start:end].to(device, non_blocking=True)
        
        z = sae.encode(chunk)
        
        s_max = torch.maximum(s_max, z.max(dim=0).values)
        s_sum += z.sum(dim=0)
        s_sq_sum += (z ** 2).sum(dim=0)
        s_active_count += (z > eps).float().sum(dim=0)
        total_l0 += (z > eps).sum().item()
    
    s_mean = s_sum / T
    s_var = (s_sq_sum / T) - (s_mean ** 2)
    s_std = torch.sqrt(torch.clamp(s_var, min=0))
    s_sparsity = s_active_count / T
    
    return {
        "max": s_max.cpu().numpy().astype(np.float32),
        "mean": s_mean.cpu().numpy().astype(np.float32),
        "std": s_std.cpu().numpy().astype(np.float32),
        "sparsity": s_sparsity.cpu().numpy().astype(np.float32),
        "l0_per_token": total_l0 / T,
    }


@torch.no_grad()
def encode_to_feature_map(
    activation: torch.Tensor,
    sae: torch.nn.Module,
    feature_id: int,
    n_image_tokens: Optional[int] = None,
    token_chunk_size: int = 256,
) -> np.ndarray:
    """
    Get per-token activation for a specific feature.
    
    Returns:
        z_f: Shape (T,) - activation of feature_id at each token
    """
    device = next(sae.parameters()).device
    
    if n_image_tokens is not None:
        x = extract_image_tokens(activation, n_image_tokens)
    else:
        if activation.dim() == 3 and activation.shape[0] == 1:
            x = activation[0]
        else:
            x = activation.reshape(-1, activation.shape[-1])
    
    x = x.float()
    T = x.shape[0]
    
    out = torch.empty(T, dtype=torch.float32, device="cpu")
    
    for start in range(0, T, token_chunk_size):
        end = min(start + token_chunk_size, T)
        chunk = x[start:end].to(device, non_blocking=True)
        
        z = sae.encode(chunk)
        z_f = z[:, feature_id].cpu()
        out[start:end] = z_f
    
    return out.numpy().astype(np.float32)


def get_decoder_direction(sae: torch.nn.Module, feature_id: int) -> np.ndarray:
    """Get the decoder column (learned direction) for a feature."""
    # decoder.weight is [input_dim, hidden_dim]
    return sae.decoder.weight[:, feature_id].detach().cpu().numpy()


def get_encoder_direction(sae: torch.nn.Module, feature_id: int) -> np.ndarray:
    """Get the encoder row for a feature."""
    # encoder.weight is [hidden_dim, input_dim]
    return sae.encoder.weight[feature_id, :].detach().cpu().numpy()
