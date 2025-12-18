# heatmap_utils.py
import numpy as np
from PIL import Image
from typing import Tuple


def token_map_to_heatmap(
    token_values: np.ndarray,
    image_width: int,
    image_height: int,
    token_grid_size: int,
    clip_percentile: float = 99.0,
    min_clip: float = 1e-6,
) -> np.ndarray:
    """
    Convert per-token activations to a smooth heatmap.
    
    Args:
        token_values: Shape (T,) where T = token_grid_size^2
        image_width: Output width
        image_height: Output height
        token_grid_size: Side length of token grid (e.g., 16 for 256 tokens)
        clip_percentile: Percentile for normalization
        min_clip: Minimum clipping value
    
    Returns:
        Heatmap array shape (image_height, image_width) in range [0, 1]
    """
    T = token_values.shape[0]
    expected_T = token_grid_size * token_grid_size
    
    if T != expected_T:
        raise ValueError(f"Token count {T} doesn't match grid {token_grid_size}x{token_grid_size}={expected_T}")
    
    grid = token_values.reshape(token_grid_size, token_grid_size)
    
    # Normalize
    vmax = np.percentile(grid, clip_percentile)
    vmax = max(vmax, grid.max(), min_clip)
    normalized = np.clip(grid / vmax, 0.0, 1.0)
    
    # Resize with bilinear interpolation
    hm = Image.fromarray((normalized * 255).astype(np.uint8), mode="L")
    hm = hm.resize((image_width, image_height), resample=Image.BILINEAR)
    
    return np.array(hm).astype(np.float32) / 255.0


def token_map_to_blocky_heatmap(
    token_values: np.ndarray,
    image_width: int,
    image_height: int,
    token_grid_size: int,
    clip_percentile: float = 99.0,
    min_clip: float = 1e-6,
) -> np.ndarray:
    """
    Convert per-token activations to a blocky (non-interpolated) heatmap.
    Each token becomes a solid color patch.
    
    Returns:
        Heatmap array shape (image_height, image_width) in range [0, 1]
    """
    T = token_values.shape[0]
    expected_T = token_grid_size * token_grid_size
    
    if T != expected_T:
        raise ValueError(f"Token count {T} doesn't match grid {token_grid_size}x{token_grid_size}={expected_T}")
    
    grid = token_values.reshape(token_grid_size, token_grid_size)
    
    # Normalize
    vmax = np.percentile(grid, clip_percentile)
    vmax = max(vmax, grid.max(), min_clip)
    normalized = np.clip(grid / vmax, 0.0, 1.0)
    
    # Repeat each value to form patches
    patch_h = image_height // token_grid_size
    patch_w = image_width // token_grid_size
    
    heatmap = np.repeat(np.repeat(normalized, patch_h, axis=0), patch_w, axis=1)
    
    return heatmap.astype(np.float32)


def overlay_heatmap(
    base_rgb: Image.Image,
    heatmap: np.ndarray,
    alpha_scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 0, 255),  # Blue
) -> Image.Image:
    """
    Overlay a colored heatmap on a base image.
    
    Args:
        base_rgb: Base PIL image
        heatmap: Float array (H, W) in range [0, 1]
        alpha_scale: Maximum overlay opacity
        color: RGB tuple for overlay color
    
    Returns:
        Composited PIL image
    """
    base = base_rgb.convert("RGBA")
    w, h = base.size
    
    if heatmap.shape != (h, w):
        raise ValueError(f"Heatmap shape {heatmap.shape} doesn't match image {(h, w)}")
    
    # Create colored overlay with variable alpha
    alpha = np.clip(heatmap * alpha_scale, 0.0, 1.0)
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    
    overlay_array = np.zeros((h, w, 4), dtype=np.uint8)
    overlay_array[:, :, 0] = color[0]
    overlay_array[:, :, 1] = color[1]
    overlay_array[:, :, 2] = color[2]
    overlay_array[:, :, 3] = alpha_uint8
    
    overlay = Image.fromarray(overlay_array, mode="RGBA")
    composited = Image.alpha_composite(base, overlay)
    
    return composited.convert("RGB")


def create_side_by_side(
    original: Image.Image,
    overlay: Image.Image,
    gap: int = 10,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Create side-by-side comparison image."""
    w, h = original.size
    
    combined = Image.new("RGB", (2 * w + gap, h), background)
    combined.paste(original, (0, 0))
    combined.paste(overlay, (w + gap, 0))
    
    return combined


def create_feature_grid(
    images: list,
    ncols: int = 4,
    padding: int = 5,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL images (should all be same size)
        ncols: Number of columns
        padding: Padding between images
        background: Background color
    
    Returns:
        Grid image
    """
    if not images:
        raise ValueError("No images provided")
    
    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols
    
    grid_w = ncols * w + (ncols - 1) * padding
    grid_h = nrows * h + (nrows - 1) * padding
    
    grid = Image.new("RGB", (grid_w, grid_h), background)
    
    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        x = col * (w + padding)
        y = row * (h + padding)
        grid.paste(img.convert("RGB"), (x, y))
    
    return grid
