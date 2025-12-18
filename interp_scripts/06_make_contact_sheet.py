# 06_make_contact_sheet.py
"""
Step 6: Create a contact sheet showing all selected features.

This script:
1. Loads selected features and their overlay images
2. Creates a grid visualization (features as rows, examples as columns)
3. Adds labels for feature IDs and coherence scores

Output:
- contact_sheet.png: Grid visualization of all features
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config


def find_overlay_path(overlays_dir: str, feature_id: int, image_id: int) -> str:
    """Find overlay image path for a given feature and image."""
    feat_dir = os.path.join(overlays_dir, f"feature_{feature_id}")
    
    # Standard name
    overlay_path = os.path.join(feat_dir, f"image_{image_id:06d}_feature_{feature_id}.png")
    
    if os.path.exists(overlay_path):
        return overlay_path
    
    raise FileNotFoundError(f"Overlay not found: {overlay_path}")


def main():
    cfg = get_config()
    
    # Paths
    selected_path = os.path.join(cfg.out_dir, "selected_features.json")
    overlays_dir = os.path.join(cfg.out_dir, cfg.overlays_subdir)
    
    if not os.path.exists(selected_path):
        raise FileNotFoundError(f"Run 04_select_features.py first")
    
    # Load selected features
    with open(selected_path, "r") as f:
        selected = json.load(f)
    
    features = selected["features"]
    n_rows = len(features)
    n_cols = cfg.top_examples_per_feature
    
    print(f"Creating {n_rows}x{n_cols} contact sheet...")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for row_idx, feat in enumerate(features):
        feature_id = int(feat["feature_id"])
        coherence = feat["clip_coherence"]
        image_ids = feat["example_image_ids"]
        
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(image_ids):
                image_id = int(image_ids[col_idx])
                
                try:
                    overlay_path = find_overlay_path(overlays_dir, feature_id, image_id)
                    img = Image.open(overlay_path).convert("RGB")
                    ax.imshow(img)
                except FileNotFoundError as e:
                    ax.text(0.5, 0.5, "Not found", ha="center", va="center", transform=ax.transAxes)
                
                # Title
                if col_idx == 0:
                    ax.set_title(f"Feature {feature_id}\ncoh={coherence:.3f}", fontsize=10, fontweight="bold")
                else:
                    ax.set_title(f"img {image_id}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            
            ax.axis("off")
    
    plt.tight_layout()
    
    # Save
    out_path = os.path.join(cfg.out_dir, "contact_sheet.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {out_path}")
    
    # Also create a version with original images for comparison
    create_original_comparison(cfg, features, overlays_dir)


def create_original_comparison(cfg, features, overlays_dir):
    """Create side-by-side comparison of original and overlay images."""
    from io_utils import load_manifest
    
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    _, id_to_record = load_manifest(manifest_path, cfg.out_dir)
    
    n_rows = len(features)
    n_cols = cfg.top_examples_per_feature * 2  # Original + Overlay
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 3 * n_rows))
    
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for row_idx, feat in enumerate(features):
        feature_id = int(feat["feature_id"])
        coherence = feat["clip_coherence"]
        image_ids = feat["example_image_ids"]
        
        for ex_idx in range(cfg.top_examples_per_feature):
            orig_col = ex_idx * 2
            over_col = ex_idx * 2 + 1
            
            ax_orig = axes[row_idx, orig_col]
            ax_over = axes[row_idx, over_col]
            
            if ex_idx < len(image_ids):
                image_id = int(image_ids[ex_idx])
                rec = id_to_record.get(image_id)
                
                if rec:
                    # Original
                    try:
                        orig_img = Image.open(rec["_image_abs_path"]).convert("RGB")
                        ax_orig.imshow(orig_img)
                    except:
                        ax_orig.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax_orig.transAxes)
                    
                    # Overlay
                    try:
                        overlay_path = find_overlay_path(overlays_dir, feature_id, image_id)
                        over_img = Image.open(overlay_path).convert("RGB")
                        ax_over.imshow(over_img)
                    except:
                        ax_over.text(0.5, 0.5, "Error", ha="center", va="center", transform=ax_over.transAxes)
                
                # Labels
                if ex_idx == 0:
                    ax_orig.set_title(f"Feature {feature_id}\n(coh={coherence:.3f})", fontsize=9)
                else:
                    ax_orig.set_title(f"Original", fontsize=8)
                ax_over.set_title(f"Overlay", fontsize=8)
            
            ax_orig.axis("off")
            ax_over.axis("off")
    
    plt.tight_layout()
    
    out_path = os.path.join(cfg.out_dir, "contact_sheet_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
