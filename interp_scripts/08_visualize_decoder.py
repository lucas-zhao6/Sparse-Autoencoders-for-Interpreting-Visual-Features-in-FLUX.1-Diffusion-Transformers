# 08_visualize_decoder.py
"""
Step 8: Visualize decoder directions for selected features.

The decoder columns ARE the learned feature directions in the original
activation space. This script visualizes them to understand what each
feature represents geometrically.

Output:
- decoder_directions.png: Visualization of decoder columns
- decoder_stats.json: Statistics about decoder directions
"""

import os
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from config import get_config
from io_utils import write_json, read_json
from sae_utils import load_sae, get_decoder_direction, get_encoder_direction


def main():
    cfg = get_config()
    
    # Paths
    selected_path = os.path.join(cfg.out_dir, "selected_features.json")
    
    if not os.path.exists(selected_path):
        raise FileNotFoundError(f"Run 04_select_features.py first")
    
    # Load selected features
    selected = read_json(selected_path)
    feature_ids = [f["feature_id"] for f in selected["features"]]
    
    # Load SAE
    print(f"Loading SAE from {cfg.sae_ckpt_path}")
    sae, sae_config = load_sae(cfg.sae_ckpt_path, device="cpu")
    
    # Get decoder directions
    print("Extracting decoder directions...")
    decoder_directions = {}
    encoder_directions = {}
    
    for fid in feature_ids:
        decoder_directions[fid] = get_decoder_direction(sae, fid)
        encoder_directions[fid] = get_encoder_direction(sae, fid)
    
    # Compute statistics
    stats = {
        "features": [],
        "pairwise_similarities": [],
    }
    
    for fid in feature_ids:
        dec = decoder_directions[fid]
        enc = encoder_directions[fid]
        
        dec_norm = np.linalg.norm(dec)
        enc_norm = np.linalg.norm(enc)
        
        # Cosine similarity between encoder and decoder directions
        if dec_norm > 0 and enc_norm > 0:
            enc_dec_cos = np.dot(enc, dec) / (enc_norm * dec_norm)
        else:
            enc_dec_cos = 0.0
        
        stats["features"].append({
            "feature_id": int(fid),
            "decoder_norm": float(dec_norm),
            "encoder_norm": float(enc_norm),
            "encoder_decoder_cosine": float(enc_dec_cos),
            "decoder_mean": float(dec.mean()),
            "decoder_std": float(dec.std()),
            "decoder_max": float(dec.max()),
            "decoder_min": float(dec.min()),
        })
    
    # Pairwise decoder similarities
    for i, fid1 in enumerate(feature_ids):
        for fid2 in feature_ids[i+1:]:
            d1 = decoder_directions[fid1]
            d2 = decoder_directions[fid2]
            
            cos_sim = 1 - cosine(d1, d2)  # cosine returns distance
            
            stats["pairwise_similarities"].append({
                "feature_1": int(fid1),
                "feature_2": int(fid2),
                "cosine_similarity": float(cos_sim),
            })
    
    # Save stats
    stats_path = os.path.join(cfg.out_dir, "decoder_stats.json")
    write_json(stats_path, stats)
    print(f"Saved: {stats_path}")
    
    # Create visualization
    n_features = len(feature_ids)
    input_dim = sae.input_dim
    
    fig, axes = plt.subplots(n_features, 2, figsize=(14, 3 * n_features))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for idx, fid in enumerate(feature_ids):
        dec = decoder_directions[fid]
        enc = encoder_directions[fid]
        
        # Decoder direction (bar plot)
        ax_dec = axes[idx, 0]
        ax_dec.bar(range(input_dim), dec, width=1.0, alpha=0.7)
        ax_dec.axhline(y=0, color='k', linewidth=0.5)
        ax_dec.set_title(f"Feature {fid} - Decoder Direction (norm={np.linalg.norm(dec):.3f})")
        ax_dec.set_xlabel("Dimension")
        ax_dec.set_ylabel("Value")
        ax_dec.set_xlim(0, input_dim)
        
        # Encoder direction (bar plot)
        ax_enc = axes[idx, 1]
        ax_enc.bar(range(input_dim), enc, width=1.0, alpha=0.7, color='orange')
        ax_enc.axhline(y=0, color='k', linewidth=0.5)
        ax_enc.set_title(f"Feature {fid} - Encoder Direction (norm={np.linalg.norm(enc):.3f})")
        ax_enc.set_xlabel("Dimension")
        ax_enc.set_ylabel("Value")
        ax_enc.set_xlim(0, input_dim)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(cfg.out_dir, "decoder_directions.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")
    
    # Create similarity matrix visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sim_matrix = np.zeros((n_features, n_features))
    for i, fid1 in enumerate(feature_ids):
        for j, fid2 in enumerate(feature_ids):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                d1 = decoder_directions[fid1]
                d2 = decoder_directions[fid2]
                sim_matrix[i, j] = 1 - cosine(d1, d2)
    
    im = ax.imshow(sim_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(feature_ids, rotation=45)
    ax.set_yticklabels(feature_ids)
    ax.set_title("Decoder Direction Cosine Similarity")
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    sim_plot_path = os.path.join(cfg.out_dir, "decoder_similarity_matrix.png")
    plt.savefig(sim_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {sim_plot_path}")
    
    # Print summary
    print("\nDecoder direction statistics:")
    for s in stats["features"]:
        print(f"  Feature {s['feature_id']}: "
              f"dec_norm={s['decoder_norm']:.3f}, "
              f"enc_dec_cos={s['encoder_decoder_cosine']:.3f}")


if __name__ == "__main__":
    main()
