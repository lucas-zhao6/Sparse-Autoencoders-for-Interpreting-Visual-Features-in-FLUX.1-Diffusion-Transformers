# 02_analyze_features.py
"""
Step 2: Analyze the S matrix to find alive features and compute statistics.

This script:
1. Loads the S matrix from step 1
2. Computes which features are "alive" (activate on at least one image)
3. Computes per-feature statistics
4. Identifies top-k images per feature

Output:
- alive_features.npy: Indices of alive features
- feature_stats.json: Statistics for each feature
- topk_per_feature.npz: Top-k image indices per feature
"""

import os
import numpy as np
from tqdm import tqdm

from config import get_config
from io_utils import write_json, count_manifest_rows


def main():
    cfg = get_config()
    
    # Paths
    s_path = os.path.join(cfg.out_dir, "S_matrix.npy")
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    
    # Load S matrix
    print(f"Loading S matrix from {s_path}")
    S_full = np.load(s_path, mmap_mode="r")
    
    # Only use rows that were actually completed
    I_valid = count_manifest_rows(manifest_path)
    S = np.array(S_full[:I_valid, :], dtype=np.float32)
    
    num_images, hidden_dim = S.shape
    print(f"S matrix shape: {num_images} images x {hidden_dim} features")
    
    eps = cfg.eps_alive
    
    # Compute per-feature statistics
    print("Computing feature statistics...")
    
    # Alive features (activate above eps on at least one image)
    max_per_feature = S.max(axis=0)
    alive_mask = max_per_feature > eps
    alive_indices = np.nonzero(alive_mask)[0].astype(np.int32)
    
    # Activation frequency (fraction of images where feature is active)
    activation_freq = (S > eps).mean(axis=0).astype(np.float32)
    
    # Mean and std of activations (when active)
    mean_activation = np.zeros(hidden_dim, dtype=np.float32)
    std_activation = np.zeros(hidden_dim, dtype=np.float32)
    
    for f in tqdm(alive_indices, desc="Computing stats"):
        vals = S[:, f]
        active_vals = vals[vals > eps]
        if len(active_vals) > 0:
            mean_activation[f] = active_vals.mean()
            std_activation[f] = active_vals.std() if len(active_vals) > 1 else 0.0
    
    # Save alive features
    alive_path = os.path.join(cfg.out_dir, "alive_features.npy")
    np.save(alive_path, alive_indices)
    print(f"Alive features: {len(alive_indices)} / {hidden_dim} ({100*len(alive_indices)/hidden_dim:.1f}%)")
    
    # Save activation frequency
    freq_path = os.path.join(cfg.out_dir, "activation_freq.npy")
    np.save(freq_path, activation_freq)
    
    # Compute top-k images per feature
    print(f"Computing top-{cfg.top_k_per_image} images per feature...")
    k = min(cfg.top_k_per_image, num_images)
    
    # Use argpartition for efficiency
    top_idx = np.zeros((k, hidden_dim), dtype=np.int32)
    top_scores = np.zeros((k, hidden_dim), dtype=np.float32)
    
    for f in tqdm(range(hidden_dim), desc="Top-k per feature"):
        col = S[:, f]
        if k >= num_images:
            order = np.argsort(col)[::-1]
            top_idx[:, f] = order[:k]
            top_scores[:, f] = col[order[:k]]
        else:
            part_idx = np.argpartition(col, -k)[-k:]
            part_scores = col[part_idx]
            order = np.argsort(part_scores)[::-1]
            top_idx[:, f] = part_idx[order]
            top_scores[:, f] = part_scores[order]
    
    topk_path = os.path.join(cfg.out_dir, "topk_per_feature.npz")
    np.savez_compressed(
        topk_path,
        top_idx=top_idx,
        top_scores=top_scores,
        k=k,
        eps=eps,
    )
    
    # Feature categorization
    dead_count = int((max_per_feature <= eps).sum())
    rare_count = int(((activation_freq > 0) & (activation_freq < 0.01)).sum())
    common_count = int(((activation_freq >= 0.01) & (activation_freq < 0.1)).sum())
    very_common_count = int((activation_freq >= 0.1).sum())
    
    # Summary statistics
    summary = {
        "num_images": int(num_images),
        "hidden_dim": int(hidden_dim),
        "eps_alive": float(eps),
        "num_alive_features": int(len(alive_indices)),
        "num_dead_features": dead_count,
        "num_rare_features": rare_count,
        "num_common_features": common_count,
        "num_very_common_features": very_common_count,
        "top_k": int(k),
        "files": {
            "alive_features": "alive_features.npy",
            "activation_freq": "activation_freq.npy",
            "topk_per_feature": "topk_per_feature.npz",
        },
    }
    
    summary_path = os.path.join(cfg.out_dir, "feature_analysis_summary.json")
    write_json(summary_path, summary)
    
    print(f"\nFeature breakdown:")
    print(f"  Dead (<{eps}): {dead_count} ({100*dead_count/hidden_dim:.1f}%)")
    print(f"  Rare (0-1%): {rare_count}")
    print(f"  Common (1-10%): {common_count}")
    print(f"  Very common (>10%): {very_common_count}")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
