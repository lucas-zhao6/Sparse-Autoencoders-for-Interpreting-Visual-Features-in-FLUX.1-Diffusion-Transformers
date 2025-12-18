# 04_select_features.py
"""
Step 4: Select most interpretable features using CLIP coherence.

This script:
1. For each alive feature, finds top-M images that activate it
2. Computes mean pairwise CLIP similarity (coherence) among those images
3. Ranks features by coherence
4. Selects top-N features with unique example images

The intuition: If a feature's top-activating images are semantically similar
(high CLIP coherence), the feature likely represents something interpretable.

Output:
- selected_features.json: Top features with example image IDs
"""

import os
import numpy as np
from tqdm import tqdm

from config import get_config
from io_utils import write_json, count_manifest_rows


def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    """
    Compute mean pairwise cosine similarity.
    Embeddings should already be L2-normalized.
    """
    m = embeddings.shape[0]
    if m < 2:
        return float("-inf")
    
    # Similarity matrix
    S = embeddings @ embeddings.T
    
    # Upper triangle (excluding diagonal)
    iu = np.triu_indices(m, k=1)
    return float(S[iu].mean())


def get_top_images_unique_prompts(
    scores: np.ndarray,
    prompt_idx: np.ndarray,
    m: int,
) -> list:
    """
    Get top-m images for a feature, ensuring each comes from a different prompt.
    """
    order = np.argsort(scores)[::-1]
    
    chosen = []
    used_prompts = set()
    
    for img_id in order:
        pidx = int(prompt_idx[img_id])
        if pidx in used_prompts:
            continue
        used_prompts.add(pidx)
        chosen.append(int(img_id))
        if len(chosen) >= m:
            break
    
    return chosen


def main():
    cfg = get_config()
    
    # Paths
    s_path = os.path.join(cfg.out_dir, "S_matrix.npy")
    alive_path = os.path.join(cfg.out_dir, "alive_features.npy")
    embed_path = os.path.join(cfg.out_dir, "clip_embeddings.npy")
    prompt_idx_path = os.path.join(cfg.out_dir, "clip_prompt_idx.npy")
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    
    # Load data
    print("Loading data...")
    I_valid = count_manifest_rows(manifest_path)
    
    S_full = np.load(s_path, mmap_mode="r")
    S = np.array(S_full[:I_valid, :], dtype=np.float32)
    
    alive_features = np.load(alive_path).astype(np.int64)
    embeddings = np.load(embed_path).astype(np.float32)
    prompt_idx = np.load(prompt_idx_path).astype(np.int32)
    
    print(f"S matrix: {S.shape}")
    print(f"Alive features: {len(alive_features)}")
    print(f"CLIP embeddings: {embeddings.shape}")
    
    # Verify alignment
    if embeddings.shape[0] != I_valid:
        raise RuntimeError(f"Embedding count {embeddings.shape[0]} != manifest count {I_valid}")
    
    # Parameters
    m = cfg.top_m_per_feature  # Candidates per feature
    n_feat = cfg.top_features  # Features to select
    n_ex = cfg.top_examples_per_feature  # Examples per feature
    
    # Compute coherence for each alive feature
    print(f"Computing CLIP coherence for {len(alive_features)} features...")
    
    coherence_list = []
    per_feature_topm = {}
    
    for f in tqdm(alive_features):
        f = int(f)
        
        # Get top-m images (unique prompts)
        img_ids = get_top_images_unique_prompts(S[:, f], prompt_idx, m)
        
        # Need enough images to compute coherence
        if len(img_ids) < max(6, n_ex):
            continue
        
        # Compute coherence
        E = embeddings[img_ids]
        coh = mean_pairwise_cosine(E)
        
        coherence_list.append((coh, f))
        per_feature_topm[f] = img_ids
    
    # Sort by coherence (descending)
    coherence_list.sort(reverse=True, key=lambda x: x[0])
    
    print(f"\nTop 10 features by coherence:")
    for coh, f in coherence_list[:10]:
        print(f"  Feature {f}: coherence={coh:.4f}")
    
    # Select top features with unique example images
    print(f"\nSelecting top {n_feat} features with {n_ex} unique examples each...")
    
    chosen_features = []
    used_global_images = set()
    
    for coh, f in coherence_list:
        img_ids = per_feature_topm[f]
        
        # Select examples, optionally enforcing global uniqueness
        chosen = []
        used_prompts = set()
        
        for img_id in img_ids:
            pidx = int(prompt_idx[img_id])
            
            if pidx in used_prompts:
                continue
            
            if cfg.enforce_global_unique_images and img_id in used_global_images:
                continue
            
            used_prompts.add(pidx)
            chosen.append(int(img_id))
            
            if len(chosen) >= n_ex:
                break
        
        if len(chosen) < n_ex:
            continue
        
        # Mark images as used
        if cfg.enforce_global_unique_images:
            for img_id in chosen:
                used_global_images.add(img_id)
        
        chosen_features.append({
            "feature_id": int(f),
            "clip_coherence": float(coh),
            "example_image_ids": chosen,
            "top_m_image_ids": img_ids,
        })
        
        if len(chosen_features) >= n_feat:
            break
    
    print(f"Selected {len(chosen_features)} features")
    
    # Save results
    output = {
        "out_dir": cfg.out_dir,
        "num_images": int(I_valid),
        "num_alive_features": int(len(alive_features)),
        "top_m_per_feature": m,
        "top_features": n_feat,
        "top_examples_per_feature": n_ex,
        "enforce_global_unique_images": cfg.enforce_global_unique_images,
        "features": chosen_features,
    }
    
    out_path = os.path.join(cfg.out_dir, "selected_features.json")
    write_json(out_path, output)
    
    print(f"\nSaved: {out_path}")
    print("\nSelected features:")
    for item in chosen_features:
        print(f"  Feature {item['feature_id']}: coherence={item['clip_coherence']:.4f}, "
              f"examples={item['example_image_ids']}")


if __name__ == "__main__":
    main()
