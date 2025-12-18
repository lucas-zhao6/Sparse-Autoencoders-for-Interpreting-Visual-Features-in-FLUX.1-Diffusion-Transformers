# 03_clip_embed.py
"""
Step 3: Compute CLIP embeddings for all generated images.

This script:
1. Loads all images from the manifest
2. Computes CLIP image embeddings
3. Saves embeddings for feature selection

Output:
- clip_embeddings.npy: Shape (num_images, embed_dim)
- clip_meta.json: Metadata about CLIP model used
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import get_config
from io_utils import load_manifest, write_json


def main():
    cfg = get_config()
    
    # Paths
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    embed_path = os.path.join(cfg.out_dir, "clip_embeddings.npy")
    meta_path = os.path.join(cfg.out_dir, "clip_meta.json")
    
    # Load manifest
    print("Loading manifest...")
    records, _ = load_manifest(manifest_path, cfg.out_dir)
    
    image_ids = np.array([int(r["image_id"]) for r in records], dtype=np.int32)
    prompt_idx = np.array([int(r.get("prompt_idx", -1)) for r in records], dtype=np.int32)
    seeds = np.array([int(r.get("seed", -1)) for r in records], dtype=np.int32)
    image_paths = [r["_image_abs_path"] for r in records]
    
    num_images = len(image_paths)
    print(f"Found {num_images} images")
    
    # Setup CLIP
    device = cfg.clip_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    print(f"Loading CLIP model: {cfg.clip_model_id}")
    model = CLIPModel.from_pretrained(cfg.clip_model_id)
    processor = CLIPProcessor.from_pretrained(cfg.clip_model_id)
    model.eval()
    model.to(device)
    
    # Compute embeddings in batches
    batch_size = cfg.clip_batch_size
    all_embeds = []
    
    print("Computing CLIP embeddings...")
    with torch.inference_mode():
        for start in tqdm(range(0, num_images, batch_size)):
            end = min(start + batch_size, num_images)
            
            # Load images
            imgs = []
            for p in image_paths[start:end]:
                imgs.append(Image.open(p).convert("RGB"))
            
            # Process and embed
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
            
            all_embeds.append(feats.cpu().to(torch.float32).numpy())
    
    # Concatenate
    embeddings = np.concatenate(all_embeds, axis=0).astype(np.float32)
    
    # Save
    np.save(embed_path, embeddings)
    
    # Also save auxiliary arrays for convenience
    np.save(os.path.join(cfg.out_dir, "clip_image_ids.npy"), image_ids)
    np.save(os.path.join(cfg.out_dir, "clip_prompt_idx.npy"), prompt_idx)
    np.save(os.path.join(cfg.out_dir, "clip_seeds.npy"), seeds)
    
    meta = {
        "clip_model_id": cfg.clip_model_id,
        "num_images": int(embeddings.shape[0]),
        "embed_dim": int(embeddings.shape[1]),
        "device_used": device,
        "normalized": True,
    }
    write_json(meta_path, meta)
    
    print(f"\nSaved embeddings: {embed_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
