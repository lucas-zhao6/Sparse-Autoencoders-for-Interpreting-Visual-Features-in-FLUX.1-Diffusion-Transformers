# 05_generate_overlays.py
"""
Step 5: Generate activation heatmap overlays for selected features.

This script:
1. Loads selected features and their example images
2. Re-runs FLUX generation to capture activations
3. Computes per-token feature activation maps
4. Creates heatmap overlays showing where features activate

Output:
- overlays/feature_X/: Folder per feature with overlay images
- overlays/overlay_summary.json: Summary of all overlays
"""

import os
import gc
import json
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline

from config import get_config
from io_utils import load_manifest, write_json, ensure_dir
from sae_utils import load_sae, encode_to_feature_map
from flux_hooks import StepActivationCatcher
from heatmap_utils import token_map_to_blocky_heatmap, overlay_heatmap


def setup_pipeline(model_id: str) -> FluxPipeline:
    """Load and configure FLUX pipeline."""
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    
    if hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    
    return pipe


def main():
    cfg = get_config()
    
    # Paths
    selected_path = os.path.join(cfg.out_dir, "selected_features.json")
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    overlays_dir = os.path.join(cfg.out_dir, cfg.overlays_subdir)
    
    if not os.path.exists(selected_path):
        raise FileNotFoundError(f"Run 04_select_features.py first: {selected_path}")
    
    # Load data
    print("Loading data...")
    with open(selected_path, "r") as f:
        selected = json.load(f)
    
    _, id_to_record = load_manifest(manifest_path, cfg.out_dir)
    
    ensure_dir(overlays_dir)
    
    # Load SAE
    print(f"Loading SAE from {cfg.sae_ckpt_path}")
    sae, _ = load_sae(cfg.sae_ckpt_path, device=cfg.sae_device)
    
    # Load FLUX
    print(f"Loading FLUX from {cfg.model_id}")
    pipe = setup_pipeline(cfg.model_id)
    
    # Setup hook
    catcher = StepActivationCatcher(target_step_index=cfg.target_step_index)
    hook_handle = pipe.transformer.transformer_blocks[cfg.active_layer].ff.register_forward_hook(
        catcher.hook_fn
    )
    
    # Summary data
    summary = {
        "layer_index": cfg.active_layer,
        "target_step_number": cfg.target_step_number,
        "height": cfg.height,
        "width": cfg.width,
        "token_grid_size": cfg.image_token_grid,
        "items": [],
    }
    
    try:
        features = selected["features"]
        total_images = sum(len(f["example_image_ids"]) for f in features)
        
        pbar = tqdm(total=total_images, desc="Generating overlays")
        
        for feat in features:
            feature_id = int(feat["feature_id"])
            
            # Create feature directory
            feat_dir = os.path.join(overlays_dir, f"feature_{feature_id}")
            ensure_dir(feat_dir)
            
            for image_id in feat["example_image_ids"]:
                image_id = int(image_id)
                rec = id_to_record[image_id]
                
                prompt = rec["prompt"]
                seed = int(rec["seed"])
                base_path = rec["_image_abs_path"]
                
                # Reset catcher
                catcher.reset()
                
                # Regenerate to capture activations
                with torch.inference_mode():
                    _ = pipe(
                        prompt=prompt,
                        height=cfg.height,
                        width=cfg.width,
                        num_inference_steps=cfg.num_inference_steps,
                        guidance_scale=cfg.guidance_scale,
                        max_sequence_length=cfg.max_sequence_length,
                        generator=torch.Generator("cuda").manual_seed(seed),
                        output_type="pil",
                    )
                
                if catcher.captured is None:
                    raise RuntimeError(f"Failed to capture activation for image {image_id}")
                
                # Compute per-token feature activation
                z_f = encode_to_feature_map(
                    activation=catcher.captured,
                    sae=sae,
                    feature_id=feature_id,
                    n_image_tokens=cfg.n_image_tokens,
                    token_chunk_size=cfg.token_chunk_size,
                )
                
                # Load base image
                base_img = Image.open(base_path).convert("RGB")
                w, h = base_img.size
                
                # Create heatmap
                heatmap = token_map_to_blocky_heatmap(
                    token_values=z_f,
                    image_width=w,
                    image_height=h,
                    token_grid_size=cfg.image_token_grid,
                    clip_percentile=cfg.clip_percentile,
                )
                
                # Create overlay
                overlay_img = overlay_heatmap(
                    base_rgb=base_img,
                    heatmap=heatmap,
                    alpha_scale=cfg.alpha_scale,
                    color=(0, 0, 255),  # Blue
                )
                
                # Save overlay
                overlay_name = f"image_{image_id:06d}_feature_{feature_id}.png"
                overlay_path = os.path.join(feat_dir, overlay_name)
                overlay_img.save(overlay_path)
                
                # Save token map
                tokenmap_name = f"image_{image_id:06d}_feature_{feature_id}_tokenmap.npy"
                tokenmap_path = os.path.join(feat_dir, tokenmap_name)
                np.save(tokenmap_path, z_f)
                
                # Record in summary
                summary["items"].append({
                    "feature_id": feature_id,
                    "image_id": image_id,
                    "prompt": prompt,
                    "seed": seed,
                    "base_image_path": os.path.relpath(base_path, cfg.out_dir),
                    "overlay_path": os.path.relpath(overlay_path, cfg.out_dir),
                    "tokenmap_path": os.path.relpath(tokenmap_path, cfg.out_dir),
                    "activation_max": float(z_f.max()),
                    "activation_mean": float(z_f.mean()),
                })
                
                # Cleanup
                catcher.captured = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.update(1)
        
        pbar.close()
        
    finally:
        hook_handle.remove()
    
    # Save summary
    summary_path = os.path.join(overlays_dir, "overlay_summary.json")
    write_json(summary_path, summary)
    
    print(f"\nDone! Overlays saved to {overlays_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
