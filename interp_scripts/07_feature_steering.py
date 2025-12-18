# 07_feature_steering.py
"""
Step 7: Feature steering experiments.

This script:
1. Loads a trained SAE and selected features
2. Generates images while amplifying/suppressing specific features
3. Creates comparison grids showing the effect of steering

Output:
- steering/feature_X/: Folder per feature with steered images
- steering/steering_summary.json: Summary of experiments
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
from sae_utils import load_sae
from flux_hooks import FeatureSteeringHook, StepActivationCatcher
from heatmap_utils import create_feature_grid


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


def generate_with_steering(
    pipe: FluxPipeline,
    sae: torch.nn.Module,
    prompt: str,
    seed: int,
    feature_id: int,
    scale: float,
    cfg,
) -> Image.Image:
    """Generate an image with feature steering."""
    
    # Setup steering hook
    steering_hook = FeatureSteeringHook(
        sae=sae,
        feature_id=feature_id,
        scale=scale,
        target_step_index=cfg.target_step_index,
        n_image_tokens=cfg.n_image_tokens,
    )
    
    hook_handle = pipe.transformer.transformer_blocks[cfg.active_layer].ff.register_forward_hook(
        steering_hook.hook_fn
    )
    
    try:
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                max_sequence_length=cfg.max_sequence_length,
                generator=torch.Generator("cuda").manual_seed(seed),
                output_type="pil",
            )
        return out.images[0]
    finally:
        hook_handle.remove()


def main():
    cfg = get_config()
    
    # Paths
    selected_path = os.path.join(cfg.out_dir, "selected_features.json")
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    steering_dir = os.path.join(cfg.out_dir, "steering")
    
    if not os.path.exists(selected_path):
        raise FileNotFoundError(f"Run 04_select_features.py first")
    
    # Load data
    print("Loading data...")
    with open(selected_path, "r") as f:
        selected = json.load(f)
    
    _, id_to_record = load_manifest(manifest_path, cfg.out_dir)
    
    ensure_dir(steering_dir)
    
    # Load SAE
    print(f"Loading SAE from {cfg.sae_ckpt_path}")
    sae, _ = load_sae(cfg.sae_ckpt_path, device=cfg.sae_device)
    
    # Load FLUX
    print(f"Loading FLUX from {cfg.model_id}")
    pipe = setup_pipeline(cfg.model_id)
    
    # Steering scales to test
    scales = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]  # 0=ablate, 1=baseline, >1=amplify
    
    # Summary
    summary = {
        "layer_index": cfg.active_layer,
        "target_step_number": cfg.target_step_number,
        "scales": scales,
        "experiments": [],
    }
    
    features = selected["features"][:5]  # Limit to top 5 for speed
    
    print(f"Running steering experiments on {len(features)} features...")
    
    for feat in tqdm(features, desc="Features"):
        feature_id = int(feat["feature_id"])
        image_ids = feat["example_image_ids"][:2]  # Use 2 examples per feature
        
        feat_dir = os.path.join(steering_dir, f"feature_{feature_id}")
        ensure_dir(feat_dir)
        
        for image_id in image_ids:
            image_id = int(image_id)
            rec = id_to_record[image_id]
            prompt = rec["prompt"]
            seed = int(rec["seed"])
            
            steered_images = []
            
            for scale in scales:
                img = generate_with_steering(
                    pipe=pipe,
                    sae=sae,
                    prompt=prompt,
                    seed=seed,
                    feature_id=feature_id,
                    scale=scale,
                    cfg=cfg,
                )
                
                # Save individual image
                img_name = f"image_{image_id:06d}_scale_{scale:.1f}.png"
                img_path = os.path.join(feat_dir, img_name)
                img.save(img_path)
                
                steered_images.append(img)
                
                # Cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create comparison grid
            grid = create_feature_grid(steered_images, ncols=len(scales), padding=5)
            grid_name = f"image_{image_id:06d}_steering_grid.png"
            grid_path = os.path.join(feat_dir, grid_name)
            grid.save(grid_path)
            
            summary["experiments"].append({
                "feature_id": feature_id,
                "image_id": image_id,
                "prompt": prompt,
                "seed": seed,
                "grid_path": os.path.relpath(grid_path, cfg.out_dir),
            })
    
    # Save summary
    summary_path = os.path.join(steering_dir, "steering_summary.json")
    write_json(summary_path, summary)
    
    print(f"\nDone! Steering experiments saved to {steering_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
