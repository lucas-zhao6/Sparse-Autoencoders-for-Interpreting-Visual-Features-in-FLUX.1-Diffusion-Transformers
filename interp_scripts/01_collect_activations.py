# 01_collect_activations.py
"""
Step 1: Generate images and collect SAE feature activations.

This script:
1. Loads FLUX and the trained SAE
2. Generates images from prompts
3. Captures activations at the target layer/step
4. Computes max feature activation per image -> S matrix

Output:
- images/ folder with generated images
- S_matrix.npy: Shape (num_images, hidden_dim)
- manifest.jsonl: Metadata for each image
"""

import os
import time
import platform
import numpy as np
import torch
from tqdm import tqdm
from numpy.lib.format import open_memmap
from diffusers import FluxPipeline

from config import get_config
from io_utils import ensure_dir, read_prompts, write_json, append_jsonl
from sae_utils import load_sae, compute_feature_activations
from flux_hooks import StepActivationCatcher


def setup_pipeline(model_id: str) -> FluxPipeline:
    """Load and configure FLUX pipeline."""
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    
    # Memory optimizations
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
    
    # Setup directories
    ensure_dir(cfg.out_dir)
    images_dir = os.path.join(cfg.out_dir, cfg.images_subdir)
    ensure_dir(images_dir)
    
    # Output paths
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    run_log_path = os.path.join(cfg.out_dir, "run_log.json")
    s_path = os.path.join(cfg.out_dir, "S_matrix.npy")
    
    # Load prompts
    prompts = read_prompts(cfg.prompts_path)
    num_prompts = len(prompts)
    num_images = num_prompts * cfg.reps_per_prompt
    
    print(f"Prompts: {num_prompts}, Reps: {cfg.reps_per_prompt}, Total images: {num_images}")
    
    # Load SAE
    print(f"Loading SAE from {cfg.sae_ckpt_path}")
    sae, sae_config = load_sae(cfg.sae_ckpt_path, device=cfg.sae_device)
    hidden_dim = sae.hidden_dim
    
    # Load FLUX
    print(f"Loading FLUX from {cfg.model_id}")
    pipe = setup_pipeline(cfg.model_id)
    
    # Setup hook
    catcher = StepActivationCatcher(target_step_index=cfg.target_step_index)
    hook_handle = pipe.transformer.transformer_blocks[cfg.active_layer].ff.register_forward_hook(
        catcher.hook_fn
    )
    
    # Create memory-mapped S matrix
    S = open_memmap(s_path, mode="w+", dtype="float32", shape=(num_images, hidden_dim))
    
    # Log run config
    run_log = {
        "out_dir": cfg.out_dir,
        "prompts_path": cfg.prompts_path,
        "num_prompts": num_prompts,
        "reps_per_prompt": cfg.reps_per_prompt,
        "num_images": num_images,
        "model_id": cfg.model_id,
        "height": cfg.height,
        "width": cfg.width,
        "num_inference_steps": cfg.num_inference_steps,
        "guidance_scale": cfg.guidance_scale,
        "max_sequence_length": cfg.max_sequence_length,
        "layer_index": cfg.active_layer,
        "target_step_number": cfg.target_step_number,
        "target_step_index": cfg.target_step_index,
        "sae_ckpt_path": cfg.sae_ckpt_path,
        "sae_hidden_dim": hidden_dim,
        "sae_config": sae_config,
        "n_image_tokens": cfg.n_image_tokens,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    write_json(run_log_path, run_log)
    
    # Generate images
    global_start = time.time()
    image_id = 0
    
    try:
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
            for rep_idx in range(cfg.reps_per_prompt):
                seed = cfg.base_seed + 2 * prompt_idx + rep_idx
                
                catcher.reset()
                
                img_name = f"img_{image_id:06d}_p{prompt_idx:04d}_r{rep_idx}_seed{seed}.png"
                img_path = os.path.join(images_dir, img_name)
                
                t0 = time.time()
                
                with torch.no_grad():
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
                
                image = out.images[0]
                image.save(img_path)
                
                # Check activation capture
                if catcher.captured is None:
                    raise RuntimeError(
                        f"Failed to capture activation at step {cfg.target_step_number}. "
                        f"Check hook placement and num_inference_steps."
                    )
                
                # Compute feature activations (image tokens only)
                s_row = compute_feature_activations(
                    activation=catcher.captured,
                    sae=sae,
                    n_image_tokens=cfg.n_image_tokens,
                    token_chunk_size=cfg.token_chunk_size,
                )
                
                S[image_id, :] = s_row
                S.flush()
                
                # Log manifest
                manifest_row = {
                    "image_id": image_id,
                    "prompt_idx": prompt_idx,
                    "rep_idx": rep_idx,
                    "seed": seed,
                    "prompt": prompt,
                    "image_path": os.path.relpath(img_path, cfg.out_dir),
                    "height": cfg.height,
                    "width": cfg.width,
                    "layer_index": cfg.active_layer,
                    "target_step_number": cfg.target_step_number,
                    "activation_shape": list(catcher.captured.shape),
                    "generation_time": time.time() - t0,
                }
                append_jsonl(manifest_path, manifest_row)
                
                image_id += 1
                
    finally:
        hook_handle.remove()
    
    # Finalize
    total_time = time.time() - global_start
    run_log["total_time_seconds"] = total_time
    run_log["images_completed"] = image_id
    write_json(run_log_path, run_log)
    
    print(f"\nDone! Generated {image_id} images in {total_time:.1f}s")
    print(f"S matrix: {s_path}")
    print(f"Images: {images_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
