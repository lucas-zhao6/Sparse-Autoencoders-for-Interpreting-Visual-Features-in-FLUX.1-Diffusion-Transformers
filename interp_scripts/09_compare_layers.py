# 09_compare_layers.py
"""
Step 9: Compare features between Layer 5 and Layer 15 SAEs.

This script:
1. Runs activation collection for BOTH layers on the same images
2. Compares which features activate for the same images
3. Analyzes representation differences between layers

Run this AFTER running the full pipeline for both layers separately.

Output:
- layer_comparison/: Comparison visualizations
- layer_comparison/comparison_summary.json: Analysis results
"""

import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config, Config
from io_utils import ensure_dir, write_json, read_prompts, load_manifest
from sae_utils import load_sae, compute_feature_activations
from flux_hooks import StepActivationCatcher


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


def collect_both_layers(cfg: Config, prompts: list, pipe: FluxPipeline, sae5, sae15):
    """
    Generate images and collect activations from both layers simultaneously.
    """
    hidden_dim = sae5.hidden_dim
    num_prompts = len(prompts)
    num_images = num_prompts * cfg.reps_per_prompt
    
    # Initialize S matrices
    S5 = np.zeros((num_images, hidden_dim), dtype=np.float32)
    S15 = np.zeros((num_images, hidden_dim), dtype=np.float32)
    
    # Setup hooks for both layers
    catcher5 = StepActivationCatcher(target_step_index=cfg.target_step_index)
    catcher15 = StepActivationCatcher(target_step_index=cfg.target_step_index)
    
    hook5 = pipe.transformer.transformer_blocks[5].ff.register_forward_hook(catcher5.hook_fn)
    hook15 = pipe.transformer.transformer_blocks[15].ff.register_forward_hook(catcher15.hook_fn)
    
    image_id = 0
    metadata = []
    
    try:
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
            for rep_idx in range(cfg.reps_per_prompt):
                seed = cfg.base_seed + 2 * prompt_idx + rep_idx
                
                catcher5.reset()
                catcher15.reset()
                
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
                
                # Compute feature activations
                s5 = compute_feature_activations(
                    catcher5.captured, sae5, n_image_tokens=cfg.n_image_tokens
                )
                s15 = compute_feature_activations(
                    catcher15.captured, sae15, n_image_tokens=cfg.n_image_tokens
                )
                
                S5[image_id] = s5
                S15[image_id] = s15
                
                metadata.append({
                    "image_id": image_id,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "seed": seed,
                })
                
                image_id += 1
                
    finally:
        hook5.remove()
        hook15.remove()
    
    return S5[:image_id], S15[:image_id], metadata


def analyze_layer_differences(S5: np.ndarray, S15: np.ndarray, eps: float = 1e-4):
    """
    Analyze differences in feature activation patterns between layers.
    """
    num_images = S5.shape[0]
    hidden_dim = S5.shape[1]
    
    # Feature activation patterns
    active5 = (S5 > eps)
    active15 = (S15 > eps)
    
    # Per-image statistics
    l0_per_image_5 = active5.sum(axis=1)
    l0_per_image_15 = active15.sum(axis=1)
    
    # Per-feature statistics
    freq5 = active5.mean(axis=0)
    freq15 = active15.mean(axis=0)
    
    # Alive features
    alive5 = np.where(freq5 > 0)[0]
    alive15 = np.where(freq15 > 0)[0]
    
    # Overlap of alive features
    alive_both = set(alive5) & set(alive15)
    alive_only5 = set(alive5) - set(alive15)
    alive_only15 = set(alive15) - set(alive5)
    
    # For each image, top-k feature overlap
    k = 50
    overlap_per_image = []
    
    for i in range(num_images):
        top5 = set(np.argsort(S5[i])[-k:])
        top15 = set(np.argsort(S15[i])[-k:])
        overlap = len(top5 & top15)
        overlap_per_image.append(overlap)
    
    return {
        "num_images": int(num_images),
        "hidden_dim": int(hidden_dim),
        "layer5": {
            "mean_l0": float(l0_per_image_5.mean()),
            "std_l0": float(l0_per_image_5.std()),
            "num_alive_features": int(len(alive5)),
        },
        "layer15": {
            "mean_l0": float(l0_per_image_15.mean()),
            "std_l0": float(l0_per_image_15.std()),
            "num_alive_features": int(len(alive15)),
        },
        "comparison": {
            "alive_in_both": int(len(alive_both)),
            "alive_only_in_5": int(len(alive_only5)),
            "alive_only_in_15": int(len(alive_only15)),
            f"top{k}_overlap_mean": float(np.mean(overlap_per_image)),
            f"top{k}_overlap_std": float(np.std(overlap_per_image)),
        },
    }


def main():
    cfg = get_config()
    
    # Check if both SAE paths exist
    if not os.path.exists(cfg.layer5_sae_path):
        print(f"ERROR: Layer 5 SAE not found at {cfg.layer5_sae_path}")
        print("Please train the layer 5 SAE first or update config.py")
        return
    
    if not os.path.exists(cfg.layer15_sae_path):
        print(f"ERROR: Layer 15 SAE not found at {cfg.layer15_sae_path}")
        print("Please train the layer 15 SAE first or update config.py")
        return
    
    # Setup output directory
    compare_dir = os.path.join(cfg.out_dir, "layer_comparison")
    ensure_dir(compare_dir)
    
    # Load prompts
    prompts = read_prompts(cfg.prompts_path)[:20]  # Limit for speed
    print(f"Using {len(prompts)} prompts for comparison")
    
    # Load SAEs
    print(f"Loading Layer 5 SAE from {cfg.layer5_sae_path}")
    sae5, _ = load_sae(cfg.layer5_sae_path, device=cfg.sae_device)
    
    print(f"Loading Layer 15 SAE from {cfg.layer15_sae_path}")
    sae15, _ = load_sae(cfg.layer15_sae_path, device=cfg.sae_device)
    
    # Load FLUX
    print(f"Loading FLUX from {cfg.model_id}")
    pipe = setup_pipeline(cfg.model_id)
    
    # Collect activations from both layers
    print("Collecting activations from both layers...")
    S5, S15, metadata = collect_both_layers(cfg, prompts, pipe, sae5, sae15)
    
    # Save S matrices
    np.save(os.path.join(compare_dir, "S_layer5.npy"), S5)
    np.save(os.path.join(compare_dir, "S_layer15.npy"), S15)
    
    # Analyze differences
    print("Analyzing layer differences...")
    analysis = analyze_layer_differences(S5, S15, eps=cfg.eps_alive)
    
    # Save analysis
    analysis_path = os.path.join(compare_dir, "comparison_summary.json")
    write_json(analysis_path, analysis)
    print(f"Saved: {analysis_path}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. L0 comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    l0_5 = (S5 > cfg.eps_alive).sum(axis=1)
    l0_15 = (S15 > cfg.eps_alive).sum(axis=1)
    
    axes[0].hist(l0_5, bins=30, alpha=0.7, label="Layer 5")
    axes[0].hist(l0_15, bins=30, alpha=0.7, label="Layer 15")
    axes[0].set_xlabel("L0 (active features)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of L0 per Image")
    axes[0].legend()
    
    axes[1].scatter(l0_5, l0_15, alpha=0.5)
    axes[1].plot([l0_5.min(), l0_5.max()], [l0_5.min(), l0_5.max()], 'r--', label="y=x")
    axes[1].set_xlabel("Layer 5 L0")
    axes[1].set_ylabel("Layer 15 L0")
    axes[1].set_title("Layer 5 vs Layer 15 L0")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "l0_comparison.png"), dpi=150)
    plt.close()
    
    # 2. Feature frequency comparison
    freq5 = (S5 > cfg.eps_alive).mean(axis=0)
    freq15 = (S15 > cfg.eps_alive).mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Only show features that are alive in at least one layer
    alive_mask = (freq5 > 0) | (freq15 > 0)
    
    ax.scatter(freq5[alive_mask], freq15[alive_mask], alpha=0.3, s=5)
    ax.plot([0, 1], [0, 1], 'r--', label="y=x")
    ax.set_xlabel("Layer 5 Activation Frequency")
    ax.set_ylabel("Layer 15 Activation Frequency")
    ax.set_title("Feature Activation Frequency: Layer 5 vs Layer 15")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(compare_dir, "frequency_comparison.png"), dpi=150)
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("LAYER COMPARISON SUMMARY")
    print("="*60)
    print(f"Images analyzed: {analysis['num_images']}")
    print(f"\nLayer 5:")
    print(f"  Mean L0: {analysis['layer5']['mean_l0']:.1f} ± {analysis['layer5']['std_l0']:.1f}")
    print(f"  Alive features: {analysis['layer5']['num_alive_features']}")
    print(f"\nLayer 15:")
    print(f"  Mean L0: {analysis['layer15']['mean_l0']:.1f} ± {analysis['layer15']['std_l0']:.1f}")
    print(f"  Alive features: {analysis['layer15']['num_alive_features']}")
    print(f"\nComparison:")
    print(f"  Features alive in both: {analysis['comparison']['alive_in_both']}")
    print(f"  Features only in L5: {analysis['comparison']['alive_only_in_5']}")
    print(f"  Features only in L15: {analysis['comparison']['alive_only_in_15']}")
    print(f"  Top-50 overlap per image: {analysis['comparison']['top50_overlap_mean']:.1f} ± {analysis['comparison']['top50_overlap_std']:.1f}")
    print("="*60)
    
    print(f"\nResults saved to {compare_dir}")


if __name__ == "__main__":
    main()
