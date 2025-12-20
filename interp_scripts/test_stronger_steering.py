# test_stronger_steering.py
"""Test stronger steering parameters."""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import FluxPipeline
import sys
sys.path.insert(0, '.')

from config import get_config
from sae_utils import load_sae
from flux_hooks import FeatureSteeringHook


def generate_steering_grid(pipe, sae, cfg, feature_id: int, prompt: str, seed: int,
                           strengths: list, step_range: tuple, label: str):
    """Generate a grid with labels."""
    
    images = []
    
    for strength in strengths:
        if strength == 0:
            # Baseline - no hook
            with torch.inference_mode():
                img = pipe(
                    prompt=prompt,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    max_sequence_length=cfg.max_sequence_length,
                    generator=torch.Generator("cuda").manual_seed(seed),
                ).images[0]
        else:
            hook = FeatureSteeringHook(
                sae=sae,
                feature_id=feature_id,
                strength=strength,
                step_range=step_range,
                n_image_tokens=cfg.n_image_tokens,
            )
            
            handle = pipe.transformer.transformer_blocks[cfg.active_layer].ff.register_forward_hook(
                hook.hook_fn
            )
            
            with torch.inference_mode():
                img = pipe(
                    prompt=prompt,
                    height=cfg.height,
                    width=cfg.width,
                    num_inference_steps=cfg.num_inference_steps,
                    guidance_scale=cfg.guidance_scale,
                    max_sequence_length=cfg.max_sequence_length,
                    generator=torch.Generator("cuda").manual_seed(seed),
                ).images[0]
            
            handle.remove()
        
        images.append((strength, img))
    
    # Create grid with labels
    n = len(images)
    label_height = 40
    grid = Image.new('RGB', (cfg.width * n, cfg.height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    for i, (strength, img) in enumerate(images):
        # Paste image
        grid.paste(img, (i * cfg.width, label_height))
        
        # Add label
        label_text = f"{strength:+.0f}" if strength != 0 else "baseline"
        # Center the text
        bbox = draw.textbbox((0, 0), label_text)
        text_width = bbox[2] - bbox[0]
        x = i * cfg.width + (cfg.width - text_width) // 2
        draw.text((x, 10), label_text, fill=(0, 0, 0))
    
    return grid


def main():
    cfg = get_config()
    
    print("Loading models...")
    sae, _ = load_sae(cfg.sae_ckpt_path, device=cfg.sae_device)
    pipe = FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    
    feature_id = 1058  # Eye gaze feature
    prompt = "a young woman talking on a yellow phone"
    seed = 42
    
    # Test different parameter combinations
    experiments = [
        {
            "name": "original",
            "strengths": [-200, -100, -50, 0, 50, 100, 200],
            "step_range": (0, 5),
        },
    ]
    #     {
    #         "name": "stronger_values",
    #         "strengths": [-400, -200, 0, 200, 400, 600],
    #         "step_range": (0, 5),
    #     },
    #     {
    #         "name": "more_steps",
    #         "strengths": [-100, -50, 0, 50, 100, 200],
    #         "step_range": (0, 15),
    #     },
    #     {
    #         "name": "combined",
    #         "strengths": [-300, -150, 0, 150, 300, 500],
    #         "step_range": (0, 12),
    #     },
    #     {
    #         "name": "extreme",
    #         "strengths": [-500, -250, 0, 250, 500, 1000],
    #         "step_range": (0, 19),
    #     },
    # ]
    
    for exp in experiments:
        print(f"\n[{exp['name']}] strengths={exp['strengths']}, steps={exp['step_range']}")
        
        grid = generate_steering_grid(
            pipe, sae, cfg, feature_id, prompt, seed,
            strengths=exp['strengths'],
            step_range=exp['step_range'],
            label=exp['name'],
        )
        
        filename = f"steering_test_{exp['name']}_teeth_{seed:04d}.png"
        grid.save(filename)
        print(f"  Saved to {filename}")
        
        # Compute diffs
        baseline_idx = exp['strengths'].index(0)
        # ... could add diff computation here
    
    print("\nDone! Compare the output images to find best parameters.")


if __name__ == "__main__":
    main()