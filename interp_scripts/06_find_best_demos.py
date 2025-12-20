# 06_find_best_demos.py
"""
Step 6: Find best prompt/seed combinations for steering demos.
Uses relevant-but-not-saturated activation filtering + LPIPS responsiveness scoring.
"""

import os
import gc
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline
import lpips
from torchvision import transforms

from config import get_config
from sae_utils import load_sae
from flux_hooks import FeatureSteeringHook
from io_utils import load_manifest, write_json, ensure_dir


def load_activations(activations_path: str) -> np.ndarray:
    """Load precomputed activations."""
    return np.load(activations_path)


def get_relevant_moderate_candidates(
    feature_acts: np.ndarray,
    n: int = 50,
    min_percentile: float = 70,
    max_percentile: float = 95,
    seed: int = None,
) -> list:
    """
    Get images where feature is active but not saturated.
    
    Args:
        feature_acts: Activation values for this feature across all images
        n: Number of candidates to return
        min_percentile: Minimum activation percentile (ensures relevance)
        max_percentile: Maximum activation percentile (ensures room to grow)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    low = np.percentile(feature_acts, min_percentile)
    high = np.percentile(feature_acts, max_percentile)
    
    # For very sparse features, low might equal high
    if low >= high:
        # Fall back to top activating images below the max
        high = np.percentile(feature_acts, 99)
        low = np.percentile(feature_acts, 50)
    
    mask = (feature_acts >= low) & (feature_acts <= high)
    candidate_ids = np.where(mask)[0]
    
    if len(candidate_ids) > n:
        candidate_ids = np.random.choice(candidate_ids, size=n, replace=False)
    
    return list(candidate_ids)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor for LPIPS."""
    t = transforms.ToTensor()(img)
    return t.unsqueeze(0) * 2 - 1


def compute_responsiveness(
    baseline_img: Image.Image,
    steered_img: Image.Image,
    lpips_fn: lpips.LPIPS,
    device: str = "cuda",
) -> float:
    """Compute LPIPS distance between baseline and steered image."""
    baseline_t = pil_to_tensor(baseline_img).to(device)
    steered_t = pil_to_tensor(steered_img).to(device)
    with torch.no_grad():
        dist = lpips_fn(baseline_t, steered_t)
    return dist.item()


def generate_image(
    pipe: FluxPipeline,
    sae: torch.nn.Module,
    cfg,
    prompt: str,
    seed: int,
    feature_id: int = None,
    strength: float = 0.0,
    step_range: tuple = (0, 5),
) -> Image.Image:
    """Generate an image, optionally with steering."""
    
    if strength == 0.0 or feature_id is None:
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                height=cfg.height,
                width=cfg.width,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                max_sequence_length=cfg.max_sequence_length,
                generator=torch.Generator("cuda").manual_seed(seed),
            )
        return out.images[0]
    
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
            )
        return out.images[0]
    finally:
        handle.remove()


def score_candidate(
    pipe: FluxPipeline,
    sae: torch.nn.Module,
    lpips_fn: lpips.LPIPS,
    cfg,
    prompt: str,
    seed: int,
    feature_id: int,
    test_strengths: list = [100.0],
    step_range: tuple = (0, 5),
    save_dir: str = None,
    img_id: int = None,
) -> dict:
    """
    Score a single candidate by generating baseline + steered and computing LPIPS.
    Tests multiple strengths and returns the best.
    """
    
    # Generate baseline once
    baseline = generate_image(
        pipe, sae, cfg, prompt, seed,
        feature_id=None, strength=0.0,
    )
    
    if save_dir is not None and img_id is not None:
        baseline.save(os.path.join(save_dir, f"{img_id:06d}_baseline.png"))
    
    best_result = None
    best_score = -1
    
    for strength in test_strengths:
        steered_pos = generate_image(
            pipe, sae, cfg, prompt, seed,
            feature_id=feature_id,
            strength=strength,
            step_range=step_range,
        )
        
        steered_neg = generate_image(
            pipe, sae, cfg, prompt, seed,
            feature_id=feature_id,
            strength=-strength,
            step_range=step_range,
        )
        
        pos_dist = compute_responsiveness(baseline, steered_pos, lpips_fn)
        neg_dist = compute_responsiveness(baseline, steered_neg, lpips_fn)
        
        # Directional consistency: good demos change similarly in both directions
        if max(pos_dist, neg_dist) > 0:
            consistency = min(pos_dist, neg_dist) / max(pos_dist, neg_dist)
        else:
            consistency = 0
        
        # Combined score: average magnitude weighted by consistency
        combined_score = (pos_dist + neg_dist) / 2 * (0.5 + 0.5 * consistency)
        
        if combined_score > best_score:
            best_score = combined_score
            best_result = {
                "best_strength": strength,
                "pos_lpips": pos_dist,
                "neg_lpips": neg_dist,
                "consistency": consistency,
                "combined_score": combined_score,
            }
            
            if save_dir is not None and img_id is not None:
                steered_pos.save(os.path.join(save_dir, f"{img_id:06d}_pos_{strength:.0f}.png"))
                steered_neg.save(os.path.join(save_dir, f"{img_id:06d}_neg_{strength:.0f}.png"))
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_result


def find_best_demos_for_feature(
    pipe: FluxPipeline,
    sae: torch.nn.Module,
    lpips_fn: lpips.LPIPS,
    cfg,
    feature_id: int,
    feature_acts: np.ndarray,
    id_to_record: dict,
    n_candidates_to_sample: int = 50,
    n_candidates_to_test: int = 10,
    n_final: int = 2,
    min_percentile: float = 70,
    max_percentile: float = 95,
    test_strengths: list = [100.0],
    step_range: tuple = (0, 5),
    seed: int = None,
    save_images: bool = False,
    save_dir: str = None,
) -> tuple:
    """
    Find the best demo candidates for a given feature.
    
    Returns:
        (best_demos, all_scored): Top demos and all scored candidates
    """
    
    # Stage 1: Get relevant but not saturated candidates
    candidate_ids = get_relevant_moderate_candidates(
        feature_acts,
        n=n_candidates_to_sample,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
        seed=seed,
    )
    
    # Filter to only IDs that exist in manifest (handle both int and str keys)
    valid_ids = []
    for i in candidate_ids:
        if i in id_to_record:
            valid_ids.append(i)
        elif str(i) in id_to_record:
            valid_ids.append(str(i))
    
    if len(valid_ids) < n_candidates_to_test:
        print(f"    Warning: only {len(valid_ids)} valid candidates found")
    
    # Limit to n_candidates_to_test
    test_ids = valid_ids[:n_candidates_to_test]
    
    # Setup save directory if needed
    feat_save_dir = None
    if save_images and save_dir is not None:
        feat_save_dir = os.path.join(save_dir, f"feature_{feature_id}")
        ensure_dir(feat_save_dir)
    
    # Stage 2: Score each candidate
    scored_candidates = []
    
    for img_id in tqdm(test_ids, desc=f"  Scoring candidates", leave=False):
        rec = id_to_record[img_id] if img_id in id_to_record else id_to_record[str(img_id)]
        prompt = rec["prompt"]
        gen_seed = int(rec["seed"])
        
        result = score_candidate(
            pipe, sae, lpips_fn, cfg,
            prompt, gen_seed, feature_id,
            test_strengths=test_strengths,
            step_range=step_range,
            save_dir=feat_save_dir,
            img_id=int(img_id) if isinstance(img_id, (int, np.integer)) else int(img_id),
        )
        
        scored_candidates.append({
            "image_id": int(img_id) if isinstance(img_id, (int, np.integer)) else int(img_id),
            "prompt": prompt,
            "seed": gen_seed,
            "activation": float(feature_acts[int(img_id) if isinstance(img_id, (int, np.integer)) else int(img_id)]),
            **result,
        })
    
    # Stage 3: Sort by combined score and return top n_final
    scored_candidates.sort(key=lambda x: -x["combined_score"])
    
    return scored_candidates[:n_final], scored_candidates


def main():
    cfg = get_config()
    
    selected_path = os.path.join(cfg.out_dir, "selected_features.json")
    manifest_path = os.path.join(cfg.out_dir, "manifest.jsonl")
    activations_path = os.path.join(cfg.out_dir, "S_matrix.npy")
    demos_dir = os.path.join(cfg.out_dir, "best_demos")
    checkpoint_path = os.path.join(demos_dir, "checkpoint.json")
    
    if not os.path.exists(selected_path):
        raise FileNotFoundError("Run 04_select_features.py first")
    
    if not os.path.exists(activations_path):
        raise FileNotFoundError(
            f"Activations file not found: {activations_path}\n"
            "This script requires precomputed activations."
        )
    
    print("Loading data...")
    with open(selected_path, "r") as f:
        selected = json.load(f)
    
    _, id_to_record = load_manifest(manifest_path, cfg.out_dir)
    ensure_dir(demos_dir)
    
    print(f"Loading activations from {activations_path}")
    all_activations = load_activations(activations_path)
    print(f"  Activations shape: {all_activations.shape}")
    
    # Validate dimensions
    n_images, n_features = all_activations.shape
    print(f"  {n_images} images, {n_features} features")
    
    print(f"Loading SAE from {cfg.sae_ckpt_path}")
    sae, _ = load_sae(cfg.sae_ckpt_path, device=cfg.sae_device)
    
    print(f"Loading FLUX from {cfg.model_id}")
    pipe = FluxPipeline.from_pretrained(cfg.model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    
    print("Loading LPIPS model...")
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    # Config
    config = {
        "n_candidates_to_sample": 50,
        "n_candidates_to_test": 10,
        "n_final": 2,
        "test_strengths": [50.0, 100.0, 200.0],
        "step_range": (0, 5),
        "min_percentile": 70,
        "max_percentile": 95,
        "seed": 42,
        "save_images": True,
    }
    
    features = selected["features"][5:10]
    
    # Load checkpoint or initialize
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            all_results = json.load(f)
        completed_features = {f["feature_id"] for f in all_results["features"]}
        print(f"  {len(completed_features)} features already completed")
    else:
        all_results = {
            "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
            "features": [],
        }
        completed_features = set()
    
    print(f"\nFinding best demos for {len(features)} features...")
    print(f"Sampling from {config['min_percentile']}-{config['max_percentile']} percentile")
    print(f"Testing {config['n_candidates_to_test']} candidates, keeping top {config['n_final']}")
    print(f"Testing strengths: {config['test_strengths']}\n")
    
    for feat in tqdm(features, desc="Features"):
        feature_id = int(feat["feature_id"])
        
        # Skip if already done
        if feature_id in completed_features:
            print(f"\nFeature {feature_id}: already completed, skipping")
            continue
        
        # Validate feature_id
        if feature_id >= n_features:
            print(f"\nFeature {feature_id}: out of bounds (max {n_features-1}), skipping")
            continue
        
        print(f"\nFeature {feature_id}:")
        
        # Get activations for this feature
        feature_acts = all_activations[:, feature_id]
        
        best_demos, all_scored = find_best_demos_for_feature(
            pipe=pipe,
            sae=sae,
            lpips_fn=lpips_fn,
            cfg=cfg,
            feature_id=feature_id,
            feature_acts=feature_acts,
            id_to_record=id_to_record,
            n_candidates_to_sample=config["n_candidates_to_sample"],
            n_candidates_to_test=config["n_candidates_to_test"],
            n_final=config["n_final"],
            min_percentile=config["min_percentile"],
            max_percentile=config["max_percentile"],
            test_strengths=config["test_strengths"],
            step_range=config["step_range"],
            seed=config["seed"],
            save_images=config["save_images"],
            save_dir=demos_dir,
        )
        
        feature_result = {
            "feature_id": feature_id,
            "best_demos": best_demos,
            "all_scored": all_scored,
        }
        all_results["features"].append(feature_result)
        
        # Save checkpoint after each feature
        write_json(checkpoint_path, all_results)
        
        # Print results
        print(f"  Best candidates:")
        for i, demo in enumerate(best_demos):
            print(f"    {i+1}. image_id={demo['image_id']}, "
                  f"score={demo['combined_score']:.4f} "
                  f"(pos={demo['pos_lpips']:.4f}, neg={demo['neg_lpips']:.4f}, "
                  f"consistency={demo['consistency']:.2f}, strength={demo['best_strength']:.0f})")
    
    # Save final results
    output_path = os.path.join(demos_dir, "best_demo_candidates.json")
    write_json(output_path, all_results)
    print(f"\nResults saved to {output_path}")
    
    # Create simplified version
    simplified = {}
    for feat_result in all_results["features"]:
        fid = feat_result["feature_id"]
        simplified[fid] = [
            {"image_id": d["image_id"], "strength": d["best_strength"]}
            for d in feat_result["best_demos"]
        ]
    
    simplified_path = os.path.join(demos_dir, "best_demo_ids.json")
    write_json(simplified_path, simplified)
    print(f"Simplified mapping saved to {simplified_path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Checkpoint removed")


if __name__ == "__main__":
    main()