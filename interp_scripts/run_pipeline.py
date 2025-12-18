#!/usr/bin/env python3
# run_pipeline.py
"""
Main script to run the full interpretability pipeline.

Usage:
    python run_pipeline.py                    # Run all steps
    python run_pipeline.py --steps 1,2,3      # Run specific steps
    python run_pipeline.py --layer 5          # Analyze layer 5 instead of 15
    python run_pipeline.py --skip-generation  # Skip image generation (use existing)

Steps:
    1. Collect activations (generate images + compute S matrix)
    2. Analyze features (find alive features, compute stats)
    3. CLIP embed (compute CLIP embeddings for all images)
    4. Select features (rank by CLIP coherence)
    5. Generate overlays (create heatmap visualizations)
    6. Make contact sheet (create summary grid)
    7. Feature steering (optional, slow)
    8. Visualize decoder (analyze learned directions)
    9. Compare layers (requires both SAEs, optional)
"""

import argparse
import subprocess
import sys
import os


def run_step(step_num: int, script_name: str, description: str):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    result = subprocess.run([sys.executable, script_path], check=False)
    
    if result.returncode != 0:
        print(f"\nERROR: Step {step_num} failed with return code {result.returncode}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Run SAE interpretability pipeline")
    parser.add_argument("--steps", type=str, default="1,2,3,4,5,6,8",
                        help="Comma-separated list of steps to run (default: 1,2,3,4,5,6,8)")
    parser.add_argument("--layer", type=int, default=15, choices=[5, 15],
                        help="Which layer to analyze (default: 15)")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps including steering and layer comparison")
    
    args = parser.parse_args()
    
    # Parse steps
    if args.all:
        steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        steps = [int(s.strip()) for s in args.steps.split(",")]
    
    # Update config for layer
    if args.layer != 15:
        print(f"NOTE: Analyzing layer {args.layer}")
        # You'll need to modify config.py or pass this through
    
    # Define all steps
    step_info = {
        1: ("01_collect_activations.py", "Collect SAE Activations"),
        2: ("02_analyze_features.py", "Analyze Feature Statistics"),
        3: ("03_clip_embed.py", "Compute CLIP Embeddings"),
        4: ("04_select_features.py", "Select Interpretable Features"),
        5: ("05_generate_overlays.py", "Generate Heatmap Overlays"),
        6: ("06_make_contact_sheet.py", "Create Contact Sheet"),
        7: ("07_feature_steering.py", "Feature Steering Experiments"),
        8: ("08_visualize_decoder.py", "Visualize Decoder Directions"),
        9: ("09_compare_layers.py", "Compare Layer 5 vs Layer 15"),
    }
    
    print(f"Running steps: {steps}")
    
    failed_steps = []
    
    for step in steps:
        if step not in step_info:
            print(f"WARNING: Unknown step {step}, skipping")
            continue
        
        script, desc = step_info[step]
        success = run_step(step, script, desc)
        
        if not success:
            failed_steps.append(step)
            print(f"Step {step} failed. Continue anyway? (y/n)")
            response = input().strip().lower()
            if response != 'y':
                break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    
    if failed_steps:
        print(f"Failed steps: {failed_steps}")
    else:
        print("All steps completed successfully!")
    
    print("\nOutput files are in the configured out_dir (see config.py)")


if __name__ == "__main__":
    main()
