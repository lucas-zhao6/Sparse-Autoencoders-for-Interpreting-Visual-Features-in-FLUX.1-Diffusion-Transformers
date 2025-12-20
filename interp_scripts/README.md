# SAE Interpretability Pipeline for FLUX.1-dev

This pipeline analyzes Sparse Autoencoder features trained on FLUX diffusion model activations. It provides tools for collecting activations, analyzing feature statistics, selecting interpretable features via CLIP coherence, generating activation heatmaps, and running feature steering experiments.

## Quick Start

```bash
# Run the default pipeline (steps 1-6, 8)
python interp_scripts/run_pipeline.py

# Or run specific steps
python interp_scripts/run_pipeline.py --steps 1,2,3,4,5,6
```

## File Overview

### Configuration & Utilities

| File | Description |
|------|-------------|
| `config.py` | **Start here.** Central configuration for all scripts (paths, model settings, hyperparameters) |
| `run_pipeline.py` | Main orchestrator script to run multiple steps in sequence |
| `sae_model.py` | SAE model definitions (TopK and L1 variants) |
| `sae_utils.py` | SAE loading and inference utilities |
| `flux_hooks.py` | FLUX activation capture and steering hooks |
| `heatmap_utils.py` | Heatmap visualization utilities |
| `io_utils.py` | File I/O utilities |
| `test_prompts.txt` | Sample prompts for testing |

### Core Pipeline Scripts (Steps 1-9)

These scripts form the main interpretability workflow. Run them in order, or use `run_pipeline.py` to orchestrate.

| Step | Script | Description | Dependencies |
|------|--------|-------------|--------------|
| 1 | `01_collect_activations.py` | Generate images + collect SAE feature activations (S matrix) | SAE checkpoint, prompts |
| 2 | `02_analyze_features.py` | Compute feature statistics, find alive/dead features | Step 1 |
| 3 | `03_clip_embed.py` | Compute CLIP embeddings for all generated images | Step 1 |
| 4 | `04_select_features.py` | Select most interpretable features by CLIP coherence | Steps 2, 3 |
| 5 | `05_generate_overlays.py` | Generate activation heatmap overlays | Step 4 |
| 6 | `06_make_contact_sheet.py` | Create summary contact sheet visualization | Step 5 |
| 7 | `07_feature_steering.py` | Basic feature steering experiments | Step 4 |
| 8 | `08_visualize_decoder.py` | Visualize SAE decoder directions | Step 4 |
| 9 | `09_compare_layers.py` | Compare features between Layer 5 and Layer 15 | Both SAE checkpoints |

### Advanced Steering Scripts (Steps 10-11)

These scripts provide a more sophisticated steering demo workflow using LPIPS-based candidate selection.

| Step | Script | Description | Dependencies |
|------|--------|-------------|--------------|
| 10 | `10_find_steering_demos.py` | Find best prompt/seed pairs for steering demos using LPIPS responsiveness scoring | Step 4, LPIPS model |
| 11 | `11_assemble_steering_grids.py` | Assemble steering comparison grids from Step 10 outputs | Step 10 |

### Auxiliary Scripts

| File | Description |
|------|-------------|
| `steering_contact_sheet.py` | Create labeled contact sheets from steering grid outputs |
| `test_stronger_steering.py` | Test script for experimenting with steering parameters |

## Setup

### 1. Update Configuration

Edit `config.py` to set your SAE checkpoint paths:

```python
# In config.py, update these paths:
layer5_sae_path: str = "path/to/layer5/sae_best.pt"
layer15_sae_path: str = "path/to/layer15/sae_best.pt"
```

### 2. Verify SAE Checkpoints

```bash
ls -la sae_training_layer5_*/sae_best.pt
ls -la sae_training_layer15_*/sae_best.pt
```

## Running the Pipeline

### Option A: Run All Steps (Recommended)

```bash
cd /path/to/your/project
python interp_scripts/run_pipeline.py
```

This runs steps 1-6 and 8 by default (skips slow steering and layer comparison).

### Option B: Run Steps Individually

```bash
cd /path/to/your/project

# Core pipeline
python interp_scripts/01_collect_activations.py
python interp_scripts/02_analyze_features.py
python interp_scripts/03_clip_embed.py
python interp_scripts/04_select_features.py
python interp_scripts/05_generate_overlays.py
python interp_scripts/06_make_contact_sheet.py

# Optional: Basic steering
python interp_scripts/07_feature_steering.py

# Analysis
python interp_scripts/08_visualize_decoder.py

# Optional: Layer comparison (requires both SAEs)
python interp_scripts/09_compare_layers.py

# Optional: Advanced steering workflow
python interp_scripts/10_find_steering_demos.py
python interp_scripts/11_assemble_steering_grids.py
```

### Option C: Run Specific Steps

```bash
# Run only specific steps
python interp_scripts/run_pipeline.py --steps 1,2,3

# Run all steps including steering and layer comparison
python interp_scripts/run_pipeline.py --all

# Analyze layer 5 instead of layer 15
python interp_scripts/run_pipeline.py --layer 5
```

## Output Structure

After running the pipeline:

```
interp_output/
├── images/                         # Generated images
│   └── img_000000_p0000_r0_seed42.png
├── S_matrix.npy                    # Feature activations (num_images x hidden_dim)
├── manifest.jsonl                  # Image metadata
├── run_log.json                    # Run configuration
├── alive_features.npy              # Indices of non-dead features
├── activation_freq.npy             # Per-feature activation frequency
├── topk_per_feature.npz            # Top-k images per feature
├── feature_analysis_summary.json   # Feature statistics
├── clip_embeddings.npy             # CLIP image embeddings
├── clip_meta.json                  # CLIP metadata
├── selected_features.json          # Top features by coherence
├── overlays/                       # Heatmap overlays
│   ├── feature_1234/
│   │   ├── image_000000_feature_1234.png
│   │   └── image_000000_feature_1234_tokenmap.npy
│   └── overlay_summary.json
├── contact_sheet.png               # Grid of all features
├── contact_sheet_comparison.png    # Original vs overlay comparison
├── decoder_directions.png          # Decoder column visualizations
├── decoder_similarity_matrix.png   # Feature similarity
├── decoder_stats.json              # Decoder statistics
├── steering/                       # Basic steering results (step 7)
├── best_demos/                     # Advanced steering demos (step 10)
│   └── best_demo_candidates.json
├── steering_grids/                 # Assembled steering grids (step 11)
└── layer_comparison/               # Layer comparison (step 9)
```

## Key Features

- **TopK SAE Support**: Automatically detects TopK vs L1 SAE from checkpoint
- **Image Token Extraction**: Properly extracts only image tokens (not text tokens) from FLUX activations
- **Unified Config**: Single configuration file for all scripts
- **Feature Steering**: Both basic and advanced (LPIPS-scored) steering workflows
- **Decoder Visualization**: Analyze learned feature directions
- **Layer Comparison**: Compare features between early (layer 5) and late (layer 15) representations

## Troubleshooting

### "ModuleNotFoundError: No module named 'sae_model'"
Run from the parent directory:
```bash
cd /path/to/your/project
python interp_scripts/01_collect_activations.py
```

### "FileNotFoundError: SAE checkpoint not found"
Update the paths in `config.py` to match your actual SAE checkpoint locations.

### "CUDA out of memory"
Try these memory optimizations:
1. Reduce `height` and `width` in config (e.g., 256 → 128)
2. Reduce `reps_per_prompt` (e.g., 2 → 1)
3. Reduce number of prompts in `test_prompts.txt`

### "Token count doesn't match grid"
Ensure:
- `height` and `width` are divisible by `patch_size` (16)
- `n_image_tokens` = (height/16) × (width/16)

## Dependencies

```bash
pip install torch diffusers transformers pillow numpy scipy matplotlib tqdm lpips
```

## Quick Test

To verify the setup with minimal computation:

1. Edit `config.py`:
   ```python
   reps_per_prompt: int = 1
   ```

2. Create minimal prompts:
   ```bash
   echo "A red apple on a white table" > interp_scripts/test_prompts.txt
   echo "A blue car on a city street" >> interp_scripts/test_prompts.txt
   ```

3. Run:
   ```bash
   python interp_scripts/run_pipeline.py --steps 1,2,3,4
   ```
