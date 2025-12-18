# SAE Interpretability Pipeline for FLUX.1-dev

This pipeline analyzes Sparse Autoencoder features trained on FLUX diffusion model activations.

## Files Overview

```
interp_scripts/
├── config.py                    # Configuration (EDIT THIS FIRST)
├── sae_model.py                 # SAE model definitions (TopK + L1)
├── sae_utils.py                 # SAE loading and inference utilities
├── flux_hooks.py                # FLUX activation capture and steering hooks
├── heatmap_utils.py             # Heatmap visualization utilities
├── io_utils.py                  # File I/O utilities
├── test_prompts.txt             # Sample prompts for testing
├── run_pipeline.py              # Main runner script
├── 01_collect_activations.py    # Generate images + collect S matrix
├── 02_analyze_features.py       # Compute feature statistics
├── 03_clip_embed.py             # Compute CLIP embeddings
├── 04_select_features.py        # Select interpretable features
├── 05_generate_overlays.py      # Generate heatmap overlays
├── 06_make_contact_sheet.py     # Create summary visualizations
├── 07_feature_steering.py       # Feature steering experiments
├── 08_visualize_decoder.py      # Visualize decoder directions
└── 09_compare_layers.py         # Compare layer 5 vs layer 15
```

## Setup

### 1. Update Configuration

Edit `config.py` to set your SAE checkpoint paths:

```python
# In config.py, update these paths to match your trained SAEs:
layer5_sae_path: str = "sae_training_layer5_t14_topk1200_x4_20251217_231756/sae_best.pt"
layer15_sae_path: str = "sae_training_layer15_t14_topk1200_x4_20251218_020502/sae_best.pt"
```

### 2. Verify SAE Checkpoints

Make sure your SAE checkpoints exist:
```bash
ls -la sae_training_layer5_t14_topk1200_x4_*/sae_best.pt
ls -la sae_training_layer15_t14_topk1200_x4_*/sae_best.pt
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

# Step 1: Generate images and collect feature activations
python interp_scripts/01_collect_activations.py

# Step 2: Analyze which features are alive and compute statistics
python interp_scripts/02_analyze_features.py

# Step 3: Compute CLIP embeddings for semantic clustering
python interp_scripts/03_clip_embed.py

# Step 4: Select most interpretable features by CLIP coherence
python interp_scripts/04_select_features.py

# Step 5: Generate heatmap overlays showing where features activate
python interp_scripts/05_generate_overlays.py

# Step 6: Create contact sheet visualization
python interp_scripts/06_make_contact_sheet.py

# Step 7 (Optional, slow): Feature steering experiments
python interp_scripts/07_feature_steering.py

# Step 8: Visualize decoder directions
python interp_scripts/08_visualize_decoder.py

# Step 9 (Optional): Compare layer 5 vs layer 15
python interp_scripts/09_compare_layers.py
```

### Option C: Run Specific Steps

```bash
# Run only steps 1, 2, and 3
python interp_scripts/run_pipeline.py --steps 1,2,3

# Run all steps including steering and layer comparison
python interp_scripts/run_pipeline.py --all

# Analyze layer 5 instead of layer 15
python interp_scripts/run_pipeline.py --layer 5
```

## Output Structure

After running the pipeline, you'll have:

```
interp_output/                      # Default output directory
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
├── steering/                       # Feature steering results (if run)
└── layer_comparison/               # Layer comparison (if run)
```

## Key Changes from Previous Scripts

1. **TopK SAE Support**: Automatically detects TopK vs L1 SAE from checkpoint config
2. **Image Token Extraction**: Properly extracts only image tokens (not text tokens) from FLUX activations
3. **Unified Config**: Single configuration file for all scripts
4. **Feature Steering**: New capability to amplify/suppress features during generation
5. **Decoder Visualization**: Analyze learned feature directions
6. **Layer Comparison**: Compare features between layer 5 and layer 15

## Troubleshooting

### "ModuleNotFoundError: No module named 'sae_model'"
Make sure you're running from the parent directory, not from inside interp_scripts:
```bash
cd /path/to/your/project
python interp_scripts/01_collect_activations.py
```

### "FileNotFoundError: SAE checkpoint not found"
Update the paths in `config.py` to match your actual SAE checkpoint locations.

### "CUDA out of memory"
The scripts use memory optimization by default. If still OOM:
1. Reduce `height` and `width` in config (e.g., 256 → 128)
2. Reduce `reps_per_prompt` (e.g., 2 → 1)
3. Reduce number of prompts in test_prompts.txt

### "Token count doesn't match grid"
This happens if `height/width` don't match expected patch size. Ensure:
- `height` and `width` are divisible by `patch_size` (16)
- `n_image_tokens` = (height/16) * (width/16)

## Dependencies

```bash
pip install torch diffusers transformers pillow numpy scipy matplotlib tqdm
```

## Quick Test

To verify everything works with minimal computation:

1. Edit `config.py`:
   ```python
   reps_per_prompt: int = 1  # Reduce from 2
   ```

2. Create a minimal prompt file:
   ```bash
   echo "A red apple on a white table" > test_prompts.txt
   echo "A blue car on a city street" >> test_prompts.txt
   ```

3. Run:
   ```bash
   python interp_scripts/run_pipeline.py --steps 1,2,3,4
   ```

This generates 2 images and tests the core pipeline.
