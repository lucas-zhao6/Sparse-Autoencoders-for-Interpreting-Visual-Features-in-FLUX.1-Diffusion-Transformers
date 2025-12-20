# Sparse Autoencoders for Interpreting Visual Features in FLUX.1 Diffusion Transformers

This repository contains code and experiments for applying Sparse Autoencoders (SAEs) to interpret intermediate representations inside FLUX.1-dev, a text-to-image diffusion transformer. The goal is to decompose internal activations into a sparse set of features that can be inspected, visualized, selected for interpretability, and optionally used for feature steering.

## Repository Structure

```
.
├── README.md                      # This file
├── train_sae.py                   # SAE training script (L1 and TopK)
├── sae_model.py                   # SAE model definitions
├── activation_collection.py       # Collect FLUX activations for training
├── prompt_sampling.py             # Prompt sampling utilities
├── prompts.txt                    # Training prompts
├── visualization_example.py       # Basic visualization example
├── setup.py                       # Package setup
└── interp_scripts/                # Main interpretability pipeline
    ├── README.md                  # Detailed pipeline documentation
    ├── config.py                  # Configuration (START HERE)
    ├── run_pipeline.py            # Main orchestrator
    ├── 01_collect_activations.py  # Generate images + collect S matrix
    ├── 02_analyze_features.py     # Compute feature statistics
    ├── 03_clip_embed.py           # Compute CLIP embeddings
    ├── 04_select_features.py      # Select interpretable features
    ├── 05_generate_overlays.py    # Generate heatmap overlays
    ├── 06_make_contact_sheet.py   # Create contact sheet visualization
    ├── 07_feature_steering.py     # Basic feature steering
    ├── 08_visualize_decoder.py    # Visualize decoder directions
    ├── 09_compare_layers.py       # Compare layer 5 vs layer 15
    ├── 10_find_steering_demos.py  # Advanced: LPIPS-scored demo selection
    ├── 11_assemble_steering_grids.py  # Advanced: assemble steering grids
    └── [utility modules]          # sae_utils.py, flux_hooks.py, etc.
```

## Getting Started

### 1. Collect Activations (if training your own SAE)

```bash
# Collect activations from FLUX at layers 5 and 15
python activation_collection.py
```

This generates `flux_activations_layer5.npy` and `flux_activations_layer15.npy`.

### 2. Train an SAE

```bash
# Train L1 SAE
python train_sae.py --layer layer15 --timestep 14 --l1_coeff 1e-3 --epochs 10

# Or train TopK SAE (recommended for guaranteed sparsity)
python train_sae.py --layer layer15 --timestep 14 --topk --topk_k 1200 --epochs 10
```

### 3. Run the Interpretability Pipeline

```bash
# First, update interp_scripts/config.py with your SAE checkpoint paths
# Then run:
python interp_scripts/run_pipeline.py
```

See `interp_scripts/README.md` for detailed documentation.

## Project Overview

### Activation Capture and Token Geometry

FLUX represents an image as a grid of patch tokens. For an image of resolution \(H \times W\) and patch size 16:

$$N = \frac{H}{16} \times \frac{W}{16}$$

For example, 256×256 yields 16×16 = 256 image tokens.

During sampling, we hook a chosen FLUX transformer block (e.g., block 15) and capture the feedforward output tensor at a chosen diffusion step. The activations have dimension \(C=3072\).

### SAE Feature Encoding

A trained SAE maps each token activation \(x_p \in \mathbb{R}^{3072}\) to a sparse feature vector \(z_p \in \mathbb{R}^{F}\), where \(F\) is typically 12288 (4× expansion).

This repo supports:
- **L1 SAE**: Standard sparse autoencoder with L1 penalty on activations
- **TopK SAE**: Keeps only the top-k activations per input (guaranteed sparsity)

### Image-Level Score Matrix S

To get a single score per image per feature, we max-pool across tokens:

$$S_{i,f} = \max_p z_{i,p,f}$$

This yields \(S \in \mathbb{R}^{I \times F}\), the main artifact used for ranking and selecting features.

### Feature Selection via CLIP Coherence

For each alive feature:
1. Find the top-M images where the feature activates strongly
2. Compute CLIP embeddings for those images
3. Rank features by mean pairwise cosine similarity (coherence)

Features with high coherence tend to represent interpretable concepts.

### Overlay Visualizations

For selected features, we compute token-level activation maps and create heatmap overlays showing where features activate spatially.

### Feature Steering

Experimental capability to modify activations during sampling:
- **Basic steering** (step 7): Scale feature activations by different amounts
- **Advanced steering** (steps 10-11): Find optimal demos using LPIPS responsiveness

## Key Scripts

| Script | Purpose |
|--------|---------|
| `activation_collection.py` | Collect training activations from FLUX |
| `train_sae.py` | Train SAE (L1 or TopK) on collected activations |
| `interp_scripts/run_pipeline.py` | Run the full interpretability pipeline |
| `interp_scripts/config.py` | Central configuration for all pipeline scripts |

## Requirements

Python 3.10 or 3.11 recommended.

```bash
pip install torch diffusers transformers pillow numpy scipy matplotlib tqdm lpips
```

For full requirements, see individual script headers.

## Example Outputs

The pipeline produces:
- **Contact sheets**: Grid visualizations of top features with activation overlays
- **Heatmaps**: Per-image activation visualizations showing where features fire
- **Steering grids**: Comparison images showing the effect of amplifying/suppressing features
- **Decoder analysis**: Visualizations of learned feature directions

## References

This work builds on:
- [Anthropic's SAE research](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [FLUX.1-dev by Black Forest Labs](https://github.com/black-forest-labs/flux)

## License

See LICENSE file for details.
