# Sparse Autoencoders for Interpreting Visual Features in FLUX.1 Diffusion Transformers

This repository contains code and experiments for applying Sparse Autoencoders (SAEs) to interpret intermediate representations inside a text to image diffusion transformer, FLUX.1 dev. The goal is to decompose internal activations into a sparse set of features that can be inspected, visualized, selected for interpretability, and optionally used for feature steering.

Most of the work is implemented as a reproducible pipeline in `interp_scripts/`.

## Project summary

We generate images from a prompt set, capture FLUX internal activations at a chosen transformer block and diffusion step, encode those activations using a trained SAE, and compute an image level feature score matrix $$S \in \mathbb{R}^{I \times F}$$, where $$I$$ is the number of generated images and $$F$$ is the SAE hidden dimension (typically $$F=12288$$). We then

- identify alive features and their activation statistics
- find top activating images per feature
- optionally compute CLIP embeddings to rank features by semantic coherence among their top examples
- generate token-level activation heatmaps and overlay them on images to localize where features activate
- produce contact sheets for report figures
- run feature steering experiments

## Repository layout

Most scripts are inside `interp_scripts/`. Earlier or prototype scripts may also exist at the repository root.

If you are new to this repository, start with `interp_scripts/config.py` and `interp_scripts/run_pipeline.py`.

## Method overview

### Activation capture and token geometry

FLUX represents an image as a grid of patch tokens. For an image of resolution $$H \times W$$ and patch size $$16$$, the number of image tokens is

$$
N = \frac{H}{16} \times \frac{W}{16}
$$

For example, $$256 \times 256$$ yields $$16 \times 16 = 256$$ image tokens.

During sampling, we hook a chosen FLUX transformer block (for example, block 15) and capture the feedforward output tensor for a chosen diffusion step index $$t$$. The captured activations are tokenized spatially and have a feature dimension $$C=3072$$ for the experiments in this repo.

### SAE feature encoding

A trained SAE maps each token activation $$x_p \in \mathbb{R}^{3072}$$ to an SAE feature vector $$z_p \in \mathbb{R}^{F}$$, usually with $$F=12288$$. This repo includes support for both L1 SAEs and TopK SAEs, depending on checkpoint configuration.

### Image level score matrix $$S$$

To get a single score per image per feature, we max pool across image tokens:

$$
S_{i,f} = \max_{p} z_{i,t,p,f}.
$$

Collecting across images yields $$S \in \mathbb{R}^{I \times F}$$. This is the main artifact used for ranking, selecting, and visualizing features.

### Alive feature statistics and top-k examples

A feature is considered alive if it exceeds a threshold $$\varepsilon$$ on at least a minimum number of images. For each alive feature, we compute the top $$k$$ images by $$S_{i,f}$$. These are used for qualitative inspection and as the input to CLIP coherence selection.

### Feature selection via CLIP coherence

For each feature, take the top $$M$$ images by $$S_{i,f}$$, with a uniqueness constraint so multiple images from the same prompt are not selected. Compute CLIP embeddings for those images and rank features by semantic coherence, typically mean pairwise cosine similarity within that feature’s top set. Features with high coherence tend to be easier to interpret.

### Overlay visualizations

For selected features, compute token-level activation maps $$z_{i,t,p,f}$$ and reshape to the patch grid. We then render overlays on top of the original image to show where a feature activates spatially. The overlay pipeline can include patch-based aggregation to produce discrete activation maps aligned to the patch grid.

### Feature steering

The repository includes an experimental steering script that modifies internal activations during sampling along a direction derived from the SAE for a given feature. Steering behavior depends strongly on which layer and steps are intervened on, the chosen direction, and the magnitude schedule. Steering code is included but is considered experimental.

## Requirements

Recommended Python version is 3.10 or 3.11 for best compatibility with common diffusion libraries.

Install dependencies from requirements.txt



