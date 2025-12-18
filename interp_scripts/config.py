# config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    """Unified configuration for SAE interpretability pipeline."""
    
    # === Output directories ===
    out_dir: str = "interp_output"
    images_subdir: str = "images"
    overlays_subdir: str = "overlays"
    
    # === SAE paths (UPDATE THESE to match your actual directories) ===
    layer5_sae_path: str = "sae_training_layer5_t14_topk1200_x4_20251217_231756/sae_best.pt"
    layer15_sae_path: str = "sae_training_layer15_t14_topk1200_x4_20251218_020502/sae_best.pt"
    
    # Which layer to analyze (5 or 15)
    active_layer: int = 15
    
    @property
    def sae_ckpt_path(self) -> str:
        if self.active_layer == 5:
            return self.layer5_sae_path
        return self.layer15_sae_path
    
    # === FLUX model config ===
    model_id: str = "black-forest-labs/FLUX.1-dev"
    height: int = 256
    width: int = 256
    patch_size: int = 16  # FLUX patch size
    num_inference_steps: int = 20
    guidance_scale: float = 3.5
    max_sequence_length: int = 512
    target_step_number: int = 14  # 1-indexed
    
    @property
    def target_step_index(self) -> int:
        return self.target_step_number - 1
    
    @property
    def n_image_tokens(self) -> int:
        return (self.height // self.patch_size) * (self.width // self.patch_size)
    
    @property
    def image_token_grid(self) -> int:
        """Side length of square image token grid."""
        return self.height // self.patch_size  # 16 for 256x256
    
    # === Data collection config ===
    prompts_path: str = "test_prompts.txt"
    reps_per_prompt: int = 2
    base_seed: int = 42
    
    # === SAE inference config ===
    sae_device: str = "cuda"
    token_chunk_size: int = 256
    
    # === Analysis config ===
    eps_alive: float = 1e-4
    top_k_per_image: int = 20  # top-k features per image for S matrix analysis
    
    # === CLIP config ===
    clip_model_id: str = "openai/clip-vit-base-patch32"
    clip_batch_size: int = 32
    clip_device: str = "cuda"
    
    # === Feature selection config ===
    top_m_per_feature: int = 20  # candidates per feature for CLIP coherence
    top_features: int = 10  # number of features to visualize
    top_examples_per_feature: int = 4  # images per feature in contact sheet
    enforce_global_unique_images: bool = True
    
    # === Visualization config ===
    patch_grid: int = 16  # for blocky overlays
    alpha_scale: float = 0.8
    clip_percentile: float = 99.0
    
    # === Layer comparison ===
    compare_layers: bool = True


def get_config(**overrides) -> Config:
    """Get config with optional overrides."""
    cfg = Config()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
