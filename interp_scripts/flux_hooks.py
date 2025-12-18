# flux_hooks.py
import torch
from typing import Optional


class StepActivationCatcher:
    """Hook to capture activations at a specific denoising step."""
    
    def __init__(self, target_step_index: int):
        """
        Args:
            target_step_index: 0-indexed step to capture (e.g., 13 for step 14)
        """
        self.target_step_index = int(target_step_index)
        self.reset()
    
    def reset(self) -> None:
        self.call_index = 0
        self.captured: Optional[torch.Tensor] = None
    
    def hook_fn(self, module, inputs, output):
        if self.call_index == self.target_step_index:
            self.captured = output.detach().cpu()
        self.call_index += 1
        return output


class FeatureSteeringHook:
    """Hook to amplify/suppress specific SAE features during generation."""
    
    def __init__(
        self,
        sae: torch.nn.Module,
        feature_id: int,
        scale: float = 2.0,
        target_step_index: int = 13,
        n_image_tokens: Optional[int] = None,
    ):
        """
        Args:
            sae: Loaded SAE model
            feature_id: Which feature to steer
            scale: Multiplier (>1 amplifies, <1 suppresses, 0 ablates)
            target_step_index: 0-indexed step to intervene
            n_image_tokens: If set, only steer image tokens
        """
        self.sae = sae
        self.feature_id = feature_id
        self.scale = scale
        self.target_step_index = target_step_index
        self.n_image_tokens = n_image_tokens
        self.device = next(sae.parameters()).device
        self.reset()
    
    def reset(self):
        self.call_index = 0
    
    def hook_fn(self, module, inputs, output):
        if self.call_index == self.target_step_index:
            output = self._steer(output)
        self.call_index += 1
        return output
    
    def _steer(self, activation: torch.Tensor) -> torch.Tensor:
        """Apply steering to activation tensor."""
        original_shape = activation.shape
        original_device = activation.device
        original_dtype = activation.dtype
        
        # Flatten to (T, D)
        if activation.dim() == 3:
            batch_size = activation.shape[0]
            x = activation.reshape(-1, activation.shape[-1])
        else:
            batch_size = None
            x = activation
        
        x = x.float().to(self.device)
        
        # Determine which tokens to steer
        if self.n_image_tokens is not None:
            # Only steer image tokens (assumed to be first)
            img_tokens = x[:self.n_image_tokens]
            other_tokens = x[self.n_image_tokens:]
            
            # Encode image tokens
            z = self.sae.encode(img_tokens)
            
            # Modify feature
            z[:, self.feature_id] *= self.scale
            
            # Decode back
            img_tokens_modified = self.sae.decode(z)
            
            # Recombine
            x_modified = torch.cat([img_tokens_modified, other_tokens], dim=0)
        else:
            # Steer all tokens
            z = self.sae.encode(x)
            z[:, self.feature_id] *= self.scale
            x_modified = self.sae.decode(z)
        
        # Restore shape and dtype
        if batch_size is not None:
            x_modified = x_modified.reshape(original_shape)
        
        return x_modified.to(original_device, dtype=original_dtype)


class MultiFeatureSteeringHook:
    """Hook to steer multiple features simultaneously."""
    
    def __init__(
        self,
        sae: torch.nn.Module,
        feature_scales: dict,  # {feature_id: scale}
        target_step_index: int = 13,
        n_image_tokens: Optional[int] = None,
    ):
        self.sae = sae
        self.feature_scales = feature_scales
        self.target_step_index = target_step_index
        self.n_image_tokens = n_image_tokens
        self.device = next(sae.parameters()).device
        self.reset()
    
    def reset(self):
        self.call_index = 0
    
    def hook_fn(self, module, inputs, output):
        if self.call_index == self.target_step_index:
            output = self._steer(output)
        self.call_index += 1
        return output
    
    def _steer(self, activation: torch.Tensor) -> torch.Tensor:
        original_shape = activation.shape
        original_device = activation.device
        original_dtype = activation.dtype
        
        if activation.dim() == 3:
            x = activation.reshape(-1, activation.shape[-1])
        else:
            x = activation
        
        x = x.float().to(self.device)
        
        if self.n_image_tokens is not None:
            img_tokens = x[:self.n_image_tokens]
            other_tokens = x[self.n_image_tokens:]
            
            z = self.sae.encode(img_tokens)
            
            for fid, scale in self.feature_scales.items():
                z[:, fid] *= scale
            
            img_tokens_modified = self.sae.decode(z)
            x_modified = torch.cat([img_tokens_modified, other_tokens], dim=0)
        else:
            z = self.sae.encode(x)
            for fid, scale in self.feature_scales.items():
                z[:, fid] *= scale
            x_modified = self.sae.decode(z)
        
        x_modified = x_modified.reshape(original_shape)
        return x_modified.to(original_device, dtype=original_dtype)
