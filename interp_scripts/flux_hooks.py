# flux_hooks.py (corrected)
import torch
from typing import Optional, Dict, Tuple


class StepActivationCatcher:
    """Hook to capture activations at a specific denoising step."""
    
    def __init__(self, target_step_index: int):
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
    """
    Hook to amplify/suppress specific SAE features during generation.
    
    Uses additive steering for stronger effect (avoids SAE reconstruction error).
    Defaults to steering early steps (0-5) which have the most impact.
    """
    
    def __init__(
        self,
        sae: torch.nn.Module,
        feature_id: int,
        strength: float = 100.0,
        step_range: Tuple[int, int] = (0, 5),  # Early steps by default
        n_image_tokens: Optional[int] = None,
        normalize_direction: bool = True,
    ):
        """
        Args:
            sae: Loaded SAE model
            feature_id: Which feature to steer
            strength: Additive strength (positive=amplify, negative=suppress)
            step_range: (start, end) inclusive range of steps to steer
            n_image_tokens: If set, only steer image tokens
            normalize_direction: If True, normalize decoder direction to unit norm
        """
        self.sae = sae
        self.feature_id = feature_id
        self.strength = strength
        self.step_range = step_range
        self.n_image_tokens = n_image_tokens
        self.device = next(sae.parameters()).device
        
        # Extract and optionally normalize decoder direction
        self.direction = sae.decoder.weight[:, feature_id].detach().clone()
        if normalize_direction:
            self.direction = self.direction / self.direction.norm()
        
        self.reset()
    
    def reset(self):
        self.call_index = 0
        self.steer_count = 0
    
    def hook_fn(self, module, inputs, output):
        if self.step_range[0] <= self.call_index <= self.step_range[1]:
            output = self._steer(output)
            self.steer_count += 1
        self.call_index += 1
        return output
    
    def _steer(self, activation: torch.Tensor) -> torch.Tensor:
        """Additive steering in decoder direction."""
        original_shape = activation.shape
        original_device = activation.device
        original_dtype = activation.dtype
        
        if activation.dim() == 3:
            x = activation.reshape(-1, activation.shape[-1])
        else:
            x = activation
        
        x = x.float().to(self.device)
        direction = self.direction.to(x.device)
        
        if self.n_image_tokens is not None:
            # Only steer image tokens
            x[:self.n_image_tokens] = x[:self.n_image_tokens] + self.strength * direction
        else:
            x = x + self.strength * direction
        
        return x.reshape(original_shape).to(original_device, dtype=original_dtype)


class EncodeDecodeSteeringHook:
    """
    Alternative: Encode-modify-decode steering.
    
    Weaker effect due to reconstruction error, but more "principled" intervention.
    Use multi-step (all steps) for visible effect.
    """
    
    def __init__(
        self,
        sae: torch.nn.Module,
        feature_id: int,
        scale: float = 0.0,  # 0=ablate, >1=amplify, <1=suppress
        step_range: Tuple[int, int] = (0, 19),  # All steps needed for enc-dec
        n_image_tokens: Optional[int] = None,
    ):
        self.sae = sae
        self.feature_id = feature_id
        self.scale = scale
        self.step_range = step_range
        self.n_image_tokens = n_image_tokens
        self.device = next(sae.parameters()).device
        self.reset()
    
    def reset(self):
        self.call_index = 0
    
    def hook_fn(self, module, inputs, output):
        if self.step_range[0] <= self.call_index <= self.step_range[1]:
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
        
        if self.n_image_tokens is not None and x.shape[0] > self.n_image_tokens:
            img_tokens = x[:self.n_image_tokens]
            other_tokens = x[self.n_image_tokens:]
        else:
            img_tokens = x
            other_tokens = None
        
        z = self.sae.encode(img_tokens)
        
        if self.scale == 0:
            z[:, self.feature_id] = 0
        else:
            z[:, self.feature_id] *= self.scale
        
        img_modified = self.sae.decode(z)
        
        if other_tokens is not None:
            x_modified = torch.cat([img_modified, other_tokens], dim=0)
        else:
            x_modified = img_modified
        
        return x_modified.reshape(original_shape).to(original_device, dtype=original_dtype)


class MultiFeatureSteeringHook:
    """Steer multiple features simultaneously using additive steering."""
    
    def __init__(
        self,
        sae: torch.nn.Module,
        feature_strengths: Dict[int, float],  # {feature_id: strength}
        step_range: Tuple[int, int] = (0, 5),
        n_image_tokens: Optional[int] = None,
        normalize_directions: bool = True,
    ):
        self.sae = sae
        self.feature_strengths = feature_strengths
        self.step_range = step_range
        self.n_image_tokens = n_image_tokens
        self.device = next(sae.parameters()).device
        
        # Pre-compute combined steering direction
        self.combined_direction = torch.zeros(sae.decoder.weight.shape[0], device=self.device)
        for fid, strength in feature_strengths.items():
            direction = sae.decoder.weight[:, fid].detach().clone()
            if normalize_directions:
                direction = direction / direction.norm()
            self.combined_direction += strength * direction
        
        self.reset()
    
    def reset(self):
        self.call_index = 0
    
    def hook_fn(self, module, inputs, output):
        if self.step_range[0] <= self.call_index <= self.step_range[1]:
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
            x[:self.n_image_tokens] = x[:self.n_image_tokens] + self.combined_direction
        else:
            x = x + self.combined_direction
        
        return x.reshape(original_shape).to(original_device, dtype=original_dtype)