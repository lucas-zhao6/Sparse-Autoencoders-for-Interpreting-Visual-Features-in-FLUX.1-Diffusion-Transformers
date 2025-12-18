# train_sae.py - Complete training script with all improvements
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sae_model import SparseAutoencoder, TopKSparseAutoencoder
from tqdm import tqdm
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

class ActivationDataset(Dataset):
    def __init__(self, activations_path, timestep=14, normalize=False):
        """
        Load activations for a single timestep across all images.
        
        Args:
            activations_path: Path to .npy file
            timestep: Which timestep to extract (0-19)
            normalize: Whether to standardize per-dimension
        """
        print(f"Extracting timestep {timestep} from all images...")
        
        mmap_data = np.load(activations_path, mmap_mode='r')
        total_samples = len(mmap_data)
        
        # Data layout: 500 images × 20 timesteps × 4096 tokens
        n_images = 500
        n_timesteps = 20
        n_tokens = 4096
        vectors_per_image = n_timesteps * n_tokens  # 81920
        
        assert total_samples == n_images * vectors_per_image, \
            f"Expected {n_images * vectors_per_image}, got {total_samples}"
        
        # Extract only the specified timestep from each image
        indices = []
        for img_idx in range(n_images):
            start = img_idx * vectors_per_image + timestep * n_tokens
            end = start + n_tokens
            indices.extend(range(start, end))
        
        indices = np.array(indices)
        subset_size = len(indices)
        
        print(f"Loading {subset_size:,} vectors (timestep {timestep} from {n_images} images)")
        
        self.activations = np.array(mmap_data[indices], dtype=np.float32)
        
        print(f"✓ Loaded {self.activations.nbytes / 1024**3:.1f} GB into RAM")
        
        # Compute statistics
        self.raw_mean = self.activations.mean()
        self.raw_std = self.activations.std()
        print(f"Dataset mean: {self.raw_mean:.4f}, std: {self.raw_std:.4f}")
        
        # Optional per-dimension normalization
        self.normalize = normalize
        if normalize:
            self.dim_mean = self.activations.mean(axis=0, keepdims=True)
            self.dim_std = self.activations.std(axis=0, keepdims=True) + 1e-8
            self.activations = (self.activations - self.dim_mean) / self.dim_std
            print(f"✓ Applied per-dimension normalization")
            print(f"Post-norm mean: {self.activations.mean():.4f}, std: {self.activations.std():.4f}")
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.activations[idx])


def create_dataloaders(dataset, batch_size, val_split=0.1, num_workers=4):
    """Create train and validation dataloaders with proper splitting"""
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    # Use random split with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    print(f"Train samples: {n_train:,}, Val samples: {n_val:,}")
    
    return train_loader, val_loader


def evaluate(model, dataloader, device):
    """Run validation and return metrics"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_l1 = 0
    total_l0 = 0
    total_frac = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            losses = model.loss(batch)
            
            total_loss += losses['total_loss'].item()
            total_recon += losses['recon_loss'].item()
            total_l1 += losses['l1_loss'].item()
            total_l0 += losses['l0_sparsity'].item()
            total_frac += losses['frac_active'].item()
            n_batches += 1
    
    model.train()
    
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon / n_batches,
        'l1_loss': total_l1 / n_batches,
        'l0_sparsity': total_l0 / n_batches,
        'frac_active': total_frac / n_batches,
    }


def plot_training_curves(training_log, output_dir):
    """Generate and save training curve plots"""
    epochs = [x['epoch'] for x in training_log]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(epochs, [x['train_loss'] for x in training_log], label='Train')
    axes[0, 0].plot(epochs, [x['val_loss'] for x in training_log], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Reconstruction loss
    axes[0, 1].plot(epochs, [x['train_recon'] for x in training_log], label='Train')
    axes[0, 1].plot(epochs, [x['val_recon'] for x in training_log], label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # L1 loss
    axes[0, 2].plot(epochs, [x['train_l1'] for x in training_log], label='Train')
    axes[0, 2].plot(epochs, [x['val_l1'] for x in training_log], label='Val')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('L1 Loss')
    axes[0, 2].set_title('L1 Loss')
    axes[0, 2].legend()
    
    # Sparsity (frac_active)
    axes[1, 0].plot(epochs, [x['train_sparsity'] * 100 for x in training_log], label='Train')
    axes[1, 0].plot(epochs, [x['val_sparsity'] * 100 for x in training_log], label='Val')
    axes[1, 0].axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Target max')
    axes[1, 0].axhline(y=5, color='g', linestyle='--', alpha=0.5, label='Target min')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('% Features Active')
    axes[1, 0].set_title('Sparsity (% Active)')
    axes[1, 0].legend()
    
    # L0 sparsity (avg features per sample)
    axes[1, 1].plot(epochs, [x['train_l0'] for x in training_log], label='Train')
    axes[1, 1].plot(epochs, [x['val_l0'] for x in training_log], label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Avg Active Features')
    axes[1, 1].set_title('L0 Sparsity')
    axes[1, 1].legend()
    
    # Learning rate
    axes[1, 2].plot(epochs, [x['lr'] for x in training_log])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150)
    plt.close()
    print(f"✓ Saved training curves to {output_dir}/training_curves.png")


def plot_feature_density(density, output_dir):
    """Plot histogram of feature activation densities"""
    density_np = density.cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Log-scale histogram
    axes[0].hist(density_np, bins=100, log=True)
    axes[0].axvline(x=1e-4, color='r', linestyle='--', label='Dead threshold')
    axes[0].set_xlabel('Activation Density')
    axes[0].set_ylabel('Count (log)')
    axes[0].set_title('Feature Density Distribution')
    axes[0].legend()
    
    # Sorted density plot
    sorted_density = np.sort(density_np)[::-1]
    axes[1].plot(sorted_density)
    axes[1].axhline(y=1e-4, color='r', linestyle='--', label='Dead threshold')
    axes[1].set_xlabel('Feature Rank')
    axes[1].set_ylabel('Activation Density')
    axes[1].set_title('Sorted Feature Densities')
    axes[1].set_yscale('log')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_density.png', dpi=150)
    plt.close()
    print(f"✓ Saved feature density plot to {output_dir}/feature_density.png")


def train_sae(
    layer_name='layer5',
    timestep=14,
    l1_coeff=1e-3,
    expansion_factor=4,
    num_epochs=10,
    batch_size=2048,
    learning_rate=1e-3,
    weight_decay=1e-5,
    warmup_steps=2000,
    resample_dead_every=5000,
    val_split=0.1,
    normalize_data=False,
    device='cuda',
    use_topk=False,
    topk_k=64,
):
    """
    Train SAE on FLUX activations for a specific timestep.
    
    Args:
        layer_name: 'layer5' or 'layer15'
        timestep: Which diffusion timestep to train on (0-19)
        l1_coeff: L1 sparsity penalty (higher = sparser)
        expansion_factor: hidden_dim = input_dim * expansion_factor
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: Number of warmup steps for LR schedule
        resample_dead_every: Resample dead features every N steps (0 to disable)
        val_split: Fraction of data for validation
        normalize_data: Whether to standardize per-dimension
        device: 'cuda' or 'cpu'
        use_topk: Use TopK SAE instead of L1 SAE
        topk_k: Number of active features for TopK SAE
    """
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_type = f'topk{topk_k}' if use_topk else f'l1{l1_coeff:.0e}'
    output_dir = f'sae_training_{layer_name}_t{timestep}_{model_type}_x{expansion_factor}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    input_dim = 3072
    hidden_dim = input_dim * expansion_factor
    
    config = {
        'layer': layer_name,
        'timestep': timestep,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'expansion_factor': expansion_factor,
        'l1_coeff': l1_coeff,
        'use_topk': use_topk,
        'topk_k': topk_k if use_topk else None,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'warmup_steps': warmup_steps,
        'resample_dead_every': resample_dead_every,
        'val_split': val_split,
        'normalize_data': normalize_data,
        'device': device,
        'save_every': 2,
    }
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    # Save config
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration:")
    print(json.dumps(config, indent=2))
    
    # Load dataset
    dataset_path = f'flux_activations_{layer_name}.npy'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found!")
        return None, None
    
    dataset = ActivationDataset(dataset_path, timestep=timestep, normalize=normalize_data)
    train_loader, val_loader = create_dataloaders(
        dataset, batch_size, val_split=val_split, num_workers=4
    )
    
    print(f"\nDataset size: {len(dataset):,} samples")
    print(f"Batches per epoch: {len(train_loader):,}")
    print(f"Hidden dim: {hidden_dim:,} ({expansion_factor}x expansion)")
    
    # Initialize model
    if use_topk:
        model = TopKSparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=topk_k
        ).to(device)
        print(f"\n*** Using TopK SAE with k={topk_k} ***")
        print(f"*** Guaranteed sparsity: {topk_k/hidden_dim*100:.2f}% ***\n")
    else:
        model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            l1_coeff=l1_coeff
        ).to(device)
        print(f"\n*** Using L1 SAE with l1_coeff={l1_coeff} ***")
        print(f"*** Target sparsity: 5-15% of features active ***\n")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine learning rate scheduler with warmup
    total_steps = num_epochs * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    global_step = 0
    training_log = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_metrics = {
            'loss': 0, 'recon': 0, 'l1': 0, 'l0': 0, 'frac': 0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            losses = model.loss(batch)
            losses['total_loss'].backward()
            
            # Gradient clipping and norm tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # CRITICAL: Normalize decoder weights
            model.normalize_decoder()
            
            scheduler.step()
            
            # Track metrics
            epoch_metrics['loss'] += losses['total_loss'].item()
            epoch_metrics['recon'] += losses['recon_loss'].item()
            epoch_metrics['l1'] += losses['l1_loss'].item()
            epoch_metrics['l0'] += losses['l0_sparsity'].item()
            epoch_metrics['frac'] += losses['frac_active'].item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{losses["total_loss"].item():.4f}',
                    'recon': f'{losses["recon_loss"].item():.4f}',
                    'L0': f'{losses["l0_sparsity"].item():.0f}',
                    'sparse': f'{losses["frac_active"].item()*100:.1f}%',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                    'grad': f'{grad_norm:.2f}'
                })
            
            # Resample dead features periodically
            if resample_dead_every > 0 and global_step > 0 and global_step % resample_dead_every == 0:
                dead_mask, density = model.get_dead_features(train_loader, device)
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    n_resampled = model.resample_dead_features(dead_mask, train_loader, device)
                    print(f"\n🔄 Step {global_step}: Resampled {n_resampled} dead features "
                          f"({n_dead/hidden_dim*100:.1f}% were dead)")
            
            global_step += 1
        
        # Epoch averages
        n_batches = len(train_loader)
        train_metrics = {
            'loss': epoch_metrics['loss'] / n_batches,
            'recon': epoch_metrics['recon'] / n_batches,
            'l1': epoch_metrics['l1'] / n_batches,
            'l0': epoch_metrics['l0'] / n_batches,
            'frac': epoch_metrics['frac'] / n_batches,
        }
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        # Sparsity assessment
        sparsity_pct = train_metrics['frac'] * 100
        if sparsity_pct > 30:
            sparsity_status = "⚠️  TOO DENSE"
        elif sparsity_pct > 15:
            sparsity_status = "⚡ ACCEPTABLE"
        else:
            sparsity_status = "✓ GOOD"
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Train Recon: {train_metrics['recon']:.4f} | Val Recon: {val_metrics['recon_loss']:.4f}")
        print(f"  Train L1: {train_metrics['l1']:.4f} | Val L1: {val_metrics['l1_loss']:.4f}")
        print(f"  Train L0: {train_metrics['l0']:.0f} | Val L0: {val_metrics['l0_sparsity']:.0f}")
        print(f"  Sparsity: {sparsity_pct:.1f}% active  {sparsity_status}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*70}\n")
        
        # Log metrics
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_recon': train_metrics['recon'],
            'train_l1': train_metrics['l1'],
            'train_l0': train_metrics['l0'],
            'train_sparsity': train_metrics['frac'],
            'val_loss': val_metrics['loss'],
            'val_recon': val_metrics['recon_loss'],
            'val_l1': val_metrics['l1_loss'],
            'val_l0': val_metrics['l0_sparsity'],
            'val_sparsity': val_metrics['frac_active'],
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'config': config,
            }, f'{output_dir}/sae_best.pt')
            print(f"✓ New best model saved (val_loss={best_val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0 or (epoch + 1) == num_epochs:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'training_log': training_log,
            }, f'{output_dir}/sae_epoch_{epoch+1}.pt')
            print(f"✓ Saved checkpoint: sae_epoch_{epoch+1}.pt")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    final_metrics = model.compute_metrics(val_loader, device)
    print(f"Reconstruction Loss: {final_metrics['recon_loss']:.4f}")
    print(f"R² Score: {final_metrics['r2_score']:.4f}")
    print(f"Avg L0 (features/sample): {final_metrics['avg_l0']:.1f}")
    print(f"Fraction Active: {final_metrics['frac_active']*100:.1f}%")
    print(f"Dead Features (<0.01%): {final_metrics['dead_features']}")
    print(f"Rare Features (0.01-1%): {final_metrics['rare_features']}")
    print(f"Common Features (1-10%): {final_metrics['common_features']}")
    print(f"Very Common (>10%): {final_metrics['very_common_features']}")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_log': training_log,
        'final_metrics': {k: v if not isinstance(v, torch.Tensor) else v.tolist() 
                         for k, v in final_metrics.items() if k != 'feature_density'},
    }, f'{output_dir}/sae_final.pt')
    
    # Save training log
    with open(f'{output_dir}/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Generate plots
    plot_training_curves(training_log, output_dir)
    plot_feature_density(final_metrics['feature_density'], output_dir)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {output_dir}/")
    print(f"  - sae_final.pt (final model)")
    print(f"  - sae_best.pt (best validation loss)")
    print(f"  - training_log.json (metrics)")
    print(f"  - training_curves.png")
    print(f"  - feature_density.png")
    print(f"  - config.json")
    print("=" * 70)
    
    return model, training_log


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train SAE on FLUX activations')
    parser.add_argument('--layer', type=str, default='layer5',
                       choices=['layer5', 'layer15'])
    parser.add_argument('--timestep', type=int, default=14)
    parser.add_argument('--l1_coeff', type=float, default=1e-3)
    parser.add_argument('--expansion', type=int, default=4,
                       help='Expansion factor (hidden_dim = input_dim * expansion)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--resample_every', type=int, default=5000,
                       help='Resample dead features every N steps (0 to disable)')
    parser.add_argument('--normalize', action='store_true',
                       help='Apply per-dimension normalization')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--topk', action='store_true',
                       help='Use TopK SAE instead of L1')
    parser.add_argument('--topk_k', type=int, default=64,
                       help='Number of active features for TopK SAE')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Training SAE for {args.layer}, timestep {args.timestep}")
    if args.topk:
        print(f"Model: TopK SAE (k={args.topk_k})")
    else:
        print(f"Model: L1 SAE (l1_coeff={args.l1_coeff})")
    print(f"Expansion factor: {args.expansion}x")
    print(f"{'='*70}\n")
    
    train_sae(
        layer_name=args.layer,
        timestep=args.timestep,
        l1_coeff=args.l1_coeff,
        expansion_factor=args.expansion,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        resample_dead_every=args.resample_every,
        normalize_data=args.normalize,
        device=args.device,
        use_topk=args.topk,
        topk_k=args.topk_k,
    )