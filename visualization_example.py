# visualize_elephants_timesteps.py
import numpy as np
import matplotlib.pyplot as plt

def visualize_elephants_evolution(layer='layer5'):
    """
    Visualize elephant activations across timesteps 5, 10, 15, 20
    
    Args:
        layer: 'layer5' or 'layer15'
    """
    
    timesteps = [5, 10, 15, 20]
    activations = {}
    
    # Load activations
    print(f"Loading elephants activations for {layer}...")
    for t in timesteps:
        filepath = f'visualization_data/elephants_timestep_{t}_{layer}.npy'
        act = np.load(filepath)  # [1, 4096, 3072]
        act = act.squeeze(0)  # [4096, 3072]
        
        # Aggregate across feature dimension (L2 norm)
        act_magnitude = np.linalg.norm(act, axis=1)  # [4096]
        
        # Reshape to 2D spatial grid (64x64)
        act_2d = act_magnitude.reshape(64, 64)
        
        activations[t] = act_2d
        print(f"  Loaded timestep {t}: shape {act_2d.shape}")
    
    # Find global min/max for consistent color scale
    vmin = min(act.min() for act in activations.values())
    vmax = max(act.max() for act in activations.values())
    
    # Create figure with extra space for colorbar
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    layer_title = "Early (Layer 5)" if layer == 'layer5' else "Late (Layer 15)"
    fig.suptitle(f'Elephants Being Bathed - Activation Evolution\n{layer_title}', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Plot each timestep
    images = []
    for idx, t in enumerate(timesteps):
        im = axes[idx].imshow(activations[t], cmap='viridis', aspect='auto', 
                              vmin=vmin, vmax=vmax)
        axes[idx].set_title(f'Timestep {t}/20', fontsize=14, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Spatial X', fontsize=11)
        axes[idx].set_ylabel('Spatial Y', fontsize=11)
        
        # Add statistics as text
        mean_val = activations[t].mean()
        max_val = activations[t].max()
        axes[idx].text(0.02, 0.98, f'Mean: {mean_val:.3f}\nMax: {max_val:.3f}', 
                       transform=axes[idx].transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        images.append(im)
    
    # Adjust layout to make room for colorbar
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])  # Leave space on right for colorbar
    
    # Add single colorbar on the right side
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(images[0], cax=cbar_ax, label='Activation Magnitude')
    cbar.ax.tick_params(labelsize=10)
    
    # Save figure
    output_path = f'elephants_timestep_evolution_{layer}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    # Print statistics
    print("\nActivation Statistics Across Timesteps:")
    print("-" * 50)
    for t in timesteps:
        act = activations[t]
        print(f"Timestep {t:2d}: Mean={act.mean():.3f}, Std={act.std():.3f}, "
              f"Max={act.max():.3f}, Min={act.min():.3f}")
    
    plt.close()
    return fig

if __name__ == "__main__":
    print("="*60)
    print("VISUALIZING ELEPHANTS TIMESTEP EVOLUTION")
    print("="*60)
    
    # Generate for both layers
    print("\n[1/2] Layer 5 (Early)...")
    visualize_elephants_evolution(layer='layer5')
    
    print("\n[2/2] Layer 15 (Late)...")
    visualize_elephants_evolution(layer='layer15')
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("Generated:")
    print("  - elephants_timestep_evolution_layer5.png")
    print("  - elephants_timestep_evolution_layer15.png")
    print("="*60)