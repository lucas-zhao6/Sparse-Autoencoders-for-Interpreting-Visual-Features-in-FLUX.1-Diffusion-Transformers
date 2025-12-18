# collect_activations.py - DISK-SAVING VERSION
from diffusers import FluxPipeline
import torch
import numpy as np
from tqdm import tqdm
import os

# Create directories - REMOVE generated_images
os.makedirs('visualization_data', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
# os.makedirs('generated_images', exist_ok=True)  # COMMENTED OUT

# Load model
print("Loading FLUX.1-dev...")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

# Prepare hooks for TWO layers
activations_layer5 = []
activations_layer15 = []

def get_activation_hook_layer5(name):
    def hook(module, input, output):
        activations_layer5.append(output.detach().cpu())
    return hook

def get_activation_hook_layer15(name):
    def hook(module, input, output):
        activations_layer15.append(output.detach().cpu())
    return hook

# Hook into early (5) and late (15) dual-stream blocks
hook_layer5 = pipe.transformer.transformer_blocks[5].ff.register_forward_hook(
    get_activation_hook_layer5('dual_block_5_ffn')
)
hook_layer15 = pipe.transformer.transformer_blocks[15].ff.register_forward_hook(
    get_activation_hook_layer15('dual_block_15_ffn')
)

# Load prompts from file
print("Loading prompts from prompts.txt...")
with open('prompts.txt', 'r') as f:
    prompts = [line.strip() for line in f.readlines()]

print(f"Loaded {len(prompts)} prompts")

# Define special prompts for visualization
viz_prompts = {
    "dog_bed": "a black dog is sitting in his dog bed",
    "dog_beach": "A small black dog sitting on top of a sandy beach",
    "elephants": "Elephants being bathed in a body of water by people in sun hats."
}

# Open memory-mapped files for incremental writing
print("Creating memory-mapped arrays for incremental saving...")
total_samples = len(prompts) * 81920  # 500 * 81920 = 40,960,000
mmap_layer5 = np.memmap(
    'flux_activations_layer5.npy.tmp', 
    dtype='float32', 
    mode='w+', 
    shape=(total_samples, 3072)
)
mmap_layer15 = np.memmap(
    'flux_activations_layer15.npy.tmp', 
    dtype='float32', 
    mode='w+', 
    shape=(total_samples, 3072)
)

current_idx = 0  # Track position in memory-mapped array

print(f"\nCollecting activations from {len(prompts)} images...")
print("Hooking into dual-stream layers 5 (early) and 15 (late)")
print("Writing incrementally to disk to avoid OOM...")
print("SKIPPING IMAGE SAVING to conserve disk space\n")

for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
    activations_layer5 = []  # Clear for this image
    activations_layer15 = []
    
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            num_inference_steps=20,
            generator=torch.Generator("cuda").manual_seed(42 + i),
            output_type="pil"
        )
    
    # SKIP IMAGE SAVING - just generate to get activations
    # image = output.images[0]
    
    # Save ONLY visualization activations (small files)
    if prompt == viz_prompts["dog_bed"]:
        # image.save('generated_images/dog_bed.png')  # SKIP
        act_t10_l5 = activations_layer5[9].float().numpy()
        np.save('visualization_data/dog_bed_timestep_10_layer5.npy', act_t10_l5)
        act_t10_l15 = activations_layer15[9].float().numpy()
        np.save('visualization_data/dog_bed_timestep_10_layer15.npy', act_t10_l15)
        tqdm.write("✓ Saved dog_bed activations")
    
    if prompt == viz_prompts["dog_beach"]:
        # image.save('generated_images/dog_beach.png')  # SKIP
        act_t10_l5 = activations_layer5[9].float().numpy()
        np.save('visualization_data/dog_beach_timestep_10_layer5.npy', act_t10_l5)
        act_t10_l15 = activations_layer15[9].float().numpy()
        np.save('visualization_data/dog_beach_timestep_10_layer15.npy', act_t10_l15)
        tqdm.write("✓ Saved dog_beach activations")
    
    if prompt == viz_prompts["elephants"]:
        # image.save('generated_images/elephants.png')  # SKIP
        for timestep_idx, timestep_name in [(4, 5), (9, 10), (14, 15), (19, 20)]:
            act_l5 = activations_layer5[timestep_idx].float().numpy()
            np.save(f'visualization_data/elephants_timestep_{timestep_name}_layer5.npy', act_l5)
            act_l15 = activations_layer15[timestep_idx].float().numpy()
            np.save(f'visualization_data/elephants_timestep_{timestep_name}_layer15.npy', act_l15)
        tqdm.write("✓ Saved elephants activations")
    
    # SKIP checkpoint images to save space
    
    # Process activations and write directly to memory-mapped file
    img_acts_l5 = torch.cat(activations_layer5, dim=0)  # [20, 4096, 3072]
    img_acts_l5 = img_acts_l5.reshape(-1, 3072).float().numpy()  # [81920, 3072]
    
    img_acts_l15 = torch.cat(activations_layer15, dim=0)
    img_acts_l15 = img_acts_l15.reshape(-1, 3072).float().numpy()
    
    # Write to disk incrementally
    mmap_layer5[current_idx:current_idx + 81920] = img_acts_l5
    mmap_layer15[current_idx:current_idx + 81920] = img_acts_l15
    current_idx += 81920
    
    # Flush to disk periodically
    if (i + 1) % 50 == 0:
        mmap_layer5.flush()
        mmap_layer15.flush()
        tqdm.write(f"Flushed to disk at {i+1} images")

# Final flush
print("\nFlushing final data to disk...")
mmap_layer5.flush()
mmap_layer15.flush()

# Convert to proper numpy format
print("Converting memory-mapped files to numpy arrays...")
del mmap_layer5
del mmap_layer15

# Rename tmp files
os.rename('flux_activations_layer5.npy.tmp', 'flux_activations_layer5_raw.dat')
os.rename('flux_activations_layer15.npy.tmp', 'flux_activations_layer15_raw.dat')

print("Creating proper .npy files...")
layer5_data = np.memmap('flux_activations_layer5_raw.dat', dtype='float32', mode='r', shape=(total_samples, 3072))
layer15_data = np.memmap('flux_activations_layer15_raw.dat', dtype='float32', mode='r', shape=(total_samples, 3072))

np.save('flux_activations_layer5.npy', layer5_data)
np.save('flux_activations_layer15.npy', layer15_data)

print(f"Layer 5 dataset shape: {layer5_data.shape}")
print(f"Layer 15 dataset shape: {layer15_data.shape}")

# Remove hooks
hook_layer5.remove()
hook_layer15.remove()

# Clean up raw data files
os.remove('flux_activations_layer5_raw.dat')
os.remove('flux_activations_layer15_raw.dat')

print("\n" + "="*60)
print("COLLECTION COMPLETE!")
print("="*60)
print(f"Layer 5 dataset: flux_activations_layer5.npy")
print(f"Layer 15 dataset: flux_activations_layer15.npy")
print(f"Visualization data saved in: visualization_data/")
print("="*60)