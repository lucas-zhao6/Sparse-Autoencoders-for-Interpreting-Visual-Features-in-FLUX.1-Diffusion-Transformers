from datasets import load_dataset
import random

print("Loading COCO captions...")
dataset = load_dataset("yerevann/coco-karpathy", split="train")

# Extract captions - 'sentences' is a list of captions for each image
all_captions = []
for item in dataset:
    all_captions.extend(item['sentences'])  # Each item has a list of 5 captions

print(f"Found {len(all_captions)} captions")

# Sample 500 diverse prompts
random.seed(42)
prompts = random.sample(all_captions, min(500, len(all_captions)))

# Clean prompts
prompts = [p.strip() for p in prompts if p and len(p.strip()) > 10]

# Save
with open('prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + '\n')

print(f"Saved {len(prompts)} prompts to prompts_500.txt")

# Print first 10
print("\nFirst 10 prompts:")
for i, p in enumerate(prompts[:10], 1):
    print(f"{i}. {p}")