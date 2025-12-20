# make_contact_sheet.py
"""Generate a contact sheet of select steering grids with labels."""

from PIL import Image, ImageDraw, ImageFont

def main():
    # Define grids to include with their descriptions
    grids = [
        {"file": "interp_output_layer5/steering_grids/grid_f3934_d0.png", "feature_id": 3934, "desc": "Edge detection"},
        {"file": "interp_output_layer5/steering_grids/grid_f8259_d1.png", "feature_id": 8259, "desc": "Edge detection"},
        {"file": "interp_output_layer5/steering_grids/grid_f10473_d0.png", "feature_id": 10473, "desc": "Scene openness / Upper background"},
        {"file": "interp_output_layer5/steering_grids/grid_f10473_d1.png", "feature_id": 10473, "desc": "Scene openness / Upper background"},
        {"file": "interp_output_layer5/steering_grids/grid_f3260_d0.png", "feature_id": 3260, "desc": "Vertical edges / tile granularity"},
        {"file": "interp_output_layer5/steering_grids/grid_f9144_d0.png", "feature_id": 9144, "desc": "Vertical light / window edges"},
    ]
    
    # Load images
    images = []
    for g in grids:
        img = Image.open(g["file"])
        images.append({**g, "img": img})
    
    # Get dimensions from first image
    grid_width, grid_height = images[0]["img"].size
    
    # Layout config
    label_height = 50
    padding = 20
    n_grids = len(images)
    
    # Create contact sheet (vertical stack)
    total_height = n_grids * (grid_height + label_height) + (n_grids + 1) * padding
    total_width = grid_width + 2 * padding
    
    sheet = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    
    # Try to load a larger font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    y_offset = padding
    
    for item in images:
        # Draw label
        label = f"Feature {item['feature_id']}: {item['desc']}"
        draw.text((padding, y_offset), label, fill=(0, 0, 0), font=font)
        y_offset += label_height
        
        # Paste grid
        sheet.paste(item["img"], (padding, y_offset))
        y_offset += grid_height + padding
    
    # Save
    output_path = "interp_output_layer5/steering_grids/contact_sheet_layer5.png"
    sheet.save(output_path)
    print(f"Saved contact sheet to {output_path}")
    print(f"Dimensions: {total_width} x {total_height}")


if __name__ == "__main__":
    main()