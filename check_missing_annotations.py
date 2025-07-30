# Script to check which images are missing from annotation file
import json
from pathlib import Path

# Paths (edit if needed)
image_folder = Path('CR7')
annotation_file = Path('auto_labels/auto_annotations.json')

# Load annotation file
with open(annotation_file, 'r', encoding='utf-8') as f:
    annotations = json.load(f)

# Get all image files (support .jpg, .png, .JPG, .PNG)
image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png')) + \
              list(image_folder.glob('*.JPG')) + list(image_folder.glob('*.PNG'))

# Get annotation keys
annotated_images = set(annotations.keys())

# Find missing images
missing = []
for img_path in image_files:
    if img_path.name not in annotated_images:
        missing.append(img_path.name)

print(f"Total images: {len(image_files)}")
print(f"Annotated: {len(annotated_images)}")
print(f"Missing: {len(missing)}")
if missing:
    print("Images missing from annotation file:")
    for name in missing:
        print(name)
else:
    print("All images are annotated.")
