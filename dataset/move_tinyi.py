import os
import shutil



data_dir = "../dataset/tiny-imagenet-200"
val_dir = os.path.join(data_dir, "val")
val_images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")
val_output_dir = os.path.join(val_dir, "organized_val")

# Create the new directory structure
os.makedirs(val_output_dir, exist_ok=True)

# Read validation annotations
with open(annotations_file, "r") as f:
    lines = f.readlines()

# Process each image
for line in lines:
    parts = line.strip().split("\t")
    filename, class_id = parts[0], parts[1]
    
    # Create class subdirectory if not exists
    class_dir = os.path.join(val_output_dir, class_id)
    os.makedirs(class_dir, exist_ok=True)
    
    # Move image to corresponding class folder
    src = os.path.join(val_images_dir, filename)
    dst = os.path.join(class_dir, filename)
    shutil.move(src, dst)

print("Validation data organized successfully!")