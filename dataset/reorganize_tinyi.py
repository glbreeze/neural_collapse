import os
import shutil

train_dir = "../dataset/tiny-imagenet-200/train"

for class_folder in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_folder)
    images_path = os.path.join(class_path, "images")

    if os.path.exists(images_path):
        for img_file in os.listdir(images_path):
            shutil.move(os.path.join(images_path, img_file), class_path)
        os.rmdir(images_path)  # Remove empty images/ folder

print("Training set fixed!")
