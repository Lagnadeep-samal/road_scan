import os
import random
import shutil

random.seed(42)

image_dir = "My Dataset/images"

base_dir = "My Dataset"

splits = [
    "train/images",
    "train/labels",
    "valid/images",
    "valid/labels",
    "test/images",
    "test/labels"
]

for folder in splits:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

images = [
    f for f in os.listdir(image_dir)
    if f.endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)

train_size = int(len(images) * 0.7)
valid_size = int(len(images) * 0.2)

train_imgs = images[:train_size]
valid_imgs = images[train_size:train_size+valid_size]
test_imgs = images[train_size+valid_size:]

def copy_files(files, split):

    for img in files:

        txt = os.path.splitext(img)[0] + ".txt"

        shutil.copy(
            os.path.join(image_dir, img),
            os.path.join(base_dir, split, "images", img)
        )

        if os.path.exists(os.path.join(image_dir, txt)):
            shutil.copy(
                os.path.join(image_dir, txt),
                os.path.join(base_dir, split, "labels", txt)
            )

copy_files(train_imgs, "train")
copy_files(valid_imgs, "valid")
copy_files(test_imgs, "test")

print("Dataset Split Completed")