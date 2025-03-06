import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split

# Define folders
data_folder = "data"
train_folder = "generated images\Train"
test_folder = "generated images\Test"

# List of activities
activities = ["lie down", "fall", "bend", "run", "sitdown", "standup", "walk"]

# Ensure train and test folders exist
for folder in [train_folder, test_folder]:
    os.makedirs(folder, exist_ok=True)
    for activity in activities:
        os.makedirs(os.path.join(folder, activity), exist_ok=True)

# Process each activity folder
for activity in activities:
    activity_path = os.path.join(data_folder, activity)
    
    if not os.path.exists(activity_path):
        print(f"Skipping {activity_path}, folder not found.")
        continue

    image_paths = []

    for filename in os.listdir(activity_path):
        # Ignore annotation files
        if filename.startswith("Annotation_") or not filename.endswith(".csv"):
            continue  

        file_path = os.path.join(activity_path, filename)

        # Load CSV using Pandas, handling text columns if necessary
        try:
            df = pd.read_csv(file_path, delimiter=",")

            # Drop non-numeric columns (e.g., labels if present)
            df = df.select_dtypes(include=[np.number])

            data = df.to_numpy()  # Convert to NumPy array

            # Generate image
            plt.figure(figsize=(6, 6))
            plt.pcolor(data, shading='flat')
            plt.axis('off')

            # Save image
            img_filename = filename.replace(".csv", ".png")
            img_path = os.path.join("generated_images", activity, img_filename)
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            image_paths.append(img_path)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Train-Test Split (80% Train, 20% Test)
    train_imgs, test_imgs = train_test_split(image_paths, test_size=0.2, random_state=42)

    # Move images to respective Train and Test folders
    for img_path in train_imgs:
        shutil.move(img_path, os.path.join(train_folder, activity, os.path.basename(img_path)))

    for img_path in test_imgs:
        shutil.move(img_path, os.path.join(test_folder, activity, os.path.basename(img_path)))

    print(f"Processed {activity}: {len(train_imgs)} train, {len(test_imgs)} test images.")

print("Dataset preparation complete.")
