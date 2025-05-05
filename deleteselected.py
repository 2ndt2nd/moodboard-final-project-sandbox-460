import os

# Set folder paths
selection_dir = 'target_delete'
illustration_dir = 'images_dataset'

# Get all image filenames in selection_images
selection_filenames = set(os.listdir(selection_dir))

# Loop through each file in illustration_dataset
for filename in selection_filenames:
    illustration_path = os.path.join(illustration_dir, filename)
    if os.path.exists(illustration_path):
        os.remove(illustration_path)
        print(f"Deleted: {filename}")
    else:
        print(f"Not found in illustration_dataset: {filename}")

input("Press Enter to exit...")  # Keeps window open