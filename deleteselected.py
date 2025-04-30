import os

# Set folder paths
selection_dir = 'selection_images'
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



# def clean_missing_images():
#     global image_features_dict
#     existing_files = set(os.listdir("illustration_dataset"))
#     keys_to_remove = [key for key in image_features_dict if key not in existing_files]

#     for key in keys_to_remove:
#         del image_features_dict[key]
#         print(f"Removed embedding for missing file: {key}")

# def main():
#     global text_features_dict, image_features_dict

#     model, preprocess = clip.load("ViT-B/32", device="cpu")
#     device = "cpu"

#     image_features_dict = torch.load("new_embeddings.pt", map_location=torch.device('cpu'), weights_only=True)
#     text_features_dict = torch.load("text_embeddings.pt", map_location=torch.device('cpu'), weights_only=True)

#     clean_missing_images()  # 👈 Clean up here

#     torch.save(image_features_dict, "cleaned_new_embeddings.pt")

#     input("Press Enter to exit...")  # Keeps window open