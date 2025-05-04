import torch
import clip
import os
from tqdm import tqdm
from PIL import Image

def extract_image_features(image_path, device, model, preprocess): 
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features / image_features.norm(dim=-1, keepdim=True)


def embed_image(device, model, preprocess, image_features_dict):
    # Load images and compute embeddings
    image_folder = "images_dataset"
    # embedding_file = "backup_image_embeddings.pt"


    # if os.path.exists(embedding_file):
    #     print(f"Loading embeddings from '{embedding_file}'...")
    #     image_features_dict = torch.load(embedding_file, map_location=device)
    #     print(f"Loaded {len(image_features_dict)} embeddings from '{embedding_file}'.")
    # else:
    #     image_features_dict = {}
    #     print(f"No existing embeddings found. Starting from scratch.")

    image_features_dict = {}
    print(f"No existing embeddings found. Starting from scratch.")

    print("Scanning for image files in the folder...")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(("png", "jpg", "jpeg"))]
    print(f"Found {len(image_files)} images in '{image_folder}'.")

    print("Scanning for new images...")
    new_images = [f for f in image_files if f not in image_features_dict]
    print(f"Found {len(new_images)} new images to process.")

    if not new_images:
        print("No new images to process.")
    else:
        for filename in tqdm(new_images, desc="Processing new images", unit="image"):
            image_path = os.path.join(image_folder, filename)
            image_features_dict[filename] = extract_image_features(image_path, device, model, preprocess)

        # Step 5: Save image embeddings to a file
        print("Saving updated embeddings to 'new_embeddings.pt'...")
        torch.save(image_features_dict, "new_embeddings.pt")
        print("Image embeddings saved to 'new_embeddings.pt'.")

# Main function
def main():
    global model, text_features_dict, image_features_dict

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    print(torch.cuda.is_available())  # Should return True if 
    print(torch.cuda.device_count())  # Number of available 

    # Load image embeddings
    # if os.path.exists("image_embeddings.pt"):
    #     image_features_dict = torch.load("image_embeddings.pt", map_location=torch.device(device))
    image_features_dict = {}
    embed_image(device, model, preprocess, image_features_dict)

    # # Start the app
    # app = QApplication(sys.argv)
    # window = MainWindow()
    # window.show()
    # sys.exit(app.exec_())

# Entry point
if __name__ == "__main__":
    main()