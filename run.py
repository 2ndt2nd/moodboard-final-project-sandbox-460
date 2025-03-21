import torch
import clip
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display
import pyclip
import io
import win32clipboard
import tkinter as tk
from PIL import Image, ImageTk

import shutil
import webbrowser

image_folder = "illustration_dataset"  # Change to your actual image folder
selected_images = {}


def extract_text_features(text):
    status = widgets.Label(value="Ready")
    display(status)

    status.value = f"Processing: {text}"  # Update status
    
    text_tokenized = clip.tokenize([text]).to(device)

    status.value = "Encoding text with CLIP..."
    with torch.no_grad():
        text_features = model.encode_text(text_tokenized)
    
    status.value = "Normalizing features..."
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    status.value = "Extraction complete!"
    return text_features


def match_query(input_query):
    # Initialize status widget
    status = widgets.Label(value="Ready")
    display(status)

    # Example query
    query = input_query

    # Get the text features for the query
    if query in text_features_dict:
        text_features = text_features_dict[query]
        status.value = f"Found cached features for: {query}"
    else:
        status.value = f"Computing features for: {query}"
        text_features = extract_text_features(query)  # Compute features if not cached
        status.value = f"Features extracted for: {query}"

    # Compute cosine similarity with a progress bar
    status.value = "Computing cosine similarity..."
    similarities = {}
    for img_name, img_features in tqdm(image_features_dict.items(), desc="Computing Similarity"):
        similarities[img_name] = torch.cosine_similarity(text_features, img_features, dim=-1).item()

    # Sort results by similarity
    sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Display top matches
    status.value = "Displaying top matches..."
    for img, score in sorted_images[:8]:
        print(f"{img}: {score:.4f}")

    status.value = "Completed!"

    return sorted_images


# UI Declaration
# Function to create the "Start" screen
def create_start_screen():
    global start_prompts  # Declare the global variable to store user input

    # Create the window
    start_window = tk.Tk()
    start_window.title("Welcome")

    # Add a label for the title
    title_label = tk.Label(start_window, text="Welcome to the Image Grid!", font=("Helvetica", 16))
    title_label.pack(pady=20)

    # Create a text box for the user to enter some text
    input_label = tk.Label(start_window, text="Enter your prompt:")
    input_label.pack(pady=5)
    
    # Textbox for the user to input text
    input_box = tk.Entry(start_window, width=30)
    input_box.pack(pady=10)

    # Create a start button that transitions to the image grid
    start_button = tk.Button(start_window, text="Start", command=lambda: start_button_clicked(start_window, input_box.get()))
    start_button.pack(pady=10)

    # Run the start screen
    start_window.mainloop()

# Function to create the image grid UI
def create_image_grid(match_results):
    global selected_images  # Make sure we use the global dictionary
    top_k = 10  # Number of images to display
    top_n = 40  # Consider the top 50 instead of just top_k
    random_subset = random.sample(match_results[:top_n], top_k)  # Pick top_k randomly
    image_files = [img for img, score in random_subset[:10]]  # Get only image names

    # Initialize the Tkinter window for the image grid
    root = tk.Tk()
    root.title("Image Grid")

    # Frame to hold the image thumbnails
    image_frame = tk.Frame(root)
    image_frame.pack(padx=10, pady=10)

    # Clear any existing widgets from the image_frame before re-adding
    for widget in image_frame.winfo_children():
        widget.destroy()

    # Create a folder to store copied images if it doesn't exist
    if not os.path.exists('copied_images'):
        os.makedirs('copied_images')

    # Create the "Copy!" button
    copy_button = tk.Button(root, text="Copy!", command=copy_selected_images)
    copy_button.pack(pady=10)

    # Create the "Shuffle!" button to load a new set of images
    shuffle_button = tk.Button(root, text="Shuffle!", command=shuffle_images)
    shuffle_button.pack(pady=10)

    # Store image references to prevent garbage collection
    img_references = {}

    # Loop through the image files and add them to the grid
    for idx, img_file in enumerate(image_files):
        # Load the image
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)

        # Resize the image to fit the grid
        image.thumbnail((150, 150))  # Resize image to fit in the UI

        # Convert image to Tkinter-compatible format
        img_tk = ImageTk.PhotoImage(image)

        # Store the reference to prevent garbage collection
        img_references[img_file] = img_tk

        # Create a label to display the image
        label = tk.Label(image_frame, image=img_tk, bg='red', bd=1)  # Set initial background to highlight, thinner border
        label.image = img_tk  # Keep a reference to the image to prevent garbage collection

        # Bind the click event to the label (make the image clickable)
        label.bind("<Button-1>", lambda event, name=img_file, label=label, img_tk=img_tk: on_image_click(name, label, img_tk))

        # Place the label in the grid
        label.grid(row=idx // 4, column=idx % 4, padx=10, pady=10)  # Adjust grid size

    # Start the Tkinter event loop to display the window
    root.mainloop()

# Function to shuffle the images (load a new set)
def shuffle_images():
    create_image_grid(start_prompts)  # Refresh image grid with a new set of images

# Function to handle image click (toggle selection)
def on_image_click(img_name, label, img_tk):
    if img_name in selected_images:
        # Image is selected, deselect it
        del selected_images[img_name]
        label.config(bg='red', bd=2)  # Reset background color to unhighlighted, thinner border
    else:
        # Image is not selected, select it
        selected_images[img_name] = True
        label.config(bg='blue', bd=5)  # Set background color to highlight it, thicker border

# Function to handle the start button click
def start_button_clicked(start_window, boxInput):
    print(f"Start prompt: {start_prompts}")  # For debugging, prints the stored input
    match_results = match_query(boxInput)
    if match_results is not None:
        start_window.destroy()  # Close the start window
        create_image_grid(match_results)  # Transition to the image grid

# Function to copy selected images to the 'copied_images' folder
def copy_selected_images():
    if not selected_images:
        print("No images selected.")
        return

    # Create the "copied_images" folder if it doesn't exist
    if not os.path.exists('copied_images'):
        os.makedirs('copied_images')

    # Loop through the selected images and copy them
    for img_name in selected_images:
        image_path = os.path.join(image_folder, img_name)
        if os.path.exists(image_path):
            dest_path = os.path.join('copied_images', img_name)
            shutil.copy(image_path, dest_path)  # Copy image to the 'copied_images' folder
            print(f"Copied {img_name} to 'copied_images'.")
            absolute_path = os.path.abspath("copied_images")
            webbrowser.open(absolute_path)
        else:
            print(f"Image {img_name} not found.")


def main():
    global model, text_features_dict, image_features_dict  # Make these global
    model, preprocess = clip.load("ViT-B/32", device="cpu")

    # Load image embeddings
    image_features_dict = torch.load("image_embeddings.pt", map_location=torch.device('cpu'))

    # Load text embeddings
    text_features_dict = torch.load("text_embeddings.pt", map_location=torch.device('cpu'))

    #Start the app
    selected_images = {}
    image_folder = "illustration_dataset"  # Change to your actual image folder

    create_start_screen()

if __name__ == "__main__":
    main()