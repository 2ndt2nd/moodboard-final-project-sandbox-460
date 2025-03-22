import torch
import clip
import os
import random
from tqdm import tqdm  # Use standard tqdm for console progress bar
import ipywidgets as widgets
from IPython.display import display
import tkinter as tk
from PIL import Image, ImageTk
import shutil
import webbrowser
import threading

# Global variables
image_folder = "illustration_dataset"  # Change to your actual image folder
last_match_results = []
text_features_dict = {}
image_features_dict = {}
model = None
device = "cpu"

# Global dictionary to store sorted results for each prompt
prompt_results_cache = {}

# Function to extract text features
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

# Function to match a query
def match_query(input_query):
    global prompt_results_cache

    # Check if results are already cached
    if input_query in prompt_results_cache:
        print("Using cached results for:", input_query)
        return prompt_results_cache[input_query]

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

    # Function to compute similarities in the background
    def compute_similarities():
        status.value = "Computing cosine similarity..."
        similarities = {}

        # Use tqdm with mininterval to reduce overhead
        for img_name, img_features in tqdm(image_features_dict.items(), desc="Computing Similarity", mininterval=0.5):
            similarities[img_name] = torch.cosine_similarity(text_features, img_features, dim=-1).item()

        # Sort results by similarity
        sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Display top matches
        status.value = "Displaying top matches..."
        for img, score in sorted_images[:8]:
            print(f"{img}: {score:.4f}")

        status.value = "Completed!"

        # Cache the sorted results for this prompt
        prompt_results_cache[input_query] = sorted_images
        return sorted_images

    # Run the similarity computation in a separate thread
    thread = threading.Thread(target=compute_similarities)
    thread.start()

    # Return the match results
    return compute_similarities()

# Function to create the start screen
def create_start_screen():
    # Create the window
    start_window = tk.Tk()
    start_window.title("Welcome")

    # Add a label for the title
    title_label = tk.Label(start_window, text="Start Your Moodboard!", font=("Comic Sans", 20))
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

    # Start the Tkinter event loop
    start_window.mainloop()

# Function to handle the start button click
def start_button_clicked(start_window, input_text):
    # Create a new Toplevel window for the image grid
    image_grid_window = tk.Toplevel(start_window)
    image_grid_window.title("Image Grid")

    # Add a label for the grid
    grid_label = tk.Label(image_grid_window, text=input_text, font=("Comic Sans", 20))
    grid_label.pack(pady=20)

    # Store the original prompt for this window
    image_grid_window.original_prompt = input_text

    # Initialize selected images for this window
    image_grid_window.selected_images = {}

    # Call the function to create the image grid in the new window
    match_results = match_query(input_text)
    create_image_grid(input_text, match_results, image_grid_window)

# # Function to create the image grid
# def create_image_grid(input_text, match_results, parent_window):
#     global last_match_results

#     last_match_results = match_results  # Store latest results for shuffling
#     top_k = 15
#     top_n = 50
#     random_subset = random.sample(last_match_results[:top_n], top_k)
#     image_files = [img for img, score in random_subset]

#     # Use the parent_window (Toplevel) to create the image grid
#     image_frame = tk.Frame(parent_window)
#     image_frame.pack(padx=10, pady=10)

#     # Store image labels for this window
#     parent_window.image_labels = []

#     for idx, img_file in enumerate(image_files):
#         image_path = os.path.join(image_folder, img_file)
#         image = Image.open(image_path)

#         image.thumbnail((200, 200))
#         img_tk = ImageTk.PhotoImage(image)

#         label = tk.Label(image_frame, image=img_tk, bg='red', bd=1)
#         label.image = img_tk
#         label.grid(row=idx % 3, column=idx // 3, padx=10, pady=10)

#         # Bind the click event to the label (make the image clickable)
#         label.bind(
#             "<Button-1>",
#             lambda event, name=img_file, label=label, img_tk=img_tk: on_image_click(name, label, img_tk, parent_window)
#         )

#         parent_window.image_labels.append(label)  # Store labels for this window

#     # Add Shuffle button
#     shuffle_button = tk.Button(parent_window, text="Shuffle!", command=lambda: shuffle_images(parent_window))
#     shuffle_button.pack(pady=10)

#     # Create the "Copy!" button
#     copy_button = tk.Button(parent_window, text="Copy!", command=lambda: copy_selected_images(parent_window))
#     copy_button.pack(pady=10)

def create_image_grid(input_text, match_results, parent_window):
    global last_match_results

    last_match_results = match_results  # Store latest results for shuffling
    top_k = 15  # Total number of images to display
    top_n = 50
    random_subset = random.sample(last_match_results[:top_n], top_k)
    image_files = [img for img, score in random_subset]

    # Define preferred grid layout (5 columns, 3 rows)
    preferred_columns = 5
    min_columns = 3  # Minimum number of columns for small screens
    column_width = 220  # Approximate width of each image + padding

    # Set a threshold width for switching to 3 columns (e.g., 800 pixels for an 11-inch tablet)
    screen_width_threshold = 800

    # Calculate the number of columns based on the screen width
    screen_width = parent_window.winfo_screenwidth()
    num_columns = preferred_columns if screen_width >= screen_width_threshold else min_columns

    # Ensure the number of columns does not exceed the number of images
    num_columns = min(num_columns, len(image_files))

    # Calculate the required window width
    window_width = num_columns * column_width
    window_height = 800  # Fixed height for the window

    # Set the window size
    parent_window.geometry(f"{window_width}x{window_height}")

    # Create a Canvas with a vertical Scrollbar
    canvas = tk.Canvas(parent_window)
    v_scrollbar = tk.Scrollbar(parent_window, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=v_scrollbar.set)

    # Pack the Canvas and Scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    v_scrollbar.pack(side="right", fill="y")

    # Create a Frame inside the Canvas to hold the images
    image_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=image_frame, anchor="nw")

    # Update the scroll region when the Frame size changes
    def update_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    image_frame.bind("<Configure>", update_scroll_region)

    # Store image labels for this window
    parent_window.image_labels = []

    # Display images in the grid
    for idx, img_file in enumerate(image_files):
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)

        image.thumbnail((200, 200))  # Thumbnail size remains fixed
        img_tk = ImageTk.PhotoImage(image)

        label = tk.Label(image_frame, image=img_tk, bg='red', bd=1)
        label.image = img_tk
        label.grid(row=idx // num_columns, column=idx % num_columns, padx=10, pady=10)

        # Bind the click event to the label
        label.bind(
            "<Button-1>",
            lambda event, name=img_file, label=label, img_tk=img_tk: on_image_click(name, label, img_tk, parent_window)
        )

        parent_window.image_labels.append(label)

    # Allow columns to expand
    for col in range(num_columns):
        image_frame.grid_columnconfigure(col, weight=1)

    # Add Shuffle button
    shuffle_button = tk.Button(parent_window, text="Shuffle!", command=lambda: shuffle_images(parent_window))
    shuffle_button.pack(pady=10)

    # Create the "Copy!" button
    copy_button = tk.Button(parent_window, text="Copy!", command=lambda: copy_selected_images(parent_window))
    copy_button.pack(pady=10)
# Function to shuffle images
def shuffle_images(window):
    if not hasattr(window, 'original_prompt'):
        print("No original prompt found for this window.")
        return

    # Clear the selected_images dictionary
    window.selected_images.clear()

    # Use the original prompt for this window
    original_prompt = window.original_prompt

    # Retrieve cached results for this prompt
    if original_prompt not in prompt_results_cache:
        print("No cached results found for this prompt.")
        return

    sorted_images = prompt_results_cache[original_prompt]

    # Pick a new random subset of images
    top_k = 15
    top_n = 50
    random_subset = random.sample(sorted_images[:top_n], top_k)
    image_files = [img for img, score in random_subset]

    # Update the existing image labels with new images
    for idx, img_file in enumerate(image_files):
        image_path = os.path.join(image_folder, img_file)
        image = Image.open(image_path)

        image.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(image)

        # Update the existing labels instead of recreating them
        window.image_labels[idx].config(image=img_tk, bg='red', bd=2)
        window.image_labels[idx].image = img_tk  # Keep a reference to prevent garbage collection

        # Rebind the click event to the new image
        window.image_labels[idx].bind(
            "<Button-1>",
            lambda event, name=img_file, label=window.image_labels[idx], img_tk=img_tk: on_image_click(name, label, img_tk, window)
        )

# Function to handle image click (toggle selection)
def on_image_click(img_name, label, img_tk, window):
    if img_name in window.selected_images:
        # Image is selected, deselect it
        del window.selected_images[img_name]
        label.config(bg='red', bd=1)  # Reset background color to unhighlighted, thinner border
    else:
        # Image is not selected, select it
        window.selected_images[img_name] = True
        label.config(bg='blue', bd=5)  # Set background color to highlight it, thicker border

# Function to copy selected images to the prompt title folder
def copy_selected_images(input_text, window):
    if not window.selected_images:
        print("No images selected.")
        return

    # Create the "copied_images" folder if it doesn't exist
    if not os.path.exists(input_text):
        os.makedirs(input_text)

    # Loop through the selected images and copy them
    for img_name in window.selected_images:
        image_path = os.path.join(image_folder, img_name)
        if os.path.exists(image_path):
            dest_path = os.path.join(input_text, img_name)
            shutil.copy(image_path, dest_path)  # Copy image to the 'copied_images' folder
            print(f"Copied {img_name} to {input_text}.")
            absolute_path = os.path.abspath(input_text)
            webbrowser.open(absolute_path)
        else:
            print(f"Image {img_name} not found.")

# Function to copy selected images to the 'copied_images' folder
def copy_selected_images(window):
    if not window.selected_images:
        print("No images selected.")
        return

    # Create the "copied_images" folder if it doesn't exist
    if not os.path.exists('copied_images'):
        os.makedirs('copied_images')

    # Loop through the selected images and copy them
    for img_name in window.selected_images:
        image_path = os.path.join(image_folder, img_name)
        if os.path.exists(image_path):
            dest_path = os.path.join('copied_images', img_name)
            shutil.copy(image_path, dest_path)  # Copy image to the 'copied_images' folder
            print(f"Copied {img_name} to 'copied_images'.")
            absolute_path = os.path.abspath("copied_images")
            webbrowser.open(absolute_path)
        else:
            print(f"Image {img_name} not found.")

# Main function
def main():
    global model, text_features_dict, image_features_dict, device

    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    device = "cpu"

    # Load image embeddings
    image_features_dict = torch.load("image_embeddings.pt", map_location=torch.device('cpu'))

    # Load text embeddings
    text_features_dict = torch.load("text_embeddings.pt", map_location=torch.device('cpu'))

    # Start the app
    create_start_screen()

# Entry point
if __name__ == "__main__":
    main()