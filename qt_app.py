import sys
import os
import random
import shutil
import webbrowser
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QGridLayout, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
import clip
from tqdm import tqdm

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
    text_tokenized = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# Function to match a query
def match_query(input_query):
    global prompt_results_cache

    # Check if results are already cached
    if input_query in prompt_results_cache:
        print("Using cached results for:", input_query)
        return prompt_results_cache[input_query]

    # Get the text features for the query
    if input_query in text_features_dict:
        text_features = text_features_dict[input_query]
    else:
        text_features = extract_text_features(input_query)

    # Function to compute similarities in the background
    def compute_similarities():
        similarities = {}
        for img_name, img_features in tqdm(image_features_dict.items(), desc="Computing Similarity", mininterval=0.5):
            similarities[img_name] = torch.cosine_similarity(text_features, img_features, dim=-1).item()

        # Sort results by similarity
        sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Cache the sorted results for this prompt
        prompt_results_cache[input_query] = sorted_images
        return sorted_images

    # Run the similarity computation in a separate thread
    thread = threading.Thread(target=compute_similarities)
    thread.start()

    # Wait for the thread to complete and return the results
    thread.join()  # Wait for the thread to finish
    return prompt_results_cache[input_query]  # Return the cached results

# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome")
        self.setGeometry(100, 100, 400, 200)

        # Create the main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Add a label for the title
        self.title_label = QLabel("Start Your Moodboard!")
        self.title_label.setStyleSheet("font-size: 30px; font-weight: bold; font-family: Arial;")
        self.layout.addWidget(self.title_label)

        # Add a label and text box for the user to enter some text
        self.input_label = QLabel("Enter your prompt:")
        self.layout.addWidget(self.input_label)

        self.input_box = QLineEdit()
        self.layout.addWidget(self.input_box)

        # Add a start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_button_clicked)
        self.layout.addWidget(self.start_button)

    def start_button_clicked(self):
        input_text = self.input_box.text()
        if not input_text:
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return

        # Create a new window for the image grid
        self.image_grid_window = ImageGridWindow(input_text)
        self.image_grid_window.show()

# Image grid window class
class ImageGridWindow(QMainWindow):
    def __init__(self, input_text):
        super().__init__()
        self.setWindowTitle("Image Grid")
        self.setGeometry(100, 50, 900, 700)

        self.input_text = input_text
        self.selected_images = {}

        # Create the main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Add a label for the grid
        self.grid_label = QLabel(input_text)
        self.grid_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.layout.addWidget(self.grid_label)

        # Create a scroll area for the image grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        # Create a widget for the scroll area
        self.scroll_widget = QWidget()
        self.scroll_area.setWidget(self.scroll_widget)
        self.grid_layout = QGridLayout(self.scroll_widget)

        # Add images to the grid
        self.match_results = match_query(input_text)
        self.create_image_grid()

        # Add Shuffle button
        self.shuffle_button = QPushButton("Shuffle!")
        self.shuffle_button.clicked.connect(self.shuffle_images)
        self.layout.addWidget(self.shuffle_button)

        # Add Copy button
        self.copy_button = QPushButton("Copy!")
        self.copy_button.clicked.connect(self.copy_selected_images)
        self.layout.addWidget(self.copy_button)

    def create_image_grid(self):
        top_k = 16
        top_n = 50
        random_subset = random.sample(self.match_results[:top_n], top_k)
        image_files = [img for img, score in random_subset]

        num_columns = 4  
        image_size = 250  
        padding = 10 
        window_width = num_columns * (image_size + padding)
        screen_geometry = QApplication.desktop().screenGeometry()
        window_height = screen_geometry.height()-100
        self.resize(window_width, window_height)

        for idx, img_file in enumerate(image_files):
            image_path = os.path.join(image_folder, img_file)
            image = QImage(image_path)
            pixmap = QPixmap.fromImage(image).scaled(image_size, image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = QLabel()
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 2px solid red;")
            label.mousePressEvent = lambda event, name=img_file, label=label: self.on_image_click(name, label)

            # Calculate row and column based on the number of columns
            row = idx // num_columns
            column = idx % num_columns
            self.grid_layout.addWidget(label, row, column, alignment=Qt.AlignCenter)

    def shuffle_images(self):
        top_k = 16
        top_n = 50
        random_subset = random.sample(self.match_results[:top_n], top_k)
        image_files = [img for img, score in random_subset]
        image_size = 250

        for idx, img_file in enumerate(image_files):
            image_path = os.path.join(image_folder, img_file)
            image = QImage(image_path)
            pixmap = QPixmap.fromImage(image).scaled(image_size, image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = self.grid_layout.itemAt(idx).widget()
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 0px solid transparent;")
            label.mousePressEvent = lambda event, name=img_file, label=label: self.on_image_click(name, label)

    def on_image_click(self, img_name, label):
        if img_name in self.selected_images:
            del self.selected_images[img_name]
            label.setStyleSheet("border: 0px solid transparent;")
        else:
            self.selected_images[img_name] = True
            label.setStyleSheet("border: 5px solid blue;")

    def copy_selected_images(self):
        if not self.selected_images:
            QMessageBox.warning(self, "Error", "No images selected.")
            return

        if not os.path.exists('copied_images'):
            os.makedirs('copied_images')

        for img_name in self.selected_images:
            image_path = os.path.join(image_folder, img_name)
            if os.path.exists(image_path):
                dest_path = os.path.join('copied_images', img_name)
                shutil.copy(image_path, dest_path)
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
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# Entry point
if __name__ == "__main__":
    main()