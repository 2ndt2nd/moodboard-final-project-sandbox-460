import sys
import os
import random
import shutil
import webbrowser
import threading
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QGridLayout, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtWebEngineWidgets import QWebEngineView
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

        # Add Open Moodboard button
        self.open_moodboard_button = QPushButton("Open Moodboard")
        self.open_moodboard_button.clicked.connect(self.open_moodboard)
        self.layout.addWidget(self.open_moodboard_button)

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

    def open_moodboard(self):
        if not self.selected_images:
            QMessageBox.warning(self, "Error", "No images selected.")
            return

        # Get the paths of the selected images
        selected_image_paths = [os.path.join(image_folder, img_name) for img_name in self.selected_images]

        # Open the moodboard canvas window
        self.moodboard_window = MoodboardCanvasWindow(selected_image_paths)
        self.moodboard_window.show()

from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsItem, QShortcut
from PyQt5.QtCore import QRectF, QPointF
from PyQt5.QtGui import QCursor, QKeySequence
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtGui import QPainter

class ImageLoaderThread(QThread):
    finished = pyqtSignal(list)  # Signal to emit when loading is complete

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        pixmaps = []
        for image_path in self.image_paths:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                pixmaps.append(pixmap)
        self.finished.emit(pixmaps)  # Emit the loaded pixmaps

class ResizablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        # Store the original pixmap for resizing
        self.original_pixmap = pixmap

    def scale_image(self, factor):
        """Scale the image by a given factor."""
        # Get the current size of the image
        current_size = self.pixmap().size()

        # Calculate the new size
        new_width = int(current_size.width() * factor)
        new_height = int(current_size.height() * factor)

        # Resize the image while maintaining aspect ratio
        scaled_pixmap = self.original_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)

class CustomGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panning = False
        self._pan_start_pos = QPointF()

    def wheelEvent(self, event):
        # Check if Ctrl is pressed
        if event.modifiers() & Qt.ControlModifier:
            # Zoom in or out based on scroll direction
            zoom_factor = 1.2 if event.angleDelta().y() > 0 else 0.8
            self.scale(zoom_factor, zoom_factor)
        else:
            # Default behavior (scroll without zooming)
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self._panning:
            # Start panning
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            # Pan the view
            delta = self._pan_start_pos - event.pos()
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._panning:
            # Stop panning
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            # Enable panning when spacebar is pressed
            self._panning = True
            self.setCursor(Qt.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            # Disable panning when spacebar is released
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

class MoodboardCanvasWindow(QMainWindow):
    def __init__(self, image_paths):
        super().__init__()
        self.setWindowTitle("Moodboard Canvas")

        screen_geometry = QApplication.desktop().screenGeometry()
        window_height = screen_geometry.height() - 100
        window_width = screen_geometry.width() - 100
        self.setGeometry(100, 100, window_width, window_height)

        # Create the main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Create a custom QGraphicsView and QGraphicsScene
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene)  # Use CustomGraphicsView
        self.layout.addWidget(self.view)

        # Track the selected image
        self.selected_item = None

        # Track the highest z-value
        self.highest_z_value = 0

        # Add Zoom Out button
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.layout.addWidget(self.zoom_out_button)

        # Add Zoom In button
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        self.layout.addWidget(self.zoom_in_button)

        # Add Reset Zoom button
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.layout.addWidget(self.reset_zoom_button)

        # Add keyboard shortcuts for zooming
        self.zoom_in_shortcut = QShortcut(QKeySequence("Ctrl+="), self)
        self.zoom_in_shortcut.activated.connect(self.zoom_in)

        self.zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        self.zoom_out_shortcut.activated.connect(self.zoom_out)

        self.reset_zoom_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self.reset_zoom_shortcut.activated.connect(self.reset_zoom)

        # Add keyboard shortcuts for scaling the selected image
        self.scale_down_shortcut = QShortcut(QKeySequence("-"), self)
        self.scale_down_shortcut.activated.connect(self.scale_down)

        self.scale_up_shortcut = QShortcut(QKeySequence("+"), self)
        self.scale_up_shortcut.activated.connect(self.scale_up)

        # Add images to the scene
        last_width_pos = 0
        for idx, image_path in enumerate(image_paths):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                last_width_pos += pixmap.width()
                resizable_item = ResizablePixmapItem(pixmap)
                resizable_item.setPos(last_width_pos + 50, 0)  # Adjust initial positions
                self.scene.addItem(resizable_item)

                # Connect the item's selection event
                resizable_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
                resizable_item.mousePressEvent = lambda event, item=resizable_item: self.select_item(item)

        # Add Save button
        self.save_button = QPushButton("Save Moodboard as SVG")
        self.save_button.clicked.connect(self.save_moodboard)
        self.layout.addWidget(self.save_button)

    def select_item(self, item):
        """Set the selected item and bring it to the topmost layer."""
        if self.selected_item:
            self.selected_item.setSelected(False)  # Deselect the previously selected item

        # Bring the clicked item to the topmost layer
        self.highest_z_value += 1
        item.setZValue(self.highest_z_value)

        # Select the new item
        self.selected_item = item
        self.selected_item.setSelected(True)

    def scale_down(self):
        """Scale the selected image down by 10%."""
        if self.selected_item:
            self.selected_item.scale_image(0.9)  # Scale down by 10%

    def scale_up(self):
        """Scale the selected image up by 10%."""
        if self.selected_item:
            self.selected_item.scale_image(1.1)  # Scale up by 10%

    def zoom_in(self):
        """Zoom in by scaling the view."""
        self.view.scale(1.2, 1.2)  # Increase scale by 20%

    def zoom_out(self):
        """Zoom out by scaling the view."""
        self.view.scale(0.8, 0.8)  # Decrease scale by 20%

    def reset_zoom(self):
        """Reset the zoom level to the original scale."""
        self.view.resetTransform()  # Reset the view's transformation matrix

    def save_moodboard(self):
        # Save the current scene as an SVG file
        svg_file = "moodboard.svg"
        with open(svg_file, "w") as f:
            f.write(self.scene_to_svg())
        print(f"Moodboard saved to {svg_file}")

    def deselect_all_items(self):
    """Deselect all items in the scene."""
    for item in self.scene.items():
        item.setSelected(False)

    def scene_to_svg(self):
        self.deselect_all_items()

        generator = QSvgGenerator()
        generator.setFileName("moodboard.svg")
        generator.setSize(self.scene.sceneRect().size().toSize())
        generator.setViewBox(self.scene.sceneRect())

        painter = QPainter()
        painter.begin(generator)
        self.scene.render(painter)
        painter.end()

        with open("moodboard.svg", "r") as f:
            return f.read()

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