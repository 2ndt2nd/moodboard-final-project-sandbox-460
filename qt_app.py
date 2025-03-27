import sys
import os
import random
import shutil
import webbrowser
import threading
import torch
import clip
from tqdm import tqdm
from PyQt5.QtCore import Qt, QObject, QUrl, QThread, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QCursor, QKeySequence, QPainter
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QGridLayout, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QShortcut, QProgressBar, QProgressDialog, QMenu


# Global variables
image_folder = "illustration_dataset"  # Change to your actual image folder
last_match_results = []
text_features_dict = {}
image_features_dict = {}
sg = None
sh = 0
sw = 0
model, preprocess = clip.load("ViT-B/32", device="cpu")
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
def match_query(input_query, progress_signal=None):
    global prompt_results_cache

    # Check if results are already cached
    if input_query in prompt_results_cache:
        print("Using cached results for:", input_query)
        if progress_signal:
            progress_signal.finished.emit(prompt_results_cache[input_query])
        return prompt_results_cache[input_query]

    # Get the text features for the query
    if input_query in text_features_dict:
        text_features = text_features_dict[input_query]
    else:
        text_features = extract_text_features(input_query)

    def compute_similarities():
        similarities = {}
        total_images = len(image_features_dict)  # Define total_images here
        
        # Process images
        for i, (img_name, img_features) in enumerate(image_features_dict.items(), 1):
            similarities[img_name] = torch.cosine_similarity(text_features, img_features, dim=-1).item()
            if progress_signal:
                progress_signal.progress_updated.emit(i, total_images)

        # Sorting images and returning on finish
        sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        prompt_results_cache[input_query] = sorted_images
        if progress_signal:
            progress_signal.finished.emit(sorted_images)
        
        return sorted_images

    thread = threading.Thread(target=compute_similarities)
    thread.start()

    if not progress_signal:  # Only wait if no signal is provided
        thread.join()
        return prompt_results_cache[input_query]

def get_closest_texts(image_name, top_k=5):
    global text_features_dict, image_features_dict
    
    img_features = image_features_dict[image_name]
    similarities = {}

    # Find words that apply to the image
    for text, text_features in text_features_dict.items():
        similarities[text] = torch.cosine_similarity(
            img_features, text_features, dim=-1
        ).item()
    
    #Return sorted array of words
    sorted_texts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(sorted_texts)
    return [text for text, score in sorted_texts]

def find_similar_images(target_img_name, top_k=16):  # Increased default to 16
    global image_features_dict
    
    target_features = image_features_dict[target_img_name]
    similarities = {}
    
    for img_name, features in image_features_dict.items():
        if img_name != target_img_name:
            similarity = torch.cosine_similarity(
                target_features.unsqueeze(0),
                features.unsqueeze(0),
                dim=-1
            ).item()
            similarities[img_name] = similarity
    
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

class ProgressSignal(QObject):
    progress_updated = pyqtSignal(int, int)  # (current, total)
    finished = pyqtSignal(list)

# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        global sg, sw, sh
        super().__init__()

        sg = QApplication.desktop().screenGeometry()
        sw = sg.width()
        sh = sg.height()
        self.setWindowTitle("Welcome")
        self.setGeometry(sw//2, sh//2, 400, 200)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.title_label = QLabel("Write out your moodboard prompts!")
        self.title_label.setStyleSheet("font-size: 30px; font-weight: bold; font-family: Arial;")
        self.layout.addWidget(self.title_label)

        self.input_label = QLabel("Enter your prompt:")
        self.layout.addWidget(self.input_label)

        self.input_box = QLineEdit()
        self.layout.addWidget(self.input_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.progress_signal = ProgressSignal()
        self.progress_signal.progress_updated.connect(self.update_progress)
        self.progress_signal.finished.connect(self.on_similarity_complete)

        # Add a start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_button_clicked)
        self.layout.addWidget(self.start_button)

## Subject for moving

    def start_button_clicked(self):
        input_text = self.input_box.text()
        if not input_text:
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        
        # Start matching with progress updates
        match_query(input_text, self.progress_signal)
   
    def update_progress(self, current, total):
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)

    def on_similarity_complete(self, results):
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        
        # Pass results to the image grid window
        self.image_grid_window = ImageGridWindow(self.input_box.text(), results)
        self.image_grid_window.show()

## Subject for moving

# Image grid window class
class ImageGridWindow(QMainWindow):
    def __init__(self, input_text, match_results=None):
        super().__init__()
        self.setWindowTitle("Image Grid")

        self.setGeometry(0, 0, sw, sh)

        self.input_text = input_text
        self.selected_images = {}
        self.match_results = match_results  # Store but don't create grid yet

        # Create UI elements but don't populate grid
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

        # Create Shuffle button
        self.shuffle_button = QPushButton("Shuffle!")
        self.shuffle_button.clicked.connect(self.shuffle_images)
        self.layout.addWidget(self.shuffle_button)

        # Create Copy button
        self.copy_button = QPushButton("Copy!")
        self.copy_button.clicked.connect(self.copy_selected_images)
        self.layout.addWidget(self.copy_button)

        # Create Moodboard button
        self.open_moodboard_button = QPushButton("Open Moodboard")
        self.open_moodboard_button.clicked.connect(self.open_moodboard)
        self.layout.addWidget(self.open_moodboard_button)

        # Add images to the grid
        if match_results:  # Use provided results if available
            self.match_results = match_results
            self.create_image_grid()

        self.reference_image = input_text if isinstance(input_text, str) else None

    def create_image_grid(self):
        if not self.match_results:
            QMessageBox.warning(self, "Error", "No images found matching the criteria")
            return
    
        # Calculate how many images we can actually show
        available_images = len(self.match_results)
        top_k = min(16, available_images)
        top_n = min(50, available_images)
        
        # Get random subset (now guaranteed to work)
        if available_images <= top_k:
            # If we have few images, just show them all
            image_files = [img for img, score in self.match_results]
        else:
            # Otherwise get a random sample
            random_subset = random.sample(self.match_results[:top_n], top_k)
            image_files = [img for img, score in random_subset]

        num_columns = 4  
        image_size = 250  
        padding = 10 
        window_width = num_columns * (image_size + padding)
        window_height = sg.height()-100
        self.resize(window_width, window_height)

        for idx, img_file in enumerate(image_files):
            image_path = os.path.join(image_folder, img_file)
            image = QImage(image_path)
            pixmap = QPixmap.fromImage(image).scaled(image_size, image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            label = QLabel()
            label.setPixmap(pixmap)
            label.setStyleSheet("border: 2px solid red;")
            
            # Connect both left and right click events
            label.mousePressEvent = lambda event, name=img_file, lbl=label: (
                self.on_image_click(name, lbl) 
                if event.button() == Qt.LeftButton 
                else (
                    self.show_context_menu(event.pos(), name, lbl) 
                    if event.button() == Qt.RightButton 
                else None)
                )
        

            # Calculate row and column based on the number of columns
            row = idx // num_columns
            column = idx % num_columns
            self.grid_layout.addWidget(label, row, column, alignment=Qt.AlignCenter)

    def shuffle_images(self):
        top_k = min(16, available_images)
        top_n = min(50, available_images)
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
            label.mousePressEvent = lambda event, name=img_file, lbl=label: (
                self.on_image_click(name, lbl) 
                if event.button() == Qt.LeftButton 
                else (
                    self.show_context_menu(event.pos(), name, lbl) 
                    if event.button() == Qt.RightButton 
                else None)
                )
            
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

    def show_similar(self, img_name):
        progress = QProgressDialog("Finding similar images...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        QApplication.processEvents()
        
        # Get visually similar images first
        similar_images = find_similar_images(img_name, top_k=16)  # Get max we might need
        text_descriptions = get_closest_texts(img_name, top_k=3)  # Get top 3 text descriptors
        
        # If we don't have enough visually similar images, supplement with text-based matches
        if len(similar_images) < 16 and text_descriptions:
            # Use the most relevant text descriptor to find additional matches
            text_based_matches = match_query(text_descriptions[0], None)
            
            # Filter out images already in similar_images and the original image
            existing_images = {img for img, _ in similar_images} | {img_name}
            additional_matches = [
                (img, score) for img, score in text_based_matches 
                if img not in existing_images
            ][:16 - len(similar_images)]
            
            similar_images.extend(additional_matches)
        
        progress.close()
        
        if similar_images:
            # Show similarity scores in tooltips
            for img, score in similar_images:
                print(f"{img}: {score:.3f}")
                
            self.similar_window = ImageGridWindow(f"Similar to {img_name}", similar_images)
            self.similar_window.show()
        else:
            QMessageBox.warning(self, "Error", "No similar images found")
        
    def show_context_menu(self, pos, img_name, label):
        menu = QMenu(self)
        
        # Add actions
        # find_source = menu.addAction("Find Original Source")
        view_action = menu.addAction("View Full Size")
        find_similar = menu.addAction("Find Similar Images")
        
        # # Connect actions to functions
        # find_source.triggered.connect(lambda: self.on_image_click(img_name, label))
        # view_action.triggered.connect(lambda: self.view_full_size(img_name))
        find_similar.triggered.connect(lambda: self.show_similar(img_name))
        
        # Show the menu at cursor position
        menu.exec_(QCursor.pos())

    def open_moodboard(self):
        if not self.selected_images:
            QMessageBox.warning(self, "Error", "No images selected.")
            return

        # Get the paths of the selected images
        selected_image_paths = [os.path.join(image_folder, img_name) for img_name in self.selected_images]

        # Open the moodboard canvas window
        self.moodboard_window = MoodboardCanvasWindow(selected_image_paths)
        self.moodboard_window.show()

        def start_button_clicked(self):
            input_text = self.input_box.text()
            if not input_text:
                QMessageBox.warning(self, "Error", "Please enter a prompt.")
                return

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.start_button.setEnabled(False)
            
            # Start matching with progress updates
            match_query(input_text, self.progress_signal)


    ## Immigrating features
    def start_button_clicked(self):
        input_text = self.input_box.text()
        if not input_text:
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        
        # Start matching with progress updates
        match_query(input_text, self.progress_signal)
   
    def update_progress(self, current, total):
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)

    def on_similarity_complete(self, results):
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        
        # Pass results to the image grid window
        self.image_grid_window = ImageGridWindow(self.input_box.text(), results)
        self.image_grid_window.show()

    ## Immigrating

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
        self.setGeometry(0, 0, sw, sh)

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
        self.scale_up_shortcut = QShortcut(QKeySequence("="), self)
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
        self.selected_item = None

    def scene_to_svg(self):
        # Ensure selection border doesn't appear in SVG
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
    global text_features_dict, image_features_dict

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    device = "cpu"

    image_features_dict = torch.load("image_embeddings.pt", map_location=torch.device('cpu'))
    text_features_dict = torch.load("text_embeddings.pt", map_location=torch.device('cpu'))

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# Entry point
if __name__ == "__main__":
    main()