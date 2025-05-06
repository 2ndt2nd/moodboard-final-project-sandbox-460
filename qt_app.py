import sys
import os
import random
import shutil
import threading
import torch
import clip
from PIL import Image
from tqdm import tqdm
from PyQt5.QtCore import Qt, QObject, QUrl, QPointF, pyqtSignal, QTimer, QMimeData, QUrl
from PyQt5.QtGui import QPixmap, QImage, QCursor, QKeySequence, QPainter
from PyQt5.QtSvg import QSvgGenerator
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QGridLayout, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsItem, QShortcut, QProgressBar, QProgressDialog, QMenu, QTabWidget, QFileDialog


# Global variables
image_folder = ""  # Change to your actual image folder
sh = 0
sw = 0
model, preprocess = clip.load("ViT-B/32", device="cpu")
device = "cpu"

# Global dictionary to store sorted results for each prompt
# prompt_results_cache = {}

def create_click_handler(parent, img_name, label):
    def handler(event):
        if event.button() == Qt.LeftButton:
            parent.on_image_click(img_name, label)
        elif event.button() == Qt.RightButton:
            parent.show_context_menu(event.pos(), img_name, label)
    return handler

# Function to extract text features
def extract_text_features(text):
    text_tokenized = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokenized)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features

# Function to match a query
def match_query(image_features_dict, text_features_dict, input_query, progress_signal=None):
    # global prompt_results_cache

    # # Check if results are already cached
    # if input_query in prompt_results_cache:
    #     if progress_signal:
    #         progress_signal.finished.emit(prompt_results_cache[input_query])
    #     return prompt_results_cache[input_query]

    # Get the text features for the query
    if input_query in text_features_dict:
        text_features = text_features_dict[input_query]
    else:
        text_features = extract_text_features(input_query)

    def compute_similarities(image_features_dict):
        similarities = {}
        total_images = len(image_features_dict)  # Define total_images here
        
        # Process images
        for i, (img_name, img_features) in enumerate(image_features_dict.items(), 1):
            similarities[img_name] = torch.cosine_similarity(text_features, img_features, dim=-1).item()
            if progress_signal:
                progress_signal.progress_updated.emit(i, total_images)

        # Sorting images and returning on finish
        sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        # prompt_results_cache[input_query] = sorted_images
        if progress_signal:
            progress_signal.finished.emit(sorted_images)
        
        return sorted_images

    thread = threading.Thread(
        target=compute_similarities,
        args=(image_features_dict,)
    )
    thread.start()

    if not progress_signal:  # Only wait if no signal is provided
        thread.join()
        return #prompt_results_cache[input_query]

def get_closest_texts(image_features_dict, text_features_dict, image_name, top_k=3):
    img_features = image_features_dict[image_name]
    similarities = {}

    # Find words that apply to the image
    for text, text_features in text_features_dict.items():
        similarities[text] = torch.cosine_similarity(
            img_features, text_features, dim=-1
        ).item()
    
    #Return sorted array of words
    sorted_texts = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [text for text, score in sorted_texts]

def find_similar_images(image_features_dict, target_img_name, top_k=16):  # Increased default to 16
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


class MainWindow(QMainWindow):
    def __init__(self):
        global sg, sw, sh
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)

        screen = QApplication.primaryScreen()
        sg = screen.geometry()
        sw, sh = sg.width(), sg.height()
        self.setWindowTitle("MoodForager")
        self.setGeometry(sw//4, sh//4, 1200, 800)
        self.image_features_dict = {}

        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.layout.addWidget(self.tab_widget)

        # Add home tab
        self.add_home_tab()

        # Initialize moodboard window (will be shown in its own window)
        self.moodboard_window = None

    def add_home_tab(self):
        """Create and add the home tab with search functionality"""
        home_tab = QWidget()
        home_layout = QVBoxLayout(home_tab)
        
        # Add all the home page widgets
        self.title_label = QLabel("MoodForager: Empowering Moodboards for Creatives")
        self.title_label.setStyleSheet("font-size: 30px; font-weight: bold; font-family: Arial;")
        home_layout.addWidget(self.title_label)

        # Add drag and drop area
        self.drop_area = QLabel("Drag & Drop Images Here to Find Similar Images")
        self.drop_area.setStyleSheet("""
            QLabel {
                font-size: 16px;
                border: 2px dashed #aaa;
                padding: 20px;
                min-height: 100px;
                qproperty-alignment: AlignCenter;
            }
            QLabel:hover {
                border: 2px dashed #666;
                background-color: #f0f0f0;
            }
        """)
        self.drop_area.setAcceptDrops(True)
        self.drop_area.dragEnterEvent = self.dragEnterEvent
        self.drop_area.dropEvent = self.dropEvent
        home_layout.addWidget(self.drop_area)

        intro_label = QLabel(
            "MoodForager allows you to quickly create moodboards and accelerate your creative ideation!\n"
            "- Retrieve images from a local library by selecting text\n"
            "- Select images to power your ideation\n"
            "- Find similar images to expand your search\n"
            "- Arrange your own moodboards and start working in no time!"
        )
        intro_label.setStyleSheet("font-size: 15px; padding: 3px; color: black;")
        home_layout.addWidget(intro_label)

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Target folder: Not selected")
        self.folder_label.setStyleSheet("font-size: 12px;")
        folder_layout.addWidget(self.folder_label)
        
        self.folder_button = QPushButton("Browse...")
        self.folder_button.setStyleSheet("font-size: 12px; min-height: 25px; padding: 2px;")
        self.folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_button)
        home_layout.addLayout(folder_layout)

        self.input_label = QLabel("Enter your prompt:")
        home_layout.addWidget(self.input_label)

        self.input_box = QLineEdit()
        self.input_box.setStyleSheet("font-size: 18px; min-height: 40px; padding: 5px;")
        home_layout.addWidget(self.input_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        home_layout.addWidget(self.progress_bar)

        self.progress_signal = ProgressSignal()
        self.progress_signal.progress_updated.connect(self.update_progress)
        self.progress_signal.finished.connect(self.on_similarity_complete)

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("font-size: 18px; min-height: 40px; padding: 5px;")
        self.start_button.clicked.connect(self.start_button_clicked)
        home_layout.addWidget(self.start_button)

        self.start_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        self.start_shortcut.activated.connect(self.start_button_clicked)

        reference_label = QLabel(
            "Using CLIP by OpenAI\n\n"
            "Dataset taken from https://github.com/BathVisArtData/PeopleArt with minor adjustments\n\n"
            "@inproceedings{westlake2016detecting,\n"
            "title={Detecting People in Artwork with CNNs},\n"
            "author={Westlake, Nicholas and Cai, Hongping and Hall, Peter},\n"
            "booktitle={European Conference on Computer Vision},\n"
            "pages={825--841},\n"
            "year={2016},\n"
            "organization={Springer}\n"
            "}"
        )
        reference_label.setStyleSheet("font-size: 9px; padding: 4px; color: gray;")
        home_layout.addWidget(reference_label)

        # Add stretch to push everything up
        home_layout.addStretch()

        # Add home tab to the tab widget
        self.tab_widget.addTab(home_tab, "Home")
        self.tab_widget.setCurrentIndex(0)

    def close_tab(self, index):
        """Close a tab at the given index"""
        if index != 0:  # Don't close the home tab
            self.tab_widget.removeTab(index)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.process_dropped_image(file_path)
        event.acceptProposedAction()

    def process_dropped_image(self, image_path):
        """Process a dropped image to find similar images"""
        if not self.image_features_dict:
            QMessageBox.warning(self, "Error", "Please select a folder with image embeddings first.")
            return

        try:
            progress = QProgressDialog("Processing image...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButton(None)
            progress.show()
            QApplication.processEvents()

            # Load and preprocess the image
            image = Image.open(image_path)
            image_input = preprocess(image).unsqueeze(0).to(device)

            # Extract features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Find similar images
            similarities = {}
            total_images = len(self.image_features_dict)
            for i, (img_name, features) in enumerate(self.image_features_dict.items(), 1):
                similarities[img_name] = torch.cosine_similarity(
                    image_features, features.unsqueeze(0), dim=-1).item()
                progress.setValue(int(i/total_images*100))
                QApplication.processEvents()

            # Sort and show results
            sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:16]
            self.add_results_tab(f"Similar to {os.path.basename(image_path)}", sorted_images)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image:\n{str(e)}")
        finally:
            progress.close()

    def select_folder(self):
        """Open a folder selection dialog and store the selected path"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Target Folder",
            "",  # Start in current directory
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            embeddings_path = os.path.join(folder, "img.pt")
            if not os.path.exists(embeddings_path):
                self.folder_label.setText(f"Target folder {folder} has no image embeddings!")
                QMessageBox.warning(self, "Error", "The selected folder doesn't contain img.pt")
                return
                
            try:
                progress = QProgressDialog("Loading image embeddings...", None, 0, 0, self)
                progress.setWindowModality(Qt.WindowModal)
                progress.setCancelButton(None)
                progress.show()
                QApplication.processEvents()
                
                def load_embeddings():
                    global image_folder
                    self.image_features_dict = torch.load(embeddings_path, map_location=torch.device('cpu'), weights_only=True)
                    self.text_features_dict = torch.load(os.path.join(folder, "text.pt"), map_location=torch.device('cpu'), weights_only=True)

                    image_folder = folder
                    self.folder_label.setText(f"Target folder: {folder}")
                    self.folder_label.setToolTip(folder)
                    progress.close()
                
                QTimer.singleShot(100, load_embeddings)

            except Exception as e:
                self.folder_label.setText(f"Error loading embeddings from {folder}")
                QMessageBox.critical(self, "Error", f"Failed to load embeddings:\n{str(e)}")

    def start_button_clicked(self):
        input_text = self.input_box.text()
        if not input_text:
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return
        
        # if image_folder == "":
        #     QMessageBox.warning(self, "Error", "Please select a folder")
        #     return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        match_query(self.image_features_dict, self.text_features_dict, input_text, self.progress_signal)
   
    def update_progress(self, current, total):
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)

    def on_similarity_complete(self, results):
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        
        # Create a new tab for the results
        self.add_results_tab(self.input_box.text(), results)

    def add_results_tab(self, title, results):
        """Add a new tab with image grid results"""
        # Check if we already have a tab with this title
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == title:
                # Switch to existing tab
                self.tab_widget.setCurrentIndex(i)
                return
        
        # Create new tab
        image_grid = ImageGridWindow(title, image_folder, self.image_features_dict, self.text_features_dict, results, main_window=self)
        scroll = QScrollArea()
        scroll.setWidget(image_grid)
        scroll.setWidgetResizable(True)
        
        # Add the tab
        self.tab_widget.addTab(scroll, title)
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    def open_moodboard(self, image_paths):
        """Open the moodboard window with selected images"""
        if not image_paths:
            QMessageBox.warning(self, "Error", "No images selected.")
            return
            
        if not hasattr(self, 'moodboard_window') or self.moodboard_window is None:
            self.moodboard_window = MoodboardCanvasWindow(image_paths)
        else:
            self.moodboard_window.add_images_to_scene(image_paths)
        
        self.moodboard_window.show()
        self.moodboard_window.raise_()
        self.moodboard_window.activateWindow()

class ImageGridWindow(QWidget):
    def __init__(self, input_text, class_folder, image_features_dict, text_features_dict, match_results=None, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.image_folder = class_folder
        self.image_features_dict = image_features_dict
        self.text_features_dict = text_features_dict

        self.input_text = input_text
        self.selected_images = set()  # Using set instead of dict for selected images
        self.match_results = match_results or []

        self.setup_ui()
        if self.match_results:
            self.create_image_grid()

    def setup_ui(self):
        """Initialize all UI components"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Header section
        header_layout = QHBoxLayout()
        self.grid_label = QLabel(f"Results for: {self.input_text}")
        self.grid_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        header_layout.addWidget(self.grid_label)
        
        self.selected_count_label = QLabel("0 selected")
        self.selected_count_label.setStyleSheet("font-size: 14px; color: #555;")
        header_layout.addWidget(self.selected_count_label)
        header_layout.addStretch()
        
        self.layout.addLayout(header_layout)

        # Image grid area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout(self.scroll_widget)
        self.grid_layout.setSpacing(15)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)

        # Button toolbar
        self.setup_toolbar()

    def setup_toolbar(self):
        """Create the bottom toolbar with action buttons"""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        # Action buttons
        self.shuffle_button = self.create_tool_button("Shuffle", self.shuffle_images)
        self.copy_button = self.create_tool_button("Copy to Folder", self.copy_selected_images)
        self.moodboard_button = self.create_tool_button("Add to Moodboard", self.open_moodboard)
        self.select_all_button = self.create_tool_button("Select All", self.select_all_images)
        self.clear_selection_button = self.create_tool_button("Clear Selection", self.clear_selection)

        # Add buttons to toolbar
        for btn in [self.shuffle_button, self.copy_button, self.moodboard_button, 
                   self.select_all_button, self.clear_selection_button]:
            toolbar.addWidget(btn)

        toolbar.addStretch()
        self.layout.addLayout(toolbar)

    def create_tool_button(self, text, handler):
        """Helper to create consistent toolbar buttons"""
        btn = QPushButton(text)
        btn.setStyleSheet("""
            QPushButton {
                font-size: 14px; 
                min-height: 30px; 
                padding: 5px 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #f0f0f0;
            }
        """)
        btn.clicked.connect(handler)
        return btn

    def create_image_grid(self):
        self.clear_grid()
        
        available_images = len(self.match_results)
        top_k = min(16, available_images)
        top_n = min(40, available_images)
        
        image_files = self.get_image_subset(self.match_results, available_images, top_k, top_n)
        self.display_images(image_files)

    def get_image_subset(self, results, available, top_k, top_n):
        if available <= top_k:
            return [img for img, _ in results]
        return [img for img, _ in random.sample(results[:top_n], top_k)]

    def display_images(self, image_files):
        num_columns = 4
        image_size = 300
        
        for idx, img_file in enumerate(image_files):
            image_path = os.path.join(self.image_folder, img_file)
            try:
                pixmap = QPixmap(image_path)
                if pixmap.isNull():
                    continue
                    
                pixmap = pixmap.scaled(image_size, image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                label = ClickableImageLabel(pixmap, img_file)
                label.clicked.connect(self.on_image_clicked)
                label.right_clicked.connect(self.show_context_menu)
                
                row = idx // num_columns
                column = idx % num_columns
                self.grid_layout.addWidget(label, row, column, alignment=Qt.AlignCenter)
                
            except Exception as e:
                print(f"Error loading image {img_file}: {str(e)}")

    def on_image_clicked(self, img_name, label):
        if img_name in self.selected_images:
            self.selected_images.remove(img_name)
            label.set_selected(False)
        else:
            self.selected_images.add(img_name)
            label.set_selected(True)
        
        self.update_selection_count()

    def update_selection_count(self):
        count = len(self.selected_images)
        self.selected_count_label.setText(f"{count} selected")
        self.copy_button.setEnabled(count > 0)
        self.moodboard_button.setEnabled(count > 0)

    def select_all_images(self):
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if isinstance(item.widget(), ClickableImageLabel):
                label = item.widget()
                self.selected_images.add(label.image_name)
                label.set_selected(True)
        self.update_selection_count()

    def clear_selection(self):
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(i)
            if isinstance(item.widget(), ClickableImageLabel):
                item.widget().set_selected(False)
        self.selected_images.clear()
        self.update_selection_count()

    def clear_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def shuffle_images(self):
        available_images = len(self.match_results)
        top_k = min(16, available_images)
        top_n = min(80, available_images)
        
        random_subset = random.sample(self.match_results[:top_n], top_k)
        image_files = [img for img, _ in random_subset]
        
        self.clear_selection()
        self.clear_grid()
        self.display_images(image_files)

    def open_image(self, img_name):
        """Open the containing folder and select the image"""
        image_path = os.path.join(self.image_folder, img_name)
        
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image not found:\n{image_path}")
            return
        try:
            if sys.platform == 'win32':
                os.startfile(os.path.normpath(image_path))
            elif sys.platform == 'darwin':
                subprocess.run(['open', '-R', image_path])
            else:
                subprocess.run(['xdg-open', os.path.dirname(image_path)])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open folder:\n{str(e)}")

    def copy_selected_images(self):
        if not self.selected_images:
            QMessageBox.warning(self, "No Selection", "Please select images to copy.")
            return

        os.makedirs('copied_images', exist_ok=True)
        copied = 0
        
        for img_name in self.selected_images:
            try:
                src_path = os.path.join(self.image_folder, img_name)
                if os.path.exists(src_path):
                    dest_path = os.path.join('copied_images', img_name)
                    shutil.copy(src_path, dest_path)
                    copied += 1
            except Exception as e:
                print(f"Error copying {img_name}: {str(e)}")
        
        if copied > 0:
            try:
                if sys.platform == 'win32':
                    os.startfile(os.path.normpath('copied_images'))
                elif sys.platform == 'darwin':
                    subprocess.run(['open', 'copied_images'])
                else:
                    subprocess.run(['xdg-open', 'copied_images'])
            except Exception as e:
                print(f"Error opening folder: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", "No images were copied.")

    def show_context_menu(self, pos, img_name):
        menu = QMenu(self)
        
        find_similar = menu.addAction("Find Similar Images")
        open_action = menu.addAction("Open Image in Photo Viewer")
        menu.addSeparator()
        select_all = menu.addAction("Select All")
        clear_selection = menu.addAction("Clear Selection")
        
        action = menu.exec_(QCursor.pos())
        
        if action == find_similar:
            self.show_similar_images(img_name)
        elif action == open_action:
            self.open_image(img_name)
        elif action == select_all:
            self.select_all_images()
        elif action == clear_selection:
            self.clear_selection()

    def show_similar_images(self, img_name):
        """Find and display similar images"""
        progress = QProgressDialog("Finding similar images...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        QApplication.processEvents()
        
        try:
            similar_images = find_similar_images(self.image_features_dict, img_name, top_k=16)
            text_descriptions = get_closest_texts(self.image_features_dict, self.text_features_dict, img_name, top_k=3)
            
            if len(similar_images) < 16 and text_descriptions:
                text_based_matches = []
                for desc in text_descriptions[:2]:
                    text_based_matches.extend(match_query(self.image_features_dict, self.text_features_dict, desc, None))
                
                existing_images = {img for img, _ in similar_images} | {img_name}
                additional_matches = [
                    (img, score) for img, score in text_based_matches 
                    if img not in existing_images
                ][:16 - len(similar_images)]
                similar_images.extend(additional_matches)
            
            if similar_images:
                title = f"Similar to {os.path.splitext(img_name)[0]}"
                self.main_window.add_results_tab(title, similar_images)
            else:
                QMessageBox.warning(self, "No Results", "No similar images found.")
                
        finally:
            progress.close()

    def open_moodboard(self):
        """Open selected images in moodboard"""
        if not self.selected_images:
            QMessageBox.warning(self, "No Selection", "Please select images to add to moodboard.")
            return
            
        image_paths = [os.path.join(self.image_folder, img_name) 
                      for img_name in self.selected_images]
        self.main_window.open_moodboard(image_paths)


class ClickableImageLabel(QLabel):
    """Custom QLabel that handles click events and selection state"""
    clicked = pyqtSignal(str, object)  # (image_name, label)
    right_clicked = pyqtSignal(QPointF, str)  # (position, image_name)
    
    def __init__(self, pixmap, image_name):
        super().__init__()
        self.image_name = image_name
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid transparent;")
        self.set_selected(False)
        
    def set_selected(self, selected):
        """Update visual selection state"""
        self.selected = selected
        border = "2px solid blue" if selected else "2px solid transparent"
        self.setStyleSheet(f"border: {border};")
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.image_name, self)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(event.pos(), self.image_name)

class ResizablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, name = None):
        super().__init__(pixmap)
        self.imnames = name
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
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(delta.y()))
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
    moodboard_items = []  
    def __init__(self, image_paths=None):
        super().__init__()
        self.setWindowTitle("Moodboard Canvas")
        mbh = sh
        if(sh>500):
            mbh-300
        self.setGeometry(0, 0, 1000, mbh)
        self.showMaximized()
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Create a custom QGraphicsView and QGraphicsScene
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene)
        self.layout.addWidget(self.view)

        self.selected_item = None
        self.highest_z_value = 0


        # Addding images to scene
        if image_paths:
            last_width_pos = 0
            for idx, image_path in enumerate(image_paths):
                pixmap = QPixmap(image_path)
                if not pixmap.isNull() and image_path not in self.moodboard_items:
                    last_width_pos += pixmap.width()
                    resizable_item = ResizablePixmapItem(pixmap)
                    resizable_item.setPos(last_width_pos + 50, 0)
                    self.scene.addItem(resizable_item)
                    resizable_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
                    resizable_item.mousePressEvent = lambda event, item=resizable_item: self.select_item(item)
                    self.moodboard_items.append(image_path)
                
        # Add Horizontal button row
        button_row = QHBoxLayout()
        
        # Zoom Out button
        self.zoom_out_button = QPushButton("Zoom Out")
        self.zoom_out_button.setStyleSheet("font-size: 15px; min-height: 30px; padding: 2px;")
        self.zoom_out_button.clicked.connect(self.zoom_out)
        button_row.addWidget(self.zoom_out_button)
        
        # Zoom In button
        self.zoom_in_button = QPushButton("Zoom In")
        self.zoom_in_button.setStyleSheet("font-size: 15px; min-height: 30px; padding: 2px;")
        self.zoom_in_button.clicked.connect(self.zoom_in)
        button_row.addWidget(self.zoom_in_button)
        
        # Clear Board button
        self.clear_board_button = QPushButton("Clear Canvas")
        self.clear_board_button.setStyleSheet("font-size: 15px; min-height: 30px; padding: 2px;")
        self.clear_board_button.clicked.connect(self.clear_board)
        button_row.addWidget(self.clear_board_button)
        

        # Reset Zoom button
        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.reset_zoom_button.setStyleSheet("font-size: 15px; min-height: 30px; padding: 2px;")
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        button_row.addWidget(self.reset_zoom_button)

        # Add Save button
        self.save_button = QPushButton("Save Moodboard as SVG")
        self.save_button.setStyleSheet("font-size: 15px; min-height: 30px; padding: 2px;")
        self.save_button.clicked.connect(self.save_moodboard)
        button_row.addWidget(self.save_button)

        shortcut_hints = QLabel(
            "Shortcuts:\n"
            "  Ctrl + = : Zoom In  |  Ctrl + - : Zoom Out  |  Ctrl + 0 : Reset Zoom\n"
            "  Ctrl + Scroll Wheel : Adjust Zoom\n"
            "  - / = : Scale Image Down / Up\n"
            "  Select Image + Backspace : Remove Selected Image\n"
            "  Spacebar  : Hold to Pan the Canvas\n"
            "  Ctrl + S : Save Moodboard to SVG"
        )
        shortcut_hints.setStyleSheet("font-size: 13px; padding: 3px; color: black;")
        self.layout.addWidget(shortcut_hints)
        self.layout.addLayout(button_row)

        # Shortcuts for zooming
        self.zoom_in_shortcut = QShortcut(QKeySequence("Ctrl+="), self)
        self.zoom_in_shortcut.activated.connect(self.zoom_in)
        self.zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        self.zoom_out_shortcut.activated.connect(self.zoom_out)
        self.reset_zoom_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self.reset_zoom_shortcut.activated.connect(self.reset_zoom)

        # Shortcuts for scaling the selected image
        self.scale_down_shortcut = QShortcut(QKeySequence("-"), self)
        self.scale_down_shortcut.activated.connect(self.scale_down)
        self.scale_up_shortcut = QShortcut(QKeySequence("="), self)
        self.scale_up_shortcut.activated.connect(self.scale_up)
        
        # Shortcuts for removing the selected image
        self.remove_image_shortcut = QShortcut(QKeySequence(Qt.Key_Backspace), self)
        self.remove_image_shortcut.activated.connect(self.remove_image)

        self.save_shortcut = QShortcut(QKeySequence("Ctrl+s"), self)
        self.save_shortcut.activated.connect(self.save_moodboard)

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
        if self.selected_item:
            self.selected_item.scale_image(0.9)

    def scale_up(self):
        if self.selected_item:
            self.selected_item.scale_image(1.1) 
            
    def remove_image(self):
        if self.selected_item:
            self.scene.removeItem(self.selected_item)
            self.selected_item = None #make sure no accidentall issues with selection

    def zoom_in(self):
        self.view.scale(1.2, 1.2) 

    def zoom_out(self):
        self.view.scale(0.8, 0.8)

    def reset_zoom(self):
        self.view.resetTransform()
    
    def clear_board(self):
        self.scene.clear()
        self.moodboard_items.clear()
        self.selected_item = None
    
    def save_moodboard(self):
        # Open Save As dialogue
        file_dialog = QFileDialog()
        file_dialog.setDefaultSuffix("svg")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("SVG Files (*.svg)")
        file_dialog.setWindowTitle("Save Moodboard As")
        
        # Suggest a default filename
        file_dialog.selectFile("moodboard.svg")
        
        if file_dialog.exec_():
            svg_file = file_dialog.selectedFiles()[0]
            try:
                with open(svg_file, "w") as f:
                    f.write(self.scene_to_svg())
                QMessageBox.information(self, "Success", 
                                    f"Moodboard successfully saved to:\n{svg_file}")
                
                # Open folder when success
                if sys.platform == 'win32':
                    subprocess.run(f'explorer /select,"{os.path.normpath(svg_file)}"', shell=True)
                elif sys.platform == 'darwin':
                    subprocess.run(['open', '-R', svg_file])
                else:  # Linux
                    subprocess.run(['xdg-open', os.path.dirname(svg_file)])
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                f"Failed to save moodboard:\n{str(e)}")

    def add_images_to_scene(self, image_paths):
        last_width_pos = 0
        
        for idx, image_path in enumerate(image_paths):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull() and image_path not in self.moodboard_items:
                last_width_pos += pixmap.width()
                resizable_item = ResizablePixmapItem(pixmap)
                resizable_item.setPos(last_width_pos + 50, 0)
                self.scene.addItem(resizable_item)
                resizable_item.setFlag(QGraphicsItem.ItemIsSelectable, True)
                resizable_item.mousePressEvent = lambda event, item=resizable_item: self.select_item(item)
                self.moodboard_items.append(image_path)


    def deselect_all_items(self):
        self.selected_item = None

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

    def closeEvent(self, event):
        self.scene.clear()
        self.moodboard_items.clear()
        self.selected_item = None


# Main function
def main():

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    device = "cpu"

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec_())

# Entry point
if __name__ == "__main__":
    main()