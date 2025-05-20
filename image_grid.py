import os
import random
import shutil
import subprocess
import sys
from PyQt5.QtCore import Qt, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                            QLabel, QLineEdit, QPushButton, QScrollArea, 
                            QMessageBox, QMenu)

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
        
        self.temp_text=""
        
        # Search input box
        search_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type prompt or drag image here")
        self.input_box.setStyleSheet("font-size: 18px; min-height: 40px; padding: 5px;")
        self.input_box.setAcceptDrops(True)
        self.input_box.dragEnterEvent = self.dragEnterEvent
        self.input_box.dragMoveEvent = self.dragMoveEvent
        self.input_box.dropEvent = self.dropEvent
        self.input_box.dragLeaveEvent = self.dragLeaveEvent
        self.input_box.returnPressed.connect(self.perform_search)
        
        self.search_button = QPushButton("Search")
        self.search_button.setStyleSheet("font-size: 18px; min-height: 40px; padding: 5px;")
        self.search_button.clicked.connect(self.perform_search)
        
        search_layout.addWidget(self.input_box)
        search_layout.addWidget(self.search_button)
        self.layout.addLayout(search_layout)
        
        self.progress_signal = ProgressSignal()
        self.progress_signal.progress_updated.connect(self.update_progress)
        self.progress_signal.finished.connect(self.on_similarity_complete)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        
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
        self.shuffle_button = create_tool_button(self, "Shuffle", self.shuffle_images)
        self.copy_button = create_tool_button(self, "Copy to Folder", self.copy_selected_images)
        self.moodboard_button = create_tool_button(self, "Add to Moodboard", self.open_moodboard)
        self.select_all_button = create_tool_button(self, "Select All", self.select_all_images)
        self.clear_selection_button = create_tool_button(self, "Clear Selection", self.clear_selection)

        # Add buttons to toolbar
        for btn in [self.shuffle_button, self.copy_button, self.moodboard_button, 
                   self.select_all_button, self.clear_selection_button]:
            toolbar.addWidget(btn)

        toolbar.addStretch()
        self.layout.addLayout(toolbar)

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

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.temp_text=self.input_box.text()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.input_box.setStyleSheet("background-color: lightgreen; font-size: 18px; min-height: 40px; padding: 5px;")
        self.input_box.setText("Drop Image Here to Search")
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.input_box.setStyleSheet("background-color: white; font-size: 18px; min-height: 40px; padding: 5px;")
        self.input_box.setText(self.temp_text)

    def dropEvent(self, event):
        self.input_box.setText(self.temp_text)
        self.input_box.setStyleSheet("background-color: white; font-size: 18px; min-height: 40px; padding: 5px;")
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.input_box.setText(self.temp_text)
                show_similar_images(self, file_path)
        event.acceptProposedAction()

    def update_progress(self, current, total):
        percent = int((current / total) * 100)
        self.progress_bar.setValue(percent)

    def on_similarity_complete(self, results):
        self.progress_bar.setVisible(False)
        self.search_button.setEnabled(True)
        
        # Create a new tab for the results
        self.main_window.add_results_tab(self.input_box.text(), results)

    def perform_search(self):
        """Handle search from within the ImageGridWindow"""
        search_text = self.input_box.text().strip()
        perform_search(self,search_text,self.image_features_dict,self.text_features_dict,self.progress_signal)

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

    def delete_image(self, img_name):
        image_path = os.path.join(self.image_folder, img_name)
        
        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image not found:\n{image_path}")
            return
        
        # # Confirm deletion
        # reply = QMessageBox.question(
        #     self,
        #     "Confirm Deletion",
        #     f"Are you sure you want to permanently delete:\n{img_name}?",
        #     QMessageBox.Yes | QMessageBox.No,
        #     QMessageBox.No
        # )
        
        if True:
            try:
                # Delete the image file
                os.remove(image_path)
                
                # Remove from features dict if it exists
                if img_name in self.image_features_dict:
                    del self.image_features_dict[img_name]
                    
                    # Save the updated embeddings back to img.pt
                    embeddings_path = os.path.join(self.image_folder, "img.pt")
                    torch.save(self.image_features_dict, embeddings_path)
                
                # Remove from current results if present
                if self.match_results:
                    self.match_results = [(name, score) for name, score in self.match_results if name != img_name]
                
                # Remove from selection if selected
                if img_name in self.selected_images:
                    self.selected_images.remove(img_name)
                    self.update_selection_count()
                
                # Refresh the grid
                self.create_image_grid()
                
                QMessageBox.information(self, "Success", f"Deleted: {img_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete image:\n{str(e)}")
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
        delete_action = menu.addAction("Delete Image from Folder")
        menu.addSeparator()
        select_all = menu.addAction("Select All")
        clear_selection = menu.addAction("Clear Selection")
        
        action = menu.exec_(QCursor.pos())
        
        if action == find_similar:
            show_similar_images(self, img_name)
        elif action == open_action:
            self.open_image(img_name)
        elif action == delete_action:
            self.delete_image(img_name)
        elif action == select_all:
            self.select_all_images()
        elif action == clear_selection:
            self.clear_selection()


    def open_moodboard(self):
        """Open selected images in moodboard"""
        if not self.selected_images:
            QMessageBox.warning(self, "No Selection", "Please select images to add to moodboard.")
            return
            
        image_paths = [os.path.join(self.image_folder, img_name) 
                      for img_name in self.selected_images]
        self.main_window.open_moodboard(image_paths)

class ClickableImageLabel(QLabel):
    """Custom QLabel that handles click events, selection state, and dragging"""
    clicked = pyqtSignal(str, object)  # (image_name, label)
    right_clicked = pyqtSignal(QPointF, str)  # (position, image_name)

    def __init__(self, pixmap, image_name):
        super().__init__()
        self.image_name = image_name
        self.setPixmap(pixmap)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid transparent;")
        self.set_selected(False)
        self._drag_start_pos = None

    def set_selected(self, selected):
        """Update visual selection state"""
        self.selected = selected
        border = "2px solid blue" if selected else "2px solid transparent"
        self.setStyleSheet(f"border: {border};")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
            self.clicked.emit(self.image_name, self)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(event.pos(), self.image_name)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            if self._drag_start_pos and (event.pos() - self._drag_start_pos).manhattanLength() > QApplication.startDragDistance():
                self.start_drag()

    def start_drag(self):
        drag = QDrag(self)
        mime_data = QMimeData()

        # Convert image path to a file URL and set it
        url = QUrl.fromLocalFile(self.image_name)
        mime_data.setUrls([url])  # This allows drop targets to treat it like a real file

        drag.setMimeData(mime_data)

        # Optional: show image thumbnail while dragging
        drag.setPixmap(self.pixmap().scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        drag.setHotSpot(QPoint(32, 32))  # Center of thumbnail

        drag.exec_(Qt.CopyAction | Qt.MoveAction)
