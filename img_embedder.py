import torch
import clip
import os
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QWidget, 
                            QPushButton, QLabel, QFileDialog, QProgressBar, 
                            QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class EmbeddingWorker(QThread):
    progress_updated = pyqtSignal(int, int, str)  # current, total, status
    finished = pyqtSignal(dict, str)  # embeddings_dict, folder_path
    error_occurred = pyqtSignal(str)

    def __init__(self, folder_path, device, model, preprocess, mode="update"):
        super().__init__()
        self.folder_path = folder_path
        self.device = device
        self.model = model
        self.preprocess = preprocess
        self.mode = mode  # "update" or "clean"
        self.canceled = False

    def run(self):
        try:
            # Check for existing embeddings
            embeddings_path = os.path.join(self.folder_path, "img.pt")
            if os.path.exists(embeddings_path):
                image_features_dict = torch.load(embeddings_path, map_location=self.device)
                self.progress_updated.emit(0, 0, f"Loaded {len(image_features_dict)} existing embeddings")
            else:
                image_features_dict = {}
                self.progress_updated.emit(0, 0, "No existing embeddings found")
            
            # Get all image files
            image_files = set(f for f in os.listdir(self.folder_path) 
                          if f.lower().endswith(("png", "jpg", "jpeg")))
            
            if not image_files:
                self.error_occurred.emit("No image files found in selected folder")
                return

            if self.mode == "clean":
                # Clean mode: remove embeddings without corresponding images
                original_count = len(image_features_dict)
                to_remove = [f for f in image_features_dict if f not in image_files]
                
                for f in to_remove:
                    del image_features_dict[f]
                
                if to_remove:
                    self.progress_updated.emit(0, 0, 
                        f"Removed {len(to_remove)} orphaned embeddings ({original_count} → {len(image_features_dict)})")
                else:
                    self.progress_updated.emit(0, 0, "No orphaned embeddings found")
                
                # Save cleaned embeddings
                torch.save(image_features_dict, embeddings_path)
                self.finished.emit(image_features_dict, self.folder_path)
                return

            # Update mode: find new images to process
            new_images = [f for f in image_files if f not in image_features_dict]
            total_images = len(new_images)
            
            if not new_images:
                self.progress_updated.emit(0, 0, "All images already embedded")
                self.finished.emit(image_features_dict, self.folder_path)
                return

            # Process new images
            processed = 0
            for filename in new_images:
                if self.canceled:
                    return
                
                try:
                    image_path = os.path.join(self.folder_path, filename)
                    image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        features = self.model.encode_image(image)
                    image_features_dict[filename] = features / features.norm(dim=-1, keepdim=True)
                    
                    processed += 1
                    status = f"Processing {processed}/{total_images}: {filename[:20]}..."
                    self.progress_updated.emit(processed, total_images, status)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue

            if not self.canceled:
                # Save updated embeddings
                torch.save(image_features_dict, embeddings_path)
                self.finished.emit(image_features_dict, self.folder_path)
                
        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self.canceled = True

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Embedder")
        self.setGeometry(100, 100, 500, 250)
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.init_ui()
        self.worker = None

    def init_ui(self):
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Folder selection
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("font-size: 12px;")
        self.layout.addWidget(self.folder_label)

        self.select_button = QPushButton("Select Image Folder")
        self.select_button.clicked.connect(self.select_folder)
        self.layout.addWidget(self.select_button)

        # Options
        self.options_widget = QWidget()
        self.options_layout = QVBoxLayout(self.options_widget)
        
        self.clean_checkbox = QCheckBox("Clean orphaned embeddings (remove entries without images)")
        self.update_checkbox = QCheckBox("Update with new images (default)")
        self.update_checkbox.setChecked(True)
        
        self.options_layout.addWidget(self.update_checkbox)
        self.options_layout.addWidget(self.clean_checkbox)
        self.layout.addWidget(self.options_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

        # Start button
        self.start_button = QPushButton("Process Embeddings")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        self.layout.addWidget(self.start_button)

        # Device info
        device_text = f"Using: {self.device.upper()}"
        if self.device == "cuda":
            device_text += f" ({torch.cuda.get_device_name(0)})"
        device_label = QLabel(device_text)
        device_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(device_label)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if folder:
            self.folder_path = folder
            self.folder_label.setText(f"Selected: {folder}")
            self.folder_label.setToolTip(folder)
            self.start_button.setEnabled(True)
            self.status_label.setText("Ready to process embeddings")
            
            # Check for existing embeddings
            embeddings_path = os.path.join(folder, "img.pt")
            if os.path.exists(embeddings_path):
                try:
                    embeddings = torch.load(embeddings_path)
                    self.status_label.setText(
                        f"Found {len(embeddings)} existing embeddings. Select an action below.")
                except:
                    self.status_label.setText("Found corrupted embeddings file")

    def start_processing(self):
        if not hasattr(self, 'folder_path'):
            return

        mode = "clean" if self.clean_checkbox.isChecked() else "update"
        
        # Setup UI for processing
        self.select_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting processing...")

        # Start worker thread
        self.worker = EmbeddingWorker(
            self.folder_path, 
            self.device, 
            self.model, 
            self.preprocess,
            mode
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.processing_complete)
        self.worker.error_occurred.connect(self.processing_error)
        self.worker.start()

    def update_progress(self, current, total, status):
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)
        self.status_label.setText(status)

    def processing_complete(self, embeddings_dict, folder_path):
        # Reset UI
        self.progress_bar.setVisible(False)
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)
        
        action = "Cleaned" if self.clean_checkbox.isChecked() else "Updated"
        self.status_label.setText(f"{action} embeddings. Total: {len(embeddings_dict)}")
        
        QMessageBox.information(self, "Success", 
            f"Successfully {action.lower()} embeddings\n"
            f"Total embeddings: {len(embeddings_dict)}\n"
            f"Saved to: {os.path.join(folder_path, 'img.pt')}")

    def processing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.status_label.setText("Error occurred")
        
        QMessageBox.critical(self, "Error", error_msg)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait()
        event.accept()

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()