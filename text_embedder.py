import torch
import clip
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QListWidget, QLineEdit, QPushButton, QLabel, QMessageBox,
                            QFileDialog)
from PyQt5.QtCore import Qt

class TextEmbeddingEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Embedding Manager")
        self.setGeometry(100, 100, 600, 500)
        
        # Initialize CLIP model
        self.device = "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.text_features_cache = {}
        self.text_features_dict = {}
        self.current_file = None
        
        self.init_ui()
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title = QLabel("Text Embedding Manager")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        
        # File operations
        file_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Embeddings...")
        load_btn.clicked.connect(self.load_embeddings_dialog)
        file_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Save Embeddings...")
        save_btn.clicked.connect(self.save_embeddings_dialog)
        file_layout.addWidget(save_btn)
        
        save_as_btn = QPushButton("Save As...")
        save_as_btn.clicked.connect(self.save_embeddings_as_dialog)
        file_layout.addWidget(save_as_btn)
        
        layout.addLayout(file_layout)
        
        # List of current prompts
        self.prompt_list = QListWidget()
        self.prompt_list.setSelectionMode(QListWidget.SingleSelection)
        self.update_prompt_list()
        layout.addWidget(QLabel("Current Prompts:"))
        layout.addWidget(self.prompt_list)
        
        # Add/Edit section
        add_edit_layout = QHBoxLayout()
        
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter new prompt")
        add_edit_layout.addWidget(self.prompt_input)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_prompt)
        add_edit_layout.addWidget(add_btn)
        
        edit_btn = QPushButton("Update Selected")
        edit_btn.clicked.connect(self.update_prompt)
        add_edit_layout.addWidget(edit_btn)
        
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_prompt)
        add_edit_layout.addWidget(remove_btn)
        
        layout.addLayout(add_edit_layout)
        
        # Bottom buttons
        btn_layout = QHBoxLayout()
        
        process_btn = QPushButton("Process All")
        process_btn.clicked.connect(self.process_all)
        btn_layout.addWidget(process_btn)
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self.clear_all)
        btn_layout.addWidget(clear_btn)
        
        layout.addLayout(btn_layout)
        
        # Status bar
        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet("color: gray;")
        layout.addWidget(self.status_bar)
        
    def update_prompt_list(self):
        self.prompt_list.clear()
        self.prompt_list.addItems(sorted(self.text_features_dict.keys()))
        
    def add_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a prompt")
            return
            
        if prompt in self.text_features_dict:
            QMessageBox.warning(self, "Warning", "This prompt already exists")
            return
            
        self.status_bar.setText(f"Processing: {prompt}")
        QApplication.processEvents()  # Update UI
        
        try:
            # Check cache first
            if prompt in self.text_features_cache:
                features = self.text_features_cache[prompt]
            else:
                # Process new prompt
                text_tokenized = clip.tokenize([prompt]).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_text(text_tokenized)
                features = features / features.norm(dim=-1, keepdim=True)
                self.text_features_cache[prompt] = features
                
            self.text_features_dict[prompt] = features
            self.update_prompt_list()
            self.prompt_input.clear()
            self.status_bar.setText(f"Added: {prompt}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process prompt: {str(e)}")
            self.status_bar.setText("Error processing prompt")
        
    def update_prompt(self):
        selected = self.prompt_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a prompt to update")
            return
            
        old_prompt = selected.text()
        new_prompt = self.prompt_input.text().strip()
        
        if not new_prompt:
            QMessageBox.warning(self, "Warning", "Please enter a new prompt")
            return
            
        if new_prompt == old_prompt:
            return
            
        if new_prompt in self.text_features_dict:
            QMessageBox.warning(self, "Warning", "This prompt already exists")
            return
            
        # Get the existing features
        features = self.text_features_dict[old_prompt]
        
        # Update the dictionaries
        del self.text_features_dict[old_prompt]
        self.text_features_dict[new_prompt] = features
        
        # Update cache if needed
        if old_prompt in self.text_features_cache:
            self.text_features_cache[new_prompt] = self.text_features_cache[old_prompt]
            del self.text_features_cache[old_prompt]
        
        self.update_prompt_list()
        self.prompt_input.clear()
        self.status_bar.setText(f"Updated '{old_prompt}' to '{new_prompt}'")
        
    def remove_prompt(self):
        selected = self.prompt_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a prompt to remove")
            return
            
        prompt = selected.text()
        reply = QMessageBox.question(self, 'Confirm', 
                                    f"Remove prompt '{prompt}'?", 
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            del self.text_features_dict[prompt]
            if prompt in self.text_features_cache:
                del self.text_features_cache[prompt]
            self.update_prompt_list()
            self.status_bar.setText(f"Removed: {prompt}")
    
    def load_embeddings_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Embeddings File", "", 
            "PyTorch Files (*.pt);;All Files (*)", 
            options=options)
        
        if file_name:
            self.load_embeddings(file_name)
    
    def load_embeddings(self, file_path):
        try:
            self.text_features_dict = torch.load(file_path, map_location=torch.device(self.device))
            self.text_features_cache = self.text_features_dict.copy()
            self.current_file = file_path
            self.update_prompt_list()
            self.status_bar.setText(f"Loaded embeddings from {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load embeddings: {str(e)}")
            self.status_bar.setText("Error loading embeddings")
    
    def save_embeddings_dialog(self):
        if self.current_file:
            self.save_embeddings(self.current_file)
        else:
            self.save_embeddings_as_dialog()
    
    def save_embeddings_as_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Embeddings As", "", 
            "PyTorch Files (*.pt);;All Files (*)", 
            options=options)
        
        if file_name:
            # Ensure .pt extension
            if not file_name.endswith('.pt'):
                file_name += '.pt'
            self.save_embeddings(file_name)
            self.current_file = file_name
    
    def save_embeddings(self, file_path):
        try:
            torch.save(self.text_features_dict, file_path)
            self.status_bar.setText(f"Embeddings saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save embeddings: {str(e)}")
            self.status_bar.setText("Error saving embeddings")
        
    def process_all(self):
        """Re-process all prompts to ensure they're in the cache"""
        count = len(self.text_features_dict)
        if count == 0:
            return
            
        self.status_bar.setText(f"Processing {count} prompts...")
        QApplication.processEvents()
        
        try:
            # Batch process all prompts not in cache
            prompts_to_process = [p for p in self.text_features_dict.keys() 
                                if p not in self.text_features_cache]
            
            if prompts_to_process:
                # Tokenize all prompts at once
                text_tokenized = clip.tokenize(prompts_to_process).to(self.device)
                
                # Process in batches if large number
                batch_size = 32
                features_list = []
                
                for i in range(0, len(text_tokenized), batch_size):
                    batch = text_tokenized[i:i+batch_size]
                    with torch.no_grad():
                        features = self.model.encode_text(batch)
                    features = features / features.norm(dim=-1, keepdim=True)
                    features_list.append(features)
                
                # Combine all features
                all_features = torch.cat(features_list, dim=0)
                
                # Update cache
                for prompt, features in zip(prompts_to_process, all_features):
                    self.text_features_cache[prompt] = features.unsqueeze(0)  # Keep batch dim
                    self.text_features_dict[prompt] = features.unsqueeze(0)
            
            self.status_bar.setText(f"Processed {len(prompts_to_process)} prompts")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process prompts: {str(e)}")
            self.status_bar.setText("Error processing prompts")
    
    def clear_all(self):
        if not self.text_features_dict:
            return
            
        reply = QMessageBox.question(self, 'Confirm', 
                                    "Clear all prompts and embeddings?", 
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.text_features_dict = {}
            self.text_features_cache = {}
            self.current_file = None
            self.update_prompt_list()
            self.status_bar.setText("Cleared all prompts")

if __name__ == "__main__":
    app = QApplication([])
    window = TextEmbeddingEditor()
    window.show()
    app.exec_()
    input("ap")