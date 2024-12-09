import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QTextEdit, QMessageBox, 
    QProgressBar, QGridLayout, QTabWidget
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QCoreApplication

class ImageSegmentationWorker(QThread):
    update_original_image = pyqtSignal(np.ndarray)
    update_segmented_image = pyqtSignal(np.ndarray)
    update_log = pyqtSignal(str)
    processing_complete = pyqtSignal(bool)
    progress_update = pyqtSignal(int)

    def __init__(self, image_paths, save_directory=None):
        super().__init__()
        self.image_paths = image_paths
        self.save_directory = save_directory
        self.is_running = True

    def create_save_directory(self):
        try:
            if not self.save_directory:
                # Default to directory of first image
                first_image_dir = os.path.dirname(self.image_paths[0])
                self.save_directory = os.path.join(first_image_dir, "image_segmentation")
            
            # Create directories for original and segmented images
            os.makedirs(os.path.join(self.save_directory, 'original'), exist_ok=True)
            os.makedirs(os.path.join(self.save_directory, 'segmented'), exist_ok=True)
            
            return self.save_directory
        except OSError as e:
            self.update_log.emit(f"Error creating save directory: {e}")
            return None

    def run(self):
        try:
            # Validate and create save directory
            save_dir = self.create_save_directory()
            if not save_dir:
                raise ValueError("Could not create save directory")
            
            self.update_log.emit(f"Saving images to: {save_dir}")

            # Process images
            for idx, image_path in enumerate(self.image_paths):
                if not self.is_running:
                    break

                try:
                    # Read image with error handling
                    if not os.path.exists(image_path):
                        self.update_log.emit(f"Image not found: {image_path}")
                        continue

                    frame = cv2.imread(image_path)
                    if frame is None:
                        self.update_log.emit(f"Could not read image: {image_path}")
                        continue

                    # Emit and save original image
                    self.update_original_image.emit(frame.copy())
                    self.safe_save_image(frame, 'original', idx)

                    # Perform segmentation
                    segmented_image = self.perform_segmentation(frame)
                    
                    # Emit and save segmented image
                    self.update_segmented_image.emit(segmented_image)
                    self.safe_save_image(segmented_image, 'segmented', idx)

                    # Update progress
                    progress = min(100, int(((idx + 1) / len(self.image_paths)) * 100))
                    self.progress_update.emit(progress)
                    self.update_log.emit(f"Processed image {idx + 1}/{len(self.image_paths)}: {os.path.basename(image_path)}")

                    # Yield control to prevent GUI freezing
                    QCoreApplication.processEvents()

                except Exception as image_error:
                    self.update_log.emit(f"Error processing image {image_path}: {image_error}")

            # Emit completion status
            if self.is_running:
                self.update_log.emit("Image processing completed successfully")
                self.processing_complete.emit(True)
            else:
                self.processing_complete.emit(False)

        except Exception as e:
            self.update_log.emit(f"Critical error during image processing: {str(e)}")
            self.processing_complete.emit(False)

    def safe_save_image(self, image, image_type, image_number):
        try:
            if image is None or image.size == 0:
                return

            # Construct filename with robust path handling
            filename = os.path.join(
                self.save_directory, 
                image_type, 
                f"{image_type}_image_{image_number:04d}.png"
            )
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            # Save image with error checking
            if not cv2.imwrite(filename, image):
                self.update_log.emit(f"Failed to save {image_type} image {image_number}")
        except Exception as e:
            self.update_log.emit(f"Error saving {image_type} image {image_number}: {e}")

    def perform_segmentation(self, frame):
        try:
            # Resize image with scale factor
            scale_factor = 0.5
            resized = cv2.resize(
                frame, 
                None, 
                fx=scale_factor, 
                fy=scale_factor, 
                interpolation=cv2.INTER_AREA
            )

            # Reshape image for clustering
            pixels = resized.reshape((-1, 3)).astype(np.float32)

            # K-means clustering
            k = min(4, pixels.shape[0])
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(
                pixels, 
                k, 
                None, 
                criteria, 
                10, 
                cv2.KMEANS_RANDOM_CENTERS
            )
        
            # Convert centers to uint8
            centers = np.uint8(centers)
        
            # Reshape labels back to image dimensions
            labels = labels.reshape(resized.shape[:2])
        
            # Create segmented image with distinct boundaries
            segmented = np.zeros_like(resized)
        
            # Assign colors and add boundaries
            for i in range(k):
                # Create mask for this cluster
                mask = labels == i
            
                # Assign cluster color
                segmented[mask] = centers[i]
            
                # Add boundary effect
                # Dilate the mask to create boundary lines
                kernel = np.ones((3,3), np.uint8)
                boundary_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                boundary_mask = boundary_mask ^ mask.astype(np.uint8)
            
                # Mark boundaries with a distinct color (black in this case)
                if np.any(boundary_mask):
                    segmented[boundary_mask.astype(bool)] = [0, 0, 0]  # Black boundary
        
            # Resize back to original dimensions
            segmented = cv2.resize(
                segmented, 
                (frame.shape[1], frame.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
            return segmented
    
        except Exception as e:
            print(f"Segmentation error: {e}")
            return frame.copy()

    def fallback_segmentation(self, frame):
        try:
            # Convert to grayscale, then back to color
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray_color
        except Exception:
            # Absolute last resort
            return frame.copy()

class VideoSegmentationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_segmentation_worker = None
        self.image_segmentation_worker = None
        self.initUI()
        self.setup_error_handling()

    def setup_error_handling(self):
        sys.excepthook = self.global_exception_handler

    def global_exception_handler(self, exc_type, exc_value, exc_traceback):
        error_message = f"Unhandled Exception: {exc_value}"
        QMessageBox.critical(self, "Critical Error", error_message)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def initUI(self):
        # Set window title and size
        self.setWindowTitle('Video and Image Segmentation')
        self.setGeometry(100, 100, 1000, 800)

        # Create main widget and central layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create tab widget for video and image segmentation
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)

        # Video Segmentation Tab
        video_tab = QWidget()
        video_layout = QVBoxLayout()
        video_tab.setLayout(video_layout)

        # Video buttons layout
        video_button_layout = QHBoxLayout()
        self.load_video_btn = QPushButton('Load Video')
        self.cancel_video_btn = QPushButton('Cancel')
    
        self.load_video_btn.clicked.connect(self.load_video)
        self.cancel_video_btn.clicked.connect(self.cancel_video_processing)
    
        self.cancel_video_btn.setEnabled(False)
    
        video_button_layout.addWidget(self.load_video_btn)
        video_button_layout.addWidget(self.cancel_video_btn)

        # Video frame display
        video_frame_layout = QHBoxLayout()
        self.video_original_frame_label = QLabel('Original Frame')
        self.video_segmented_frame_label = QLabel('Segmented Frame')
    
        # Set fixed size or minimum size for frame labels to ensure visibility
        self.video_original_frame_label.setMinimumSize(400, 300)
        self.video_segmented_frame_label.setMinimumSize(400, 300)
    
        # Center align and set border for labels
        self.video_original_frame_label.setAlignment(Qt.AlignCenter)
        self.video_segmented_frame_label.setAlignment(Qt.AlignCenter)
        self.video_original_frame_label.setStyleSheet("border: 1px solid gray;")
        self.video_segmented_frame_label.setStyleSheet("border: 1px solid gray;")
    
        video_frame_layout.addWidget(self.video_original_frame_label)
        video_frame_layout.addWidget(self.video_segmented_frame_label)

        # Video progress and log
        self.video_progress_bar = QProgressBar()
        self.video_log_display = QTextEdit()
        self.video_log_display.setReadOnly(True)

        # Add widgets to video layout
        video_layout.addLayout(video_button_layout)
        video_layout.addLayout(video_frame_layout)
        video_layout.addWidget(self.video_progress_bar)
        video_layout.addWidget(self.video_log_display)

        # Image Segmentation Tab
        image_tab = QWidget()
        image_layout = QVBoxLayout()
        image_tab.setLayout(image_layout)

        # Image buttons layout
        image_button_layout = QHBoxLayout()
        self.load_images_btn = QPushButton('Load Images')
        self.cancel_images_btn = QPushButton('Cancel')
    
        self.load_images_btn.clicked.connect(self.load_images)
        self.cancel_images_btn.clicked.connect(self.cancel_image_processing)
    
        self.cancel_images_btn.setEnabled(False)
    
        image_button_layout.addWidget(self.load_images_btn)
        image_button_layout.addWidget(self.cancel_images_btn)

        # Image frame display
        image_frame_layout = QHBoxLayout()
        self.image_original_label = QLabel('Original Image')
        self.image_segmented_label = QLabel('Segmented Image')
    
        # Set fixed size or minimum size for frame labels to ensure visibility
        self.image_original_label.setMinimumSize(400, 300)
        self.image_segmented_label.setMinimumSize(400, 300)
    
        # Center align and set border for labels
        self.image_original_label.setAlignment(Qt.AlignCenter)
        self.image_segmented_label.setAlignment(Qt.AlignCenter)
        self.image_original_label.setStyleSheet("border: 1px solid gray;")
        self.image_segmented_label.setStyleSheet("border: 1px solid gray;")
    
        image_frame_layout.addWidget(self.image_original_label)
        image_frame_layout.addWidget(self.image_segmented_label)

        # Image progress and log
        self.image_progress_bar = QProgressBar()
        self.image_log_display = QTextEdit()
        self.image_log_display.setReadOnly(True)

        # Add widgets to image layout
        image_layout.addLayout(image_button_layout)
        image_layout.addLayout(image_frame_layout)
        image_layout.addWidget(self.image_progress_bar)
        image_layout.addWidget(self.image_log_display)

        # Add tabs
        tab_widget.addTab(video_tab, "Video Segmentation")
        tab_widget.addTab(image_tab, "Image Segmentation")

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(
            self, 
            'Select Video File', 
            '', 
            'Video Files (*.mp4 *.avi *.mov)'
        )
        
        if video_path:
            # Reset UI
            self.reset_video_ui()

            # Ask for save directory
            save_directory = QFileDialog.getExistingDirectory(
                self, 
                'Select Save Directory for Frames'
            )

            # Create and start segmentation worker
            self.video_segmentation_worker = SegmentationWorker(
                video_path, 
                save_directory if save_directory else None
            )
            
            # Connect signals
            self.video_segmentation_worker.update_original_image.connect(self.display_video_original_frame)
            self.video_segmentation_worker.update_segmented_image.connect(self.display_video_segmented_frame)
            self.video_segmentation_worker.update_log.connect(self.update_video_log)
            self.video_segmentation_worker.progress_update.connect(self.update_video_progress)
            self.video_segmentation_worker.processing_complete.connect(self.on_video_processing_complete)
            self.video_segmentation_worker.processing_stopped.connect(self.on_video_processing_stopped)

            # Enable/Disable buttons
            self.load_video_btn.setEnabled(False)
            self.cancel_video_btn.setEnabled(True)

            # Start processing
            self.video_segmentation_worker.start()

    def load_images(self):
        image_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            'Select Image Files', 
            '', 
            'Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)'
        )
        
        if image_paths:
            # Reset UI
            self.reset_image_ui()

            # Ask for save directory
            save_directory = QFileDialog.getExistingDirectory(
                self, 
                'Select Save Directory for Segmented Images'
            )

            # Create and start image segmentation worker
            self.image_segmentation_worker = ImageSegmentationWorker(
                image_paths, 
                save_directory if save_directory else None
            )
            
            # Connect signals
            self.image_segmentation_worker.update_original_image.connect(self.display_image_original_frame)
            self.image_segmentation_worker.update_segmented_image.connect(self.display_image_segmented_frame)
            self.image_segmentation_worker.update_log.connect(self.update_image_log)
            self.image_segmentation_worker.progress_update.connect(self.update_image_progress)
            self.image_segmentation_worker.processing_complete.connect(self.on_image_processing_complete)

            # Enable/Disable buttons
            self.load_images_btn.setEnabled(False)
            self.cancel_images_btn.setEnabled(True)

            # Start processing
            self.image_segmentation_worker.start()

    def reset_video_ui(self):
        # Clear previous logs and progress
        self.video_log_display.clear()
        self.video_progress_bar.setValue(0)

        # Reset frame labels
        self.video_original_frame_label.clear()
        self.video_original_frame_label.setText('Original Frame')
        self.video_segmented_frame_label.clear()
        self.video_segmented_frame_label.setText('Segmented Frame')

    def reset_image_ui(self):
        # Clear previous logs and progress
        self.image_log_display.clear()
        self.image_progress_bar.setValue(0)

        # Reset image labels
        self.image_original_label.clear()
        self.image_original_label.setText('Original Image')
        self.image_segmented_label.clear()
        self.image_segmented_label.setText('Segmented Image')

    def cancel_video_processing(self):
        if self.video_segmentation_worker:
            # Stop the worker thread
            self.video_segmentation_worker.stop()
            
            # Disable cancel button
            self.cancel_video_btn.setEnabled(False)
            self.load_video_btn.setEnabled(True)

    def cancel_image_processing(self):
        if self.image_segmentation_worker:
            # Stop the worker thread
            self.image_segmentation_worker.is_running = False
            
            # Disable cancel button
            self.cancel_images_btn.setEnabled(False)
            self.load_images_btn.setEnabled(True)

    def on_video_processing_stopped(self):
        # Reset UI elements
        self.cancel_video_btn.setEnabled(False)
        self.load_video_btn.setEnabled(True)
        self.video_progress_bar.setValue(0)

    def on_video_processing_complete(self, success):
        # Re-enable load video button
        self.load_video_btn.setEnabled(True)
        self.cancel_video_btn.setEnabled(False)

        # Show message based on processing result
        if success:
            QMessageBox.information(self, 'Processing Complete', 'Video segmentation finished successfully!')
        else:
            # Only show error if not manually stopped
            if self.video_segmentation_worker and not self.video_segmentation_worker.is_running:
                QMessageBox.information(self, 'Processing Stopped', 'Video processing was cancelled by user.')
            else:
                QMessageBox.warning(self, 'Processing Failed', 'Video segmentation encountered an error.')

    def on_image_processing_complete(self, success):
        # Re-enable load images button
        self.load_images_btn.setEnabled(True)
        self.cancel_images_btn.setEnabled(False)

        # Show message based on processing result
        if success:
            QMessageBox.information(self, 'Processing Complete', 'Image segmentation finished successfully!')
        else:
            QMessageBox.warning(self, 'Processing Failed', 'Image segmentation encountered an error.')



    def display_video_original_frame(self, frame):
        self.display_frame_in_label(frame, self.video_original_frame_label)

    def display_video_segmented_frame(self, frame):
        self.display_frame_in_label(frame, self.video_segmented_frame_label)

    def display_image_original_frame(self, frame):
        self.display_frame_in_label(frame, self.image_original_label)

    def display_image_segmented_frame(self, frame):
        self.display_frame_in_label(frame, self.image_segmented_label)

    def display_frame_in_label(self, frame, label):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(
            rgb_frame.data, 
            w, 
            h, 
            bytes_per_line, 
            QImage.Format_RGB888
        )
        
        # Scale image
        pixmap = QPixmap.fromImage(qt_image).scaled(
            label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        label.setPixmap(pixmap)

    def update_video_log(self, message):
        self.video_log_display.append(message)

    def update_image_log(self, message):
        self.image_log_display.append(message)

    def update_video_progress(self, value):
        self.video_progress_bar.setValue(value)

    def update_image_progress(self, value):
        self.image_progress_bar.setValue(value)

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video and Image Segmentation Processor")
    
    # Enhanced error handling for the entire application
    try:
        window = VideoSegmentationGUI()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()




