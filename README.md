# Video and Image Segmentation Processor

## Overview
A PyQt5-based desktop application for advanced image and video segmentation using K-means clustering algorithm.

## Features
- Dual-mode processing (Video and Image segmentation)
- K-means clustering for image segmentation
- Real-time progress tracking
- Flexible save directory selection
- Error handling and logging

## Requirements
- Python 3.7+
- PyQt5
- OpenCV
- NumPy
- Torch

## Installation
```bash
pip install PyQt5 opencv-python numpy torch
```

## Usage
1. Launch the application
2. Select Video/Image Segmentation tab
3. Click "Load Video" or "Load Images"
4. Choose save directory
5. View segmentation results in real-time

## Key Components
- Advanced segmentation using K-means
- Multi-threaded processing
- Comprehensive error handling
- Intuitive GUI design

## Segmentation Technique
Uses K-means clustering to:
- Resize images
- Cluster pixel colors
- Create segmented images with distinct boundaries

## Supported Formats
- Videos: .mp4, .avi, .mov
- Images: .png, .jpg, .jpeg, .bmp, .tiff

## Error Handling
- Global exception management
- Detailed logging
- User-friendly error messages

## License
MIT License
