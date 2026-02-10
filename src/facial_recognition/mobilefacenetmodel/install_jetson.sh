#!/bin/bash

# Installation script for Face Recognition on NVIDIA Jetson Nano
# This script installs all required dependencies

echo "==================================================="
echo "Face Recognition Setup for NVIDIA Jetson Nano"
echo "==================================================="

# Update system
echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Python dependencies
echo "Installing Python and pip..."
sudo apt-get install -y python3-pip python3-dev

# Install OpenCV dependencies
echo "Installing OpenCV dependencies..."
sudo apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

# Install other required libraries
echo "Installing system libraries..."
sudo apt-get install -y \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install Python packages
echo "Installing Python packages..."
pip3 install --upgrade setuptools wheel

# Install numpy (critical for Jetson)
echo "Installing numpy..."
pip3 install numpy

# Install scipy
echo "Installing scipy..."
pip3 install scipy

# Install onnxruntime-gpu for Jetson
echo "Installing onnxruntime for GPU acceleration..."
# For Jetson Nano with JetPack 4.6+
pip3 install onnxruntime-gpu

# Install InsightFace
echo "Installing InsightFace..."
pip3 install insightface

# Install additional dependencies
pip3 install scikit-learn

echo ""
echo "==================================================="
echo "Installation complete!"
echo "==================================================="
echo ""
echo "Next steps:"
echo "1. Organize your training images in folders:"
echo "   training_data/"
echo "     ├── person1/"
echo "     ├── person2/"
echo "     ├── person3/"
echo "     └── person4/"
echo ""
echo "2. Run: python3 face_recognition_pipeline.py"
echo ""
echo "Note: First run will download the model (~5MB)"
echo "==================================================="
