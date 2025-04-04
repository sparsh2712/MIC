import numpy as np
import h5py
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import cv2

def load_mat_file(file_path):
    """
    Load data from a .mat file.
    
    Args:
        file_path: Path to .mat file
        
    Returns:
        data: Dictionary of loaded data
    """
    try:
        data = {}
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][()]
        return data
    except:
        # Fall back to scipy.io for older MATLAB files
        return sio.loadmat(file_path)

def save_results_to_txt(results, file_path):
    """
    Save results to a text file.
    
    Args:
        results: Dictionary of results
        file_path: Path to save the file
    """
    with open(file_path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def ensure_data_extracted(data_dir):
    """
    Ensure that all zip files in the data directory are extracted.
    
    Args:
        data_dir: Path to data directory
    """
    # Check if anatomicalSegmentations directory exists
    seg_dir = os.path.join(data_dir, 'anatomicalSegmentations')
    if not os.path.exists(seg_dir):
        # Extract the anatomicalSegmentations.zip file
        zip_path = os.path.join(data_dir, 'anatomicalSegmentations.zip')
        if os.path.exists(zip_path):
            extract_zip(zip_path, data_dir)
    
    # Check if anatomicalSegmentationsDistorted directory exists
    distorted_dir = os.path.join(data_dir, 'anatomicalSegmentationsDistorted')
    if not os.path.exists(distorted_dir):
        # Extract the anatomicalSegmentationsDistorted.zip file
        zip_path = os.path.join(data_dir, 'anatomicalSegmentationsDistorted.zip')
        if os.path.exists(zip_path):
            extract_zip(zip_path, data_dir)

def load_segmentation_images(dir_path):
    """
    Load segmentation images from a directory.
    
    Args:
        dir_path: Path to directory containing images
        
    Returns:
        images: List of segmentation images
    """
    images = []
    image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    
    for file_name in sorted(image_files):
        file_path = os.path.join(dir_path, file_name)
        img = np.array(Image.open(file_path).convert('L')) / 255.0  # Normalize to [0, 1]
        images.append(img)
    
    return images

def extract_boundary_points(segmentation, n_points=56):
    """
    Extract boundary points from a segmentation image.
    
    Args:
        segmentation: (H, W) array of segmentation
        n_points: Number of points to extract per boundary
        
    Returns:
        points: (N, 2) array of boundary points
    """
    # Threshold the image
    binary = (segmentation > 0.5).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) == 0:
        print("Warning: No contours found in segmentation")
        return np.zeros((0, 2))
    
    # Extract points from the largest contour
    points = []
    
    # For cardiac shapes, we might have inner and outer boundaries
    if len(contours) >= 2:
        # Outer boundary
        outer_contour = contours[0]
        outer_points = sample_contour(outer_contour, n_points // 2)
        points.extend(outer_points)
        
        # Inner boundary
        inner_contour = contours[1]
        inner_points = sample_contour(inner_contour, n_points // 2)
        points.extend(inner_points)
    else:
        # Just one boundary
        points = sample_contour(contours[0], n_points)
    
    # Convert to numpy array
    points = np.array(points)
    
    # If we didn't get enough points, duplicate some
    if len(points) < n_points:
        print(f"Warning: Only found {len(points)} points, padding to {n_points}")
        # Create indices for the points we have, repeating as necessary
        indices = np.mod(np.arange(n_points), len(points))
        points = points[indices]
    
    # If we got too many points, subsample
    if len(points) > n_points:
        # Sample evenly
        indices = np.linspace(0, len(points) - 1, n_points, dtype=int)
        points = points[indices]
    
    return points

def sample_contour(contour, n_points):
    """
    Sample n_points from a contour.
    
    Args:
        contour: OpenCV contour
        n_points: Number of points to sample
        
    Returns:
        points: List of sampled points
    """
    contour = contour.squeeze()
    
    if len(contour) <= n_points:
        return contour
    
    # Sample points uniformly
    indices = np.linspace(0, len(contour) - 1, n_points, dtype=int)
    return contour[indices]

def create_directory(dir_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        dir_path: Path to directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)