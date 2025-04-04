import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import os

def plot_pointsets(pointsets, title='Pointsets', save_path=None, colors=None):
    """
    Plot multiple pointsets with random colors.
    
    Args:
        pointsets: List of (N, 2) arrays of 2D points
        title: Plot title
        save_path: Path to save the figure
        colors: List of colors (if None, random colors are generated)
    """
    plt.figure(figsize=(8, 8))
    
    if colors is None:
        # Generate random colors
        colors = np.random.rand(len(pointsets), 3)
    
    for i, points in enumerate(pointsets):
        plt.plot(points[:, 0], points[:, 1], 'o-', color=colors[i], markersize=4, linewidth=1)
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_mean_and_aligned(mean_shape, aligned_shapes, title='Mean and Aligned Shapes', save_path=None):
    """
    Plot mean shape and aligned shapes.
    
    Args:
        mean_shape: (N, 2) array of mean shape
        aligned_shapes: List of aligned (N, 2) arrays
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 8))
    
    # Plot aligned shapes
    colors = np.random.rand(len(aligned_shapes), 3)
    for i, shape in enumerate(aligned_shapes):
        plt.plot(shape[:, 0], shape[:, 1], '-', color=colors[i], alpha=0.3, linewidth=1)
    
    # Plot mean shape
    plt.plot(mean_shape[:, 0], mean_shape[:, 1], 'ro-', linewidth=2, markersize=4, label='Mean Shape')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_eigenvalues(eigenvalues, title='Eigenvalues', save_path=None):
    """
    Plot eigenvalues.
    
    Args:
        eigenvalues: Array of eigenvalues
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_mean_and_variations(mean_shape, aligned_shapes, variations, title='Mean and Variations', save_path=None):
    """
    Plot mean shape, aligned shapes, and shape variations.
    
    Args:
        mean_shape: (N, 2) array of mean shape
        aligned_shapes: List of aligned (N, 2) arrays
        variations: List of tuples (pos_var, neg_var) for each mode
        title: Plot title
        save_path: Path to save the figure
    """
    n_modes = len(variations)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 6))
    
    if n_modes == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot aligned shapes
        for shape in aligned_shapes:
            ax.plot(shape[:, 0], shape[:, 1], '-', color='lightgray', alpha=0.3, linewidth=1)
        
        # Plot mean shape
        ax.plot(mean_shape[:, 0], mean_shape[:, 1], 'k-', linewidth=2, label='Mean')
        
        # Plot variations
        pos_var, neg_var = variations[i]
        ax.plot(pos_var[:, 0], pos_var[:, 1], 'r-', linewidth=2, label=f'+{i+1}')
        ax.plot(neg_var[:, 0], neg_var[:, 1], 'b-', linewidth=2, label=f'-{i+1}')
        
        ax.set_title(f'Mode {i+1}')
        ax.axis('equal')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_segmentation(image, title='Segmentation', save_path=None):
    """
    Plot a segmentation image.
    
    Args:
        image: (H, W) array of segmentation
        title: Plot title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(6, 6))
    
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_multiple_segmentations(images, titles, save_path=None):
    """
    Plot multiple segmentation images side by side.
    
    Args:
        images: List of (H, W) arrays of segmentations
        titles: List of titles
        save_path: Path to save the figure
    """
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(6 * n_images, 6))
    
    if n_images == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_comparison_pdf(original_images, processed_images, output_pdf):
    """
    Create a PDF with original and processed images side by side.
    
    Args:
        original_images: List of original images
        processed_images: List of processed images
        output_pdf: Path to save the PDF
    """
    with PdfPages(output_pdf) as pdf:
        for i, (orig, proc) in enumerate(zip(original_images, processed_images)):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(orig, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            ax2.imshow(proc, cmap='gray')
            ax2.set_title('Processed Image')
            ax2.axis('off')
            
            plt.suptitle(f'Image {i+1}')
            plt.tight_layout()
            
            pdf.savefig(fig)
            plt.close(fig)