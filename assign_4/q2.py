"""
Question 2: Shape Analysis on Human Cardiac Shapes
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.shape_utils import (
    compute_mean_shape_preshape, compute_mean_shape_full,
    compute_shape_modes, generate_shape_variations
)
from utils.visualization import (
    plot_pointsets, plot_mean_and_aligned,
    plot_eigenvalues, plot_mean_and_variations
)
from utils.io_utils import (
    load_segmentation_images, extract_boundary_points,
    create_directory, save_results_to_txt, ensure_data_extracted
)

def modified_plot_pointsets(pointsets, title='Pointsets', save_path=None, colors=None):
    """
    Plot multiple pointsets with random colors, treating each pointset as having two separate rings.
    
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
        half_points = len(points) // 2
        
        # Plot outer ring
        outer_points = points[:half_points]
        # Close the loop by adding the first point at the end
        outer_points_closed = np.vstack([outer_points, outer_points[0]])
        plt.plot(outer_points_closed[:, 0], outer_points_closed[:, 1], '-', color=colors[i], markersize=4, linewidth=1)
        
        # Plot inner ring
        inner_points = points[half_points:]
        # Close the loop by adding the first point at the end
        inner_points_closed = np.vstack([inner_points, inner_points[0]])
        plt.plot(inner_points_closed[:, 0], inner_points_closed[:, 1], '-', color=colors[i], markersize=4, linewidth=1)
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def modified_plot_mean_and_aligned(mean_shape, aligned_shapes, title='Mean and Aligned Shapes', save_path=None):
    """
    Plot mean shape and aligned shapes, treating each shape as having two separate rings.
    
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
        half_points = len(shape) // 2
        
        # Outer ring
        outer_points = shape[:half_points]
        outer_points_closed = np.vstack([outer_points, outer_points[0]])
        plt.plot(outer_points_closed[:, 0], outer_points_closed[:, 1], '-', color=colors[i], alpha=0.3, linewidth=1)
        
        # Inner ring
        inner_points = shape[half_points:]
        inner_points_closed = np.vstack([inner_points, inner_points[0]])
        plt.plot(inner_points_closed[:, 0], inner_points_closed[:, 1], '-', color=colors[i], alpha=0.3, linewidth=1)
    
    # Plot mean shape
    half_points = len(mean_shape) // 2
    
    # Outer ring of mean shape
    outer_mean = mean_shape[:half_points]
    outer_mean_closed = np.vstack([outer_mean, outer_mean[0]])
    plt.plot(outer_mean_closed[:, 0], outer_mean_closed[:, 1], 'ro-', linewidth=2, markersize=4, label='Mean Outer Boundary')
    
    # Inner ring of mean shape
    inner_mean = mean_shape[half_points:]
    inner_mean_closed = np.vstack([inner_mean, inner_mean[0]])
    plt.plot(inner_mean_closed[:, 0], inner_mean_closed[:, 1], 'bo-', linewidth=2, markersize=4, label='Mean Inner Boundary')
    
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def modified_plot_mean_and_variations(mean_shape, aligned_shapes, variations, title='Mean and Variations', save_path=None):
    """
    Plot mean shape, aligned shapes, and shape variations, treating each shape as having two separate rings.
    
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
    
    half_points = len(mean_shape) // 2
    
    for i, ax in enumerate(axes):
        # Plot aligned shapes (faded in background)
        for shape in aligned_shapes:
            # Outer ring
            outer_points = shape[:half_points]
            outer_points_closed = np.vstack([outer_points, outer_points[0]])
            ax.plot(outer_points_closed[:, 0], outer_points_closed[:, 1], '-', color='lightgray', alpha=0.2, linewidth=1)
            
            # Inner ring
            inner_points = shape[half_points:]
            inner_points_closed = np.vstack([inner_points, inner_points[0]])
            ax.plot(inner_points_closed[:, 0], inner_points_closed[:, 1], '-', color='lightgray', alpha=0.2, linewidth=1)
        
        # Plot mean shape
        # Outer ring of mean
        outer_mean = mean_shape[:half_points]
        outer_mean_closed = np.vstack([outer_mean, outer_mean[0]])
        ax.plot(outer_mean_closed[:, 0], outer_mean_closed[:, 1], 'k-', linewidth=2, label='Mean')
        
        # Inner ring of mean
        inner_mean = mean_shape[half_points:]
        inner_mean_closed = np.vstack([inner_mean, inner_mean[0]])
        ax.plot(inner_mean_closed[:, 0], inner_mean_closed[:, 1], 'k-', linewidth=2)
        
        # Plot variations
        pos_var, neg_var = variations[i]
        
        # Outer ring of positive variation
        outer_pos = pos_var[:half_points]
        outer_pos_closed = np.vstack([outer_pos, outer_pos[0]])
        ax.plot(outer_pos_closed[:, 0], outer_pos_closed[:, 1], 'r-', linewidth=2, label=f'+{i+1}')
        
        # Inner ring of positive variation
        inner_pos = pos_var[half_points:]
        inner_pos_closed = np.vstack([inner_pos, inner_pos[0]])
        ax.plot(inner_pos_closed[:, 0], inner_pos_closed[:, 1], 'r-', linewidth=2)
        
        # Outer ring of negative variation
        outer_neg = neg_var[:half_points]
        outer_neg_closed = np.vstack([outer_neg, outer_neg[0]])
        ax.plot(outer_neg_closed[:, 0], outer_neg_closed[:, 1], 'b-', linewidth=2, label=f'-{i+1}')
        
        # Inner ring of negative variation
        inner_neg = neg_var[half_points:]
        inner_neg_closed = np.vstack([inner_neg, inner_neg[0]])
        ax.plot(inner_neg_closed[:, 0], inner_neg_closed[:, 1], 'b-', linewidth=2)
        
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

def extract_cardiac_boundary_points(segmentation, n_points=40):
    """
    Extract boundary points from a cardiac segmentation image, 
    ensuring both inner and outer boundaries are identified.
    
    Args:
        segmentation: (H, W) array of segmentation
        n_points: Total number of points to extract (will be split between boundaries)
        
    Returns:
        points: (N, 2) array of boundary points
    """
    # Threshold the image
    binary = (segmentation > 0.5).astype(np.uint8)
    
    # Find contours
    import cv2
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if len(contours) == 0:
        print("Warning: No contours found in segmentation")
        return np.zeros((0, 2))
    
    # Points to return
    points = []
    
    # For cardiac shapes, we need to handle inner and outer boundaries
    if len(contours) >= 2:
        # Outer boundary
        outer_contour = contours[0]
        outer_contour = outer_contour.squeeze()
        
        # Ensure we have the right number of points for the outer contour
        n_outer = n_points // 2
        if len(outer_contour) < n_outer:
            # Not enough points, duplicate some
            indices = np.mod(np.arange(n_outer), len(outer_contour))
            outer_points = outer_contour[indices]
        else:
            # Sample evenly
            indices = np.linspace(0, len(outer_contour) - 1, n_outer, dtype=int)
            outer_points = outer_contour[indices]
        
        points.extend(outer_points)
        
        # Inner boundary
        inner_contour = contours[1]
        inner_contour = inner_contour.squeeze()
        
        # Ensure we have the right number of points for the inner contour
        n_inner = n_points - n_outer
        if len(inner_contour) < n_inner:
            # Not enough points, duplicate some
            indices = np.mod(np.arange(n_inner), len(inner_contour))
            inner_points = inner_contour[indices]
        else:
            # Sample evenly
            indices = np.linspace(0, len(inner_contour) - 1, n_inner, dtype=int)
            inner_points = inner_contour[indices]
        
        points.extend(inner_points)
    else:
        # Just one boundary - use all points for it
        outer_contour = contours[0].squeeze()
        
        if len(outer_contour) < n_points:
            # Not enough points, duplicate some
            indices = np.mod(np.arange(n_points), len(outer_contour))
            points = outer_contour[indices]
        else:
            # Sample evenly
            indices = np.linspace(0, len(outer_contour) - 1, n_points, dtype=int)
            points = outer_contour[indices]
    
    # Convert to numpy array
    points = np.array(points)
    
    return points

def q2():
    # Hardcode the number of points to extract per segmentation
    n_points = 40
    
    print(f"Processing Question 2: Shape Analysis on Human Cardiac Shapes (using {n_points} points)")
    
    # Create results directory
    results_dir = 'q2_results'
    create_directory(results_dir)
    
    # Create setpoint_extracted_imgs directory
    extracted_imgs_dir = os.path.join(results_dir, 'setpoint_extracted_imgs')
    create_directory(extracted_imgs_dir)
    
    # Ensure data is extracted
    ensure_data_extracted('assign4_data')
    
    # Load cardiac segmentation images
    segmentation_dir = os.path.join('assign4_data', 'anatomicalSegmentations')
    segmentations = load_segmentation_images(segmentation_dir)
    
    print(f"Loaded {len(segmentations)} segmentation images")
    
    # (a) Extract boundary points from segmentations
    print(f"(a) Extracting boundary points from segmentations...")
    shapes = []
    for i, seg in enumerate(segmentations):
        # Extract points with custom function for cardiac shapes
        points = extract_cardiac_boundary_points(seg, n_points=n_points)
        
        # Skip if no points were extracted
        if points.shape[0] == 0:
            print(f"Warning: No points extracted from segmentation {i+1}")
            continue
        
        shapes.append(points)
        
        # Save the point extraction for visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(seg, cmap='gray')
        
        # Color-code the two boundaries
        half_points = len(points) // 2
        
        # Plot outer boundary (first half of points) in red
        outer_points = points[:half_points]
        # Add the first point at the end to close the loop
        outer_points_closed = np.vstack([outer_points, outer_points[0]])
        plt.plot(outer_points_closed[:, 0], outer_points_closed[:, 1], 'r-', linewidth=2, label='Outer Boundary')
        plt.scatter(outer_points[:, 0], outer_points[:, 1], c='red', s=15)
        
        # Plot inner boundary (second half of points) in blue
        inner_points = points[half_points:]
        # Add the first point at the end to close the loop
        inner_points_closed = np.vstack([inner_points, inner_points[0]])
        plt.plot(inner_points_closed[:, 0], inner_points_closed[:, 1], 'b-', linewidth=2, label='Inner Boundary')
        plt.scatter(inner_points[:, 0], inner_points[:, 1], c='blue', s=15)
        
        plt.title(f'Segmentation {i+1} with Extracted Points')
        plt.axis('off')
        plt.legend()
        
        # Save only to the setpoint_extracted_imgs directory
        plt.savefig(os.path.join(extracted_imgs_dir, f'points_extraction_{i+1}.png'))
        plt.close()
    
    print(f"Processed {len(shapes)} valid shapes, each with {shapes[0].shape[0]} points")
    
    # (b) Plot initial pointsets
    print("(b) Plotting initial pointsets...")
    modified_plot_pointsets(shapes, title='Initial Cardiac Shapes', 
                  save_path=os.path.join(results_dir, 'initial_shapes.png'))
    
    # (c) Compute and plot mean shapes using both methods
    print("(c) Computing mean shapes using both alignment methods...")
    
    # Method 1: Pre-shape alignment (Code11)
    mean_shape_1, aligned_shapes_1 = compute_mean_shape_preshape(shapes)
    modified_plot_mean_and_aligned(mean_shape_1, aligned_shapes_1, 
                         title='Mean Shape (Pre-shape Alignment)',
                         save_path=os.path.join(results_dir, 'mean_shape_preshape.png'))
    
    # Method 2: Full alignment (Code22)
    mean_shape_2, aligned_shapes_2 = compute_mean_shape_full(shapes)
    modified_plot_mean_and_aligned(mean_shape_2, aligned_shapes_2, 
                         title='Mean Shape (Full Alignment)',
                         save_path=os.path.join(results_dir, 'mean_shape_full.png'))
    
    # (d) Compute and plot eigenvalues for top 3 modes
    print("(d) Computing principal modes of shape variation...")
    
    # Method 1: Pre-shape alignment
    eigenvalues_1, eigenvectors_1 = compute_shape_modes(aligned_shapes_1, mean_shape_1, n_modes=3)
    plot_eigenvalues(eigenvalues_1, title='Eigenvalues (Pre-shape Alignment)',
                    save_path=os.path.join(results_dir, 'eigenvalues_preshape.png'))
    
    # Method 2: Full alignment
    eigenvalues_2, eigenvectors_2 = compute_shape_modes(aligned_shapes_2, mean_shape_2, n_modes=3)
    plot_eigenvalues(eigenvalues_2, title='Eigenvalues (Full Alignment)',
                    save_path=os.path.join(results_dir, 'eigenvalues_full.png'))
    
    # Save eigenvalues to text file
    eigenvalue_results = {
        'eigenvalues_preshape': eigenvalues_1,
        'eigenvalues_full': eigenvalues_2
    }
    save_results_to_txt(eigenvalue_results, os.path.join(results_dir, 'eigenvalues.txt'))
    
    # (e) Generate and plot shape variations
    print("(e) Generating shape variations along principal modes...")
    
    # Method 1: Pre-shape alignment
    variations_1 = generate_shape_variations(mean_shape_1, eigenvectors_1, eigenvalues_1, n_std=3)
    modified_plot_mean_and_variations(mean_shape_1, aligned_shapes_1, variations_1,
                            title='Shape Variations (Pre-shape Alignment)',
                            save_path=os.path.join(results_dir, 'variations_preshape.png'))
    
    # Method 2: Full alignment
    variations_2 = generate_shape_variations(mean_shape_2, eigenvectors_2, eigenvalues_2, n_std=3)
    modified_plot_mean_and_variations(mean_shape_2, aligned_shapes_2, variations_2,
                            title='Shape Variations (Full Alignment)',
                            save_path=os.path.join(results_dir, 'variations_full.png'))
    
    print("Q2 completed. Results saved to 'q2_results' directory.")

if __name__ == "__main__":
    q2()