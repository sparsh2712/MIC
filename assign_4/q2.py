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

def q2():
    # Hardcode the number of points to extract per segmentation
    n_points = 40
    
    print(f"Processing Question 2: Shape Analysis on Human Cardiac Shapes (using {n_points} points)")
    
    # Create results directory
    results_dir = 'q2_results'
    create_directory(results_dir)
    
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
        # Extract points with hardcoded number
        points = extract_boundary_points(seg, n_points=n_points)
        
        # Skip if no points were extracted
        if points.shape[0] == 0:
            print(f"Warning: No points extracted from segmentation {i+1}")
            continue
            
        # If not exactly n_points points, resample
        if points.shape[0] != n_points:
            print(f"Resampling segmentation {i+1} from {points.shape[0]} to {n_points} points")
            # Simple linear resampling
            indices = np.linspace(0, points.shape[0] - 1, n_points, dtype=int)
            points = points[indices]
            
        shapes.append(points)
        
        # Save the point extraction for visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(seg, cmap='gray')
        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=10)
        plt.title(f'Segmentation {i+1} with Extracted Points')
        plt.axis('off')
        plt.savefig(os.path.join(results_dir, f'points_extraction_{i+1}.png'))
        plt.close()
    
    print(f"Processed {len(shapes)} valid shapes, each with {shapes[0].shape[0]} points")
    
    # (b) Plot initial pointsets
    print("(b) Plotting initial pointsets...")
    plot_pointsets(shapes, title='Initial Cardiac Shapes', 
                  save_path=os.path.join(results_dir, 'initial_shapes.png'))
    
    # (c) Compute and plot mean shapes using both methods
    print("(c) Computing mean shapes using both alignment methods...")
    
    # Method 1: Pre-shape alignment (Code11)
    mean_shape_1, aligned_shapes_1 = compute_mean_shape_preshape(shapes)
    plot_mean_and_aligned(mean_shape_1, aligned_shapes_1, 
                         title='Mean Shape (Pre-shape Alignment)',
                         save_path=os.path.join(results_dir, 'mean_shape_preshape.png'))
    
    # Method 2: Full alignment (Code22)
    mean_shape_2, aligned_shapes_2 = compute_mean_shape_full(shapes)
    plot_mean_and_aligned(mean_shape_2, aligned_shapes_2, 
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
    plot_mean_and_variations(mean_shape_1, aligned_shapes_1, variations_1,
                            title='Shape Variations (Pre-shape Alignment)',
                            save_path=os.path.join(results_dir, 'variations_preshape.png'))
    
    # Method 2: Full alignment
    variations_2 = generate_shape_variations(mean_shape_2, eigenvectors_2, eigenvalues_2, n_std=3)
    plot_mean_and_variations(mean_shape_2, aligned_shapes_2, variations_2,
                            title='Shape Variations (Full Alignment)',
                            save_path=os.path.join(results_dir, 'variations_full.png'))
    
    print("Q2 completed. Results saved to 'q2_results' directory.")

if __name__ == "__main__":
    q2()