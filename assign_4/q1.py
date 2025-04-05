import os
import numpy as np
import matplotlib.pyplot as plt
from utils.shape_utils import (
    to_preshape, align_preshapes, align_pointsets,
    compute_mean_shape_preshape, compute_mean_shape_full,
    compute_shape_modes, generate_shape_variations
)
from utils.visualization import (
    plot_pointsets, plot_mean_and_aligned,
    plot_eigenvalues, plot_mean_and_variations
)
from utils.io_utils import load_mat_file, create_directory, save_results_to_txt

def q1():
    print("Processing Question 1: Shape Analysis on Human Hand Shapes")
    
    # Create results directory
    results_dir = 'q1_results'
    create_directory(results_dir)
    
    # Load hand data
    data = load_mat_file(os.path.join('assign4_data', 'hands2D.mat'))
    hands = data['shapes']
    
    # Print shape to verify
    print(f"Hand data shape: {hands.shape}")
    
    # Convert to list of point arrays
    shapes = []
    if hands.shape == (40, 56, 2):
        # Direct format (n_shapes, n_points, 2)
        print("Processing (n_shapes, n_points, 2) format")
        for i in range(hands.shape[0]):
            shapes.append(hands[i])
    elif len(hands.shape) == 3 and hands.shape[0] == 2:
        # Format (2, n_points, n_shapes)
        print("Processing (2, n_points, n_shapes) format")
        for i in range(hands.shape[2]):
            shape = np.column_stack((hands[0, :, i], hands[1, :, i])).T
            shapes.append(shape)
    else:
        print(f"Unexpected shape format: {hands.shape}")
        
    print(f"Processed {len(shapes)} shapes, each with {shapes[0].shape[0]} points")
    
    # (a) Plot initial pointsets
    print("(a) Plotting initial pointsets...")
    plot_pointsets(shapes, title='Initial Hand Shapes', 
                  save_path=os.path.join(results_dir, 'initial_shapes.png'))
    
    # (b) Compute and plot mean shapes using both methods
    print("(b) Computing mean shapes using both alignment methods...")
    
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
    
    # (c) Compute and plot eigenvalues for top 3 modes
    print("(c) Computing principal modes of shape variation...")
    
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
    
    # (d) Generate and plot shape variations
    print("(d) Generating shape variations along principal modes...")
    
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
    
    print("Q1 completed. Results saved to 'q1_results' directory.")

if __name__ == "__main__":
    q1()