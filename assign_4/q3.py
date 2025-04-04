"""
Question 3: Robust Shape Mean
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.shape_utils import to_preshape, align_preshapes
from utils.robust_shape import compute_squared_procrustes_mean, compute_l1_procrustes_mean
from utils.visualization import plot_pointsets, plot_mean_and_aligned
from utils.io_utils import load_mat_file, create_directory, save_results_to_txt

def q3():
    print("Processing Question 3: Robust Shape Mean")
    
    # Create results directory
    results_dir = 'q3_results'
    create_directory(results_dir)
    
    # Load elliptical shapes data
    data = load_mat_file(os.path.join('assign4_data', 'robustShapeMean2D.mat'))
    # Check the actual key in the mat file (assuming it's 'shapes')
    shape_data = data['pointsets']
    
    # Convert to list of point arrays
    shapes = []
    if len(shape_data.shape) == 3:
        # If data is stored as [2, n_points, n_shapes]
        for i in range(shape_data.shape[2]):
            shape = np.column_stack((shape_data[0, :, i], shape_data[1, :, i])).T
            shapes.append(shape)
    else:
        # Handle other possible formats
        print(f"Shape data has unusual format: {shape_data.shape}")
        # Try to infer the format and extract shapes accordingly
        if len(shape_data.shape) == 2:
            n_points = shape_data.shape[0] // 2
            for i in range(0, shape_data.shape[1]):
                shape = shape_data[:, i].reshape(2, n_points).T
                shapes.append(shape)
    
    # (a) Display original pointsets
    print("Displaying original pointsets...")
    plot_pointsets(shapes, title='Original Elliptical Shapes', 
                  save_path=os.path.join(results_dir, 'original_shapes.png'))
    
    # (a) Compute and display L2 (squared) procrustes mean
    print("Computing L2 (squared) Procrustes mean...")
    l2_mean, l2_aligned, l2_iters, l2_cost = compute_squared_procrustes_mean(shapes)
    
    # Save L2 results
    plot_mean_and_aligned(l2_mean, l2_aligned, 
                         title='L2 Procrustes Mean and Aligned Shapes',
                         save_path=os.path.join(results_dir, 'l2_mean_and_aligned.png'))
    
    # (b) Compute and display L1 procrustes mean
    print("Computing L1 Procrustes mean...")
    l1_mean, l1_aligned, l1_iters, l1_cost = compute_l1_procrustes_mean(shapes)
    
    # Save L1 results
    plot_mean_and_aligned(l1_mean, l1_aligned, 
                         title='L1 Procrustes Mean and Aligned Shapes',
                         save_path=os.path.join(results_dir, 'l1_mean_and_aligned.png'))
    
    # Save numerical results
    results = {
        'l2_iterations': l2_iters,
        'l2_final_cost': l2_cost,
        'l1_iterations': l1_iters,
        'l1_final_cost': l1_cost
    }
    save_results_to_txt(results, os.path.join(results_dir, 'robust_mean_results.txt'))
    
    # Compare the means directly
    plt.figure(figsize=(8, 8))
    plt.plot(l2_mean[:, 0], l2_mean[:, 1], 'r-', linewidth=2, label='L2 Mean')
    plt.plot(l1_mean[:, 0], l1_mean[:, 1], 'b-', linewidth=2, label='L1 Mean')
    plt.title('Comparison of L1 and L2 Procrustes Means')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'mean_comparison.png'))
    plt.close()
    
    print("Q3 completed. Results saved to 'q3_results' directory.")

if __name__ == "__main__":
    q3()