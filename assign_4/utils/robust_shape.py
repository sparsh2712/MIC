import numpy as np
from scipy import linalg, optimize
from utils.shape_utils import to_preshape, align_preshapes, procrustes_distance

def compute_squared_procrustes_mean(shapes, max_iter=100, tol=1e-6):
    """
    Compute the mean shape by minimizing the sum of squared Procrustes distances.
    
    Args:
        shapes: List of (N, D) arrays of N D-dimensional points
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        mean_shape: (N, D) array representing the robust mean shape
        aligned_shapes: List of aligned shapes
    """
    # Initialize with first shape
    mean_shape = shapes[0].copy()
    mean_shape, _, _ = to_preshape(mean_shape)
    
    prev_cost = float('inf')
    iter_count = 0
    
    while iter_count < max_iter:
        # Align all shapes to the mean
        aligned_shapes = []
        total_cost = 0
        
        for shape in shapes:
            # Transform to pre-shape
            shape_preshape, _, _ = to_preshape(shape)
            
            # Align to mean
            aligned_shape, _ = align_preshapes(mean_shape, shape_preshape)
            aligned_shapes.append(aligned_shape)
            
            # Compute squared Procrustes distance
            dist = np.sum((mean_shape - aligned_shape)**2)
            total_cost += dist
        
        # Compute new mean (equal weights for all shapes in squared case)
        new_mean = np.mean(aligned_shapes, axis=0)
        new_mean, _, _ = to_preshape(new_mean)
        
        # Check convergence
        if np.abs(total_cost - prev_cost) < tol:
            break
        
        prev_cost = total_cost
        mean_shape = new_mean
        iter_count += 1
    
    return mean_shape, aligned_shapes, iter_count, total_cost

def compute_l1_procrustes_mean(shapes, max_iter=100, tol=1e-6):
    """
    Compute the mean shape by minimizing the sum of L1 Procrustes distances.
    
    Args:
        shapes: List of (N, D) arrays of N D-dimensional points
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        mean_shape: (N, D) array representing the robust mean shape
        aligned_shapes: List of aligned shapes
    """
    # Initialize with first shape
    mean_shape = shapes[0].copy()
    mean_shape, _, _ = to_preshape(mean_shape)
    
    prev_cost = float('inf')
    iter_count = 0
    
    while iter_count < max_iter:
        # Align all shapes to the mean
        aligned_shapes = []
        distances = []
        total_cost = 0
        
        for shape in shapes:
            # Transform to pre-shape
            shape_preshape, _, _ = to_preshape(shape)
            
            # Align to mean
            aligned_shape, _ = align_preshapes(mean_shape, shape_preshape)
            aligned_shapes.append(aligned_shape)
            
            # Compute L1 Procrustes distance (absolute differences)
            dist = np.sqrt(np.sum((mean_shape - aligned_shape)**2))
            distances.append(dist)
            total_cost += dist
        
        # Convert to numpy arrays
        aligned_shapes_array = np.array(aligned_shapes)
        distances_array = np.array(distances)
        
        # Compute weights (inversely proportional to distances to be more robust)
        weights = 1.0 / (distances_array + 1e-10)  # Add small constant to avoid division by zero
        weights = weights / np.sum(weights)
        
        # Compute weighted median (geometric median approximation for L1)
        weighted_sum = np.zeros_like(mean_shape)
        for i, shape in enumerate(aligned_shapes):
            weighted_sum += weights[i] * shape
        
        new_mean = weighted_sum
        new_mean, _, _ = to_preshape(new_mean)
        
        # Check convergence
        if np.abs(total_cost - prev_cost) < tol:
            break
        
        prev_cost = total_cost
        mean_shape = new_mean
        iter_count += 1
    
    return mean_shape, aligned_shapes, iter_count, total_cost