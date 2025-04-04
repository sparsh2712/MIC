import numpy as np
from scipy import linalg

def center_pointset(points):
    """
    Center a pointset to have zero mean.
    
    Args:
        points: (N, D) array of N D-dimensional points
        
    Returns:
        centered_points: (N, D) array of centered points
        translation: (D,) translation vector
    """
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    return centered_points, centroid

def scale_pointset(points):
    """
    Scale a pointset to have unit Frobenius norm.
    
    Args:
        points: (N, D) array of N D-dimensional points
        
    Returns:
        scaled_points: (N, D) array of scaled points
        scale: scaling factor
    """
    scale = np.sqrt(np.sum(points**2))
    if scale > 0:
        scaled_points = points / scale
    else:
        scaled_points = points
        scale = 1.0
    return scaled_points, scale

def to_preshape(points):
    """
    Transform pointset to pre-shape space (centered, unit size).
    
    Args:
        points: (N, D) array of N D-dimensional points
        
    Returns:
        preshape: (N, D) array in pre-shape space
        translation: (D,) translation vector
        scale: scaling factor
    """
    centered_points, translation = center_pointset(points)
    preshape, scale = scale_pointset(centered_points)
    return preshape, translation, scale

def align_preshapes(X, Y):
    """
    Align pre-shape Y to pre-shape X by finding optimal rotation (Code1).
    
    Args:
        X: (N, D) array of N D-dimensional points in pre-shape space
        Y: (N, D) array of N D-dimensional points in pre-shape space
        
    Returns:
        Y_aligned: (N, D) array of aligned points
        R: (D, D) rotation matrix
    """
    # Compute SVD
    XY_T = X.T @ Y
    U, _, Vt = linalg.svd(XY_T)
    
    # Ensure proper rotation (det=1)
    det = np.linalg.det(U @ Vt)
    if det < 0:
        Vt[-1, :] = -Vt[-1, :]
    
    # Compute rotation matrix
    R = U @ Vt
    
    # Apply rotation
    Y_aligned = Y @ R.T
    
    return Y_aligned, R

def align_pointsets(X, Y):
    """
    Align pointset Y to pointset X by solving for scale, translation, and rotation (Code2).
    
    Args:
        X: (N, D) array of N D-dimensional points
        Y: (N, D) array of N D-dimensional points
        
    Returns:
        Y_aligned: (N, D) array of aligned points
        params: dict with transformation parameters
    """
    # Center both pointsets
    X_centered, X_trans = center_pointset(X)
    Y_centered, Y_trans = center_pointset(Y)
    
    # Compute optimal scale
    X_norm = np.sqrt(np.sum(X_centered**2))
    Y_norm = np.sqrt(np.sum(Y_centered**2))
    scale = X_norm / Y_norm if Y_norm > 0 else 1.0
    
    # Scale Y
    Y_scaled = Y_centered * scale
    
    # Find optimal rotation
    Y_rotated, R = align_preshapes(X_centered, Y_scaled)
    
    # Compute full transformation
    Y_aligned = Y_rotated + X_trans
    
    params = {
        'rotation': R,
        'scale': scale,
        'translation': X_trans - Y_trans * scale
    }
    
    return Y_aligned, params

def procrustes_distance(X, Y):
    """
    Compute Procrustes distance between two shapes.
    
    Args:
        X: (N, D) array of N D-dimensional points
        Y: (N, D) array of N D-dimensional points
        
    Returns:
        distance: Procrustes distance
    """
    # Convert to pre-shapes
    X_preshape, _, _ = to_preshape(X)
    Y_preshape, _, _ = to_preshape(Y)
    
    # Align Y to X
    Y_aligned, _ = align_preshapes(X_preshape, Y_preshape)
    
    # Compute distance
    distance = np.sqrt(np.sum((X_preshape - Y_aligned)**2))
    
    return distance

def compute_mean_shape_preshape(shapes, max_iter=100, tol=1e-6):
    """
    Compute mean shape using Code11 (pre-shape alignment).
    
    Args:
        shapes: List of (N, D) arrays of N D-dimensional points
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        mean_shape: (N, D) array representing the mean shape
        aligned_shapes: List of aligned shapes
    """
    # Initialize with first shape
    mean_shape = shapes[0].copy()
    mean_shape, _, _ = to_preshape(mean_shape)
    
    prev_mean = None
    iter_count = 0
    
    while iter_count < max_iter:
        # Transform mean to pre-shape
        mean_preshape, mean_trans, mean_scale = to_preshape(mean_shape)
        
        # Align all shapes to the mean
        aligned_shapes = []
        for shape in shapes:
            # Transform to pre-shape
            shape_preshape, _, _ = to_preshape(shape)
            
            # Align to mean
            aligned_shape, _ = align_preshapes(mean_preshape, shape_preshape)
            aligned_shapes.append(aligned_shape)
        
        # Compute new mean
        new_mean = np.mean(aligned_shapes, axis=0)
        new_mean, _, _ = to_preshape(new_mean)
        
        # Check convergence
        if prev_mean is not None:
            change = np.sqrt(np.sum((new_mean - prev_mean)**2))
            if change < tol:
                break
        
        prev_mean = new_mean
        mean_shape = new_mean
        iter_count += 1
    
    return mean_shape, aligned_shapes

def compute_mean_shape_full(shapes, max_iter=100, tol=1e-6):
    """
    Compute mean shape using Code22 (full alignment).
    
    Args:
        shapes: List of (N, D) arrays of N D-dimensional points
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        mean_shape: (N, D) array representing the mean shape
        aligned_shapes: List of aligned shapes
    """
    # Initialize with first shape
    mean_shape = shapes[0].copy()
    
    prev_mean = None
    iter_count = 0
    
    while iter_count < max_iter:
        # Align all shapes to the mean
        aligned_shapes = []
        for shape in shapes:
            # Align to mean
            aligned_shape, _ = align_pointsets(mean_shape, shape)
            aligned_shapes.append(aligned_shape)
        
        # Compute new mean
        new_mean = np.mean(aligned_shapes, axis=0)
        
        # Check convergence
        if prev_mean is not None:
            change = np.sqrt(np.sum((new_mean - prev_mean)**2))
            if change < tol:
                break
        
        prev_mean = new_mean
        mean_shape = new_mean
        iter_count += 1
    
    return mean_shape, aligned_shapes

def compute_shape_modes(aligned_shapes, mean_shape, n_modes=3):
    """
    Compute principal modes of shape variation.
    
    Args:
        aligned_shapes: List of aligned (N, D) arrays
        mean_shape: (N, D) array of mean shape
        n_modes: Number of principal modes to compute
        
    Returns:
        eigenvalues: Array of eigenvalues
        eigenvectors: Array of eigenvectors
    """
    # Stack shapes into a matrix
    n_shapes = len(aligned_shapes)
    n_points, n_dims = aligned_shapes[0].shape
    shape_matrix = np.zeros((n_shapes, n_points * n_dims))
    
    for i, shape in enumerate(aligned_shapes):
        shape_matrix[i] = shape.flatten()
    
    # Center the data
    mean_vector = mean_shape.flatten()
    centered_data = shape_matrix - mean_vector
    
    # Compute covariance matrix
    cov_matrix = (centered_data.T @ centered_data) / (n_shapes - 1)
    
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return top n_modes
    return eigenvalues[:n_modes], eigenvectors[:, :n_modes]

def generate_shape_variations(mean_shape, eigenvectors, eigenvalues, n_std=2):
    """
    Generate shape variations along principal modes.
    
    Args:
        mean_shape: (N, D) array of mean shape
        eigenvectors: Array of eigenvectors
        eigenvalues: Array of eigenvalues
        n_std: Number of standard deviations
        
    Returns:
        variations: List of shape variations (positive and negative)
    """
    variations = []
    mean_vector = mean_shape.flatten()
    n_points, n_dims = mean_shape.shape
    
    for i in range(len(eigenvalues)):
        std_dev = np.sqrt(eigenvalues[i])
        
        # Positive variation
        pos_var = mean_vector + n_std * std_dev * eigenvectors[:, i]
        pos_shape = pos_var.reshape(n_points, n_dims)
        
        # Negative variation
        neg_var = mean_vector - n_std * std_dev * eigenvectors[:, i]
        neg_shape = neg_var.reshape(n_points, n_dims)
        
        variations.append((pos_shape, neg_shape))
    
    return variations