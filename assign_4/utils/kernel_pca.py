import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import rbf_kernel

def compute_pca(X, n_components=None):
    """
    Compute PCA on vectorized images.
    
    Args:
        X: (n_samples, n_features) array of vectorized images
        n_components: Number of components to keep
        
    Returns:
        mean: Mean image
        components: Principal components
        explained_variance: Explained variance
        explained_variance_ratio: Explained variance ratio
    """
    # Compute mean
    mean = np.mean(X, axis=0)
    
    # Center the data
    X_centered = X - mean
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Compute explained variance ratio
    explained_variance = eigenvalues
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    
    # Keep only n_components
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        explained_variance = explained_variance[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]
    
    return mean, eigenvectors, explained_variance, explained_variance_ratio

def project_pca(X, mean, components, n_components=None):
    """
    Project data onto PCA components.
    
    Args:
        X: (n_samples, n_features) array of vectorized images
        mean: Mean image
        components: Principal components
        n_components: Number of components to use
        
    Returns:
        X_proj: (n_samples, n_features) array of projected images
    """
    if n_components is not None:
        components = components[:, :n_components]
    
    # Center the data
    X_centered = X - mean
    
    # Project onto components
    X_transformed = X_centered @ components
    
    # Transform back to image space
    X_proj = X_transformed @ components.T + mean
    
    return X_proj

def compute_kernel_pca(X, n_components=None, gamma=None):
    """
    Compute kernel PCA with Gaussian kernel.
    
    Args:
        X: (n_samples, n_features) array of vectorized images
        n_components: Number of components to keep
        gamma: Parameter of the RBF kernel
        
    Returns:
        kernel_matrix: Kernel matrix
        alphas: Eigenvectors in feature space
        lambdas: Eigenvalues
    """
    # Compute kernel matrix
    K = rbf_kernel(X, gamma=gamma)
    
    # Center the kernel matrix
    n_samples = K.shape[0]
    one_n = np.ones((n_samples, n_samples)) / n_samples
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Compute eigenvectors and eigenvalues
    lambdas, alphas = linalg.eigh(K_centered)
    
    # Sort in descending order
    idx = np.argsort(lambdas)[::-1]
    lambdas = lambdas[idx]
    alphas = alphas[:, idx]
    
    # Normalize eigenvectors
    alphas = alphas / np.sqrt(lambdas)
    
    # Keep only n_components
    if n_components is not None:
        alphas = alphas[:, :n_components]
        lambdas = lambdas[:n_components]
    
    return K, alphas, lambdas

def project_kernel_pca(X, X_train, alphas, gamma=None, n_components=None):
    """
    Project data onto kernel PCA components.
    
    Args:
        X: (n_samples, n_features) array of vectorized images to project
        X_train: (n_train_samples, n_features) array of training images
        alphas: Eigenvectors in feature space
        gamma: Parameter of the RBF kernel
        n_components: Number of components to use
        
    Returns:
        Y: (n_samples, n_components) array of projections
    """
    if n_components is not None:
        alphas = alphas[:, :n_components]
    
    # Compute kernel matrix between X and X_train
    K = rbf_kernel(X, X_train, gamma=gamma)
    
    # Center the kernel matrix (approximate)
    n_train = X_train.shape[0]
    one_n = np.ones((X.shape[0], n_train)) / n_train
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    K_train_mean = np.mean(K_train, axis=0)
    K_centered = K - np.mean(K, axis=1, keepdims=True) - np.mean(K, axis=0, keepdims=True) + np.mean(K_train)
    
    # Project onto components
    Y = K_centered @ alphas
    
    return Y

def compute_preimage(Y, X_train, gamma=None, n_neighbors=5, n_iter=100, learning_rate=0.1):
    """
    Compute pre-image of kernel PCA projection.
    
    Args:
        Y: (n_components,) array of projection in feature space
        X_train: (n_train_samples, n_features) array of training images
        gamma: Parameter of the RBF kernel
        n_neighbors: Number of neighbors to use for initialization
        n_iter: Number of optimization iterations
        learning_rate: Learning rate for optimization
        
    Returns:
        preimage: (n_features,) array of pre-image
    """
    n_train, n_features = X_train.shape
    
    # Compute kernel matrix of training data
    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    
    # Compute weights for each training example
    weights = np.zeros(n_train)
    for i in range(n_train):
        kernel_diff = 0
        for j in range(Y.shape[0]):
            kernel_diff += (K_train[i, j] - Y[j])**2
        weights[i] = -kernel_diff
    
    # Normalize weights using softmax
    weights = np.exp(weights - np.max(weights))
    weights = weights / np.sum(weights)
    
    # Initialize pre-image as weighted average of training examples
    preimage = np.zeros(n_features)
    for i in range(n_train):
        preimage += weights[i] * X_train[i]
    
    # Refine pre-image using gradient descent
    for _ in range(n_iter):
        # Compute kernel values
        k_values = np.zeros(n_train)
        for i in range(n_train):
            diff = preimage - X_train[i]
            k_values[i] = np.exp(-gamma * np.sum(diff**2))
        
        # Compute gradient
        gradient = np.zeros(n_features)
        for i in range(n_train):
            diff = preimage - X_train[i]
            gradient += 2 * gamma * k_values[i] * (preimage - X_train[i])
        
        # Update pre-image
        preimage -= learning_rate * gradient
    
    # Ensure values are in [0, 1] for an image
    preimage = np.clip(preimage, 0, 1)
    
    return preimage