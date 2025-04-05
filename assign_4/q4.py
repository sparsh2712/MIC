
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.kernel_pca import (
    compute_pca, project_pca, 
    compute_kernel_pca, project_kernel_pca, compute_preimage
)
from utils.visualization import (
    plot_segmentation, plot_multiple_segmentations, create_comparison_pdf
)
from utils.io_utils import (
    load_segmentation_images, create_directory, 
    ensure_data_extracted, save_results_to_txt
)

def q4():
    print("Processing Question 4: Kernel PCA for Segmentation Modeling")
    
    # Create results directory
    results_dir = 'q4_results'
    create_directory(results_dir)
    
    # Ensure data is extracted
    ensure_data_extracted('assign4_data')
    
    # Load segmentation images
    segmentation_dir = os.path.join('assign4_data', 'anatomicalSegmentations')
    segmentations = load_segmentation_images(segmentation_dir)
    
    # Resize to 64x64 to reduce computational complexity
    resized_segmentations = []
    for seg in segmentations:
        resized = np.array(plt.imshow(seg, cmap='gray').get_array())
        if resized.shape != (64, 64):
            from skimage.transform import resize
            resized = resize(seg, (64, 64), anti_aliasing=True, preserve_range=True)
        resized_segmentations.append(resized)
    
    # Vectorize the segmentations
    n_samples = len(resized_segmentations)
    n_pixels = 64 * 64
    X = np.zeros((n_samples, n_pixels))
    for i, seg in enumerate(resized_segmentations):
        X[i] = seg.flatten()
    
    # (a) Implement standard PCA
    print("(a) Implementing standard PCA...")
    
    # Compute PCA
    mean, components, explained_variance, explained_variance_ratio = compute_pca(X)
    
    # Plot eigen spectrum
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 21), explained_variance[:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('PCA Eigen Spectrum')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'pca_eigen_spectrum.png'))
    plt.close()
    
    # Plot mean image
    plt.figure(figsize=(6, 6))
    plt.imshow(mean.reshape(64, 64), cmap='gray')
    plt.title('Mean Segmentation')
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, 'pca_mean_image.png'))
    plt.close()
    
    # Plot first 2 modes of variation
    for i in range(2):
        plt.figure(figsize=(15, 5))
        
        # Mean - 2*std
        neg_var = mean - 2 * np.sqrt(explained_variance[i]) * components[:, i]
        neg_var = np.clip(neg_var, 0, 1).reshape(64, 64)
        
        # Mean
        mean_img = mean.reshape(64, 64)
        
        # Mean + 2*std
        pos_var = mean + 2 * np.sqrt(explained_variance[i]) * components[:, i]
        pos_var = np.clip(pos_var, 0, 1).reshape(64, 64)
        
        plt.subplot(1, 3, 1)
        plt.imshow(neg_var, cmap='gray')
        plt.title(f'PC{i+1} - 2σ')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mean_img, cmap='gray')
        plt.title('Mean')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pos_var, cmap='gray')
        plt.title(f'PC{i+1} + 2σ')
        plt.axis('off')
        
        plt.savefig(os.path.join(results_dir, f'pca_mode{i+1}_variation.png'))
        plt.close()
    
    # (b) Implement kernel PCA
    print("(b) Implementing kernel PCA...")
    
    # Parameters for the Gaussian kernel
    gamma = 1.0 / (n_pixels * np.var(X))
    
    # Compute kernel PCA
    K, alphas, lambdas = compute_kernel_pca(X, gamma=gamma)
    
    # Plot eigen spectrum
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 21), lambdas[:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Kernel PCA Eigen Spectrum')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'kpca_eigen_spectrum.png'))
    plt.close()
    
    # Compute pre-image of the mean in RKHS
    # Since the mean in RKHS is at the origin, we need to project and then compute pre-image
    preimage_mean = compute_preimage(np.zeros(alphas.shape[0]), X, gamma=gamma)
    preimage_mean = preimage_mean.reshape(64, 64)
    
    # Plot pre-image of the mean
    plt.figure(figsize=(6, 6))
    plt.imshow(preimage_mean, cmap='gray')
    plt.title('Pre-image of RKHS Mean')
    plt.axis('off')
    plt.savefig(os.path.join(results_dir, 'kpca_preimage_mean.png'))
    plt.close()
    
    # (c) Implement projection onto PCA and kernel PCA modes for distorted segmentations
    print("(c) Implementing projection onto PCA and kernel PCA modes...")
    
    # Load distorted segmentations
    distorted_dir = os.path.join('assign4_data', 'anatomicalSegmentationsDistorted')
    distorted_segmentations = load_segmentation_images(distorted_dir)
    
    # Resize and vectorize distorted segmentations
    distorted_resized = []
    X_distorted = np.zeros((len(distorted_segmentations), n_pixels))
    for i, seg in enumerate(distorted_segmentations):
        if seg.shape != (64, 64):
            from skimage.transform import resize
            resized = resize(seg, (64, 64), anti_aliasing=True, preserve_range=True)
        else:
            resized = seg
        distorted_resized.append(resized)
        X_distorted[i] = resized.flatten()
    
    # Project distorted segmentations onto PCA components
    X_pca_proj = project_pca(X_distorted, mean, components, n_components=3)
    
    # Project onto kernel PCA and compute pre-images
    kpca_preimages = []
    for i in range(X_distorted.shape[0]):
        # Project onto kernel PCA components
        y = project_kernel_pca(X_distorted[i:i+1], X, alphas, gamma=gamma, n_components=3)
        
        # Compute pre-image
        preimage = compute_preimage(y[0], X, gamma=gamma)
        kpca_preimages.append(preimage.reshape(64, 64))
    
    # Reshape PCA projections
    pca_projected_images = [X_pca_proj[i].reshape(64, 64) for i in range(X_distorted.shape[0])]
    
    # Create PDF comparing original and corrected images
    original_images = distorted_resized
    
    # Create PDF for PCA projections
    create_comparison_pdf(
        original_images, 
        pca_projected_images, 
        os.path.join(results_dir, 'pca_projections.pdf')
    )
    
    # Create PDF for kernel PCA projections
    create_comparison_pdf(
        original_images, 
        kpca_preimages, 
        os.path.join(results_dir, 'kpca_projections.pdf')
    )
    
    # Save some example images for visual comparison
    for i in range(min(5, len(original_images))):
        plot_multiple_segmentations(
            [original_images[i], pca_projected_images[i], kpca_preimages[i]],
            ['Original', 'PCA Projection', 'Kernel PCA Projection'],
            save_path=os.path.join(results_dir, f'comparison_sample_{i+1}.png')
        )
    
    # Save eigenvalues to text file
    eigenvalue_results = {
        'pca_explained_variance': explained_variance,
        'pca_explained_variance_ratio': explained_variance_ratio,
        'kpca_eigenvalues': lambdas
    }
    save_results_to_txt(eigenvalue_results, os.path.join(results_dir, 'eigenvalues.txt'))
    
    print("Q4 completed. Results saved to 'q4_results' directory.")

if __name__ == "__main__":
    q4()