import os
import cv2
import numpy as np
import pandas as pd
import glob
from utils.base_denoiser import BaseDenoiser
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.linear_model import orthogonal_mp_gram
import scipy.sparse as sparse
from scipy import linalg
from time import time
import matplotlib.pyplot as plt

class OptimizedKSVDDenoiser(BaseDenoiser):
    def __init__(self, patch_size=5, n_components=32, sparsity=3, n_iter=5,
                 use_global=True, sigma_est='adaptive'):
        super().__init__(f"OptimizedKSVD_p{patch_size}_n{n_components}_s{sparsity}")
        self.patch_size = patch_size
        self.n_components = n_components
        self.sparsity = sparsity
        self.n_iter = n_iter
        self.use_global = use_global
        self.sigma_est = sigma_est
        self.global_dictionary = None
        
    def initialize_dictionary(self, patches):
        """Initialize dictionary using DCT basis (faster than random patches)."""
        n_features = self.patch_size ** 2
        
        # Create overcomplete DCT dictionary
        from scipy.fftpack import dct
        dictionary = np.zeros((self.n_components, n_features))
        
        # Vectorized DCT basis creation
        for i in range(self.n_components):
            basis_vector = np.zeros(n_features)
            basis_vector[i % n_features] = 1
            basis_matrix = basis_vector.reshape(self.patch_size, self.patch_size)
            dictionary[i] = dct(dct(basis_matrix.T, norm='ortho').T, norm='ortho').flatten()
        
        # Normalize atoms
        norms = np.linalg.norm(dictionary, axis=1)
        dictionary /= norms[:, np.newaxis]
        
        return dictionary
    
    def estimate_noise_sigma(self, image):
        """Estimate noise standard deviation using robust median estimator."""
        # Extract high-frequency components
        high_freq = cv2.Laplacian(image, cv2.CV_64F)
        # Robust sigma estimation using median absolute deviation
        sigma = np.median(np.abs(high_freq)) / 0.6745
        return sigma
    
    def fast_omp_batch(self, signals, dictionary, sparsity, gram=None):
        """Highly optimized batch OMP using precomputed Gram matrix."""
        if gram is None:
            gram = dictionary @ dictionary.T
        
        # Precompute correlations
        correlations = dictionary @ signals.T
        
        # Use sklearn's optimized implementation
        coefficients = orthogonal_mp_gram(
            gram, correlations, n_nonzero_coefs=sparsity, copy_Gram=False, copy_Xy=False
        ).T
        
        return coefficients
    
    def ksvd_core(self, patches, dictionary, n_iter=None):
        """Optimized K-SVD core algorithm."""
        n_iter = n_iter or self.n_iter
        n_patches = patches.shape[0]
        n_features = patches.shape[1]
        
        # Ensure dictionary has correct shape
        if dictionary.shape[1] != n_features:
            dictionary = dictionary.T
        
        # Precompute Gram matrix once
        gram = dictionary @ dictionary.T
        
        for iteration in range(n_iter):
            # Sparse coding step
            coefficients = self.fast_omp_batch(patches, dictionary, self.sparsity, gram)
            
            # Dictionary update step - optimized version
            for j in range(self.n_components):
                # Find patches that use this atom
                indices = np.where(coefficients[:, j] != 0)[0]
                
                if len(indices) == 0:
                    continue
                
                # Compute error without atom j contribution
                error = patches[indices] - coefficients[indices] @ dictionary
                error += np.outer(coefficients[indices, j], dictionary[j])
                
                # Update atom using fast SVD
                try:
                    # Use randomized SVD for efficiency on the error matrix (patches x features)
                    u, s, vh = linalg.svd(error, full_matrices=False, lapack_driver='gesdd')
                    # Take the first right singular vector as the new atom
                    dictionary[j] = vh[0]
                    # Update coefficients as left singular vector times singular value
                    coefficients[indices, j] = u[:, 0] * s[0]
                except Exception as e:
                    # Fallback if SVD fails - take the mean of error patches
                    dictionary[j] = np.mean(error, axis=0)
                    norm = np.linalg.norm(dictionary[j])
                    if norm > 0:
                        dictionary[j] /= norm
                    # Update coefficients using projection
                    coefficients[indices, j] = error @ dictionary[j]
            
            # Update Gram matrix
            gram = dictionary @ dictionary.T
        
        return dictionary, coefficients
    
    def extract_patches_fast(self, image, stride=None):
        """Extract patches with configurable stride for speed/quality tradeoff."""
        stride = stride or self.patch_size // 2
        
        # Use sliding window approach for faster extraction
        h, w = image.shape
        patches = []
        indices = []
        
        for i in range(0, h - self.patch_size + 1, stride):
            for j in range(0, w - self.patch_size + 1, stride):
                patch = image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch.flatten())
                indices.append((i, j))
        
        return np.array(patches), indices
    
    def reconstruct_image_fast(self, denoised_patches, indices, image_shape):
        """Fast image reconstruction with optimized averaging."""
        reconstructed = np.zeros(image_shape)
        weight_map = np.zeros(image_shape)
        
        # Vectorized reconstruction
        for patch, (i, j) in zip(denoised_patches, indices):
            patch_2d = patch.reshape(self.patch_size, self.patch_size)
            reconstructed[i:i+self.patch_size, j:j+self.patch_size] += patch_2d
            weight_map[i:i+self.patch_size, j:j+self.patch_size] += 1
        
        # Avoid division by zero
        mask = weight_map > 0
        reconstructed[mask] /= weight_map[mask]
        
        return reconstructed
    
    def visualize_dictionary(self, dictionary, output_path):
        """Visualize dictionary atoms in a grid and save as PNG."""
        n_atoms = dictionary.shape[0]
        
        # Determine grid size based on number of atoms
        grid_size = int(np.ceil(np.sqrt(n_atoms)))
        
        # Create figure with appropriate size
        plt.figure(figsize=(12, 12))
        
        # Normalize dictionary globally for better visualization
        dict_min = dictionary.min()
        dict_max = dictionary.max()
        normalized_dict = (dictionary - dict_min) / (dict_max - dict_min + 1e-10)
        
        # Create a grid of dictionary atoms
        grid_image = np.ones((grid_size * self.patch_size + grid_size - 1,
                             grid_size * self.patch_size + grid_size - 1)) * 0.5
        
        # Place each atom in the grid
        for i in range(min(n_atoms, grid_size * grid_size)):
            row = i // grid_size
            col = i % grid_size
            
            # Reshape atom to 2D patch
            atom = normalized_dict[i].reshape(self.patch_size, self.patch_size)
            
            # Calculate position with 1-pixel gap between patches
            row_pos = row * (self.patch_size + 1)
            col_pos = col * (self.patch_size + 1)
            
            # Place atom in the grid
            grid_image[row_pos:row_pos + self.patch_size, 
                      col_pos:col_pos + self.patch_size] = atom
        
        # Remove axes and white space
        plt.axis('off')
        plt.imshow(grid_image, cmap='gray', interpolation='nearest')
        plt.title(f'K-SVD Dictionary (p={self.patch_size}, n={self.n_components}, s={self.sparsity})', 
                 fontsize=14)
        plt.tight_layout(pad=0)
        
        # Save high-quality PNG
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Dictionary visualization saved to {output_path}")
    
    def train_global(self, clean_images_dir, max_patches=30000, save_dict_path=None):
        """Train global dictionary efficiently."""
        print(f"Training global dictionary on {clean_images_dir}")
        
        image_files = glob.glob(os.path.join(clean_images_dir, "*.png"))
        if not image_files:
            raise ValueError(f"No images found in {clean_images_dir}")
        
        # Extract patches with larger stride for training efficiency
        all_patches = []
        patches_per_image = max_patches // len(image_files)
        
        for img_path in image_files[:50]:  # Limit to 50 images for efficiency
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            patches, _ = self.extract_patches_fast(img, stride=self.patch_size)
            
            # Random sampling for efficiency
            if len(patches) > patches_per_image:
                idx = np.random.choice(len(patches), patches_per_image, replace=False)
                patches = patches[idx]
            
            all_patches.append(patches)
        
        if not all_patches:
            raise ValueError("No valid patches extracted")
        
        combined_patches = np.vstack(all_patches)
        
        # Normalize patches
        patch_mean = np.mean(combined_patches, axis=1, keepdims=True)
        patch_std = np.std(combined_patches, axis=1, keepdims=True) + 1e-8
        patches_normalized = (combined_patches - patch_mean) / patch_std
        
        # Initialize dictionary
        dictionary = self.initialize_dictionary(patches_normalized)
        
        # Train K-SVD
        print(f"Training on {len(patches_normalized)} patches...")
        self.global_dictionary, _ = self.ksvd_core(patches_normalized, dictionary)
        
        # Save dictionary visualization if path provided
        if save_dict_path:
            self.visualize_dictionary(self.global_dictionary, save_dict_path)
        
        return self.global_dictionary
    
    def denoise(self, noisy_image):
        """Optimized denoising pipeline."""
        # Estimate noise level if adaptive
        if self.sigma_est == 'adaptive':
            sigma = self.estimate_noise_sigma(noisy_image)
        else:
            sigma = 25.0  # Default value
        
        # Extract patches efficiently
        patches, indices = self.extract_patches_fast(noisy_image)
        
        # Normalize patches
        patch_mean = np.mean(patches, axis=1, keepdims=True)
        patch_std = np.std(patches, axis=1, keepdims=True) + 1e-8
        patches_normalized = (patches - patch_mean) / patch_std
        
        # Use global dictionary or train local one
        if self.use_global and self.global_dictionary is not None:
            dictionary = self.global_dictionary
        else:
            # Quick training on the current image
            dictionary = self.initialize_dictionary(patches_normalized)
            dictionary, _ = self.ksvd_core(patches_normalized, dictionary, n_iter=3)
        
        # Sparse coding with adaptive sparsity based on noise level
        adaptive_sparsity = max(1, min(self.sparsity, int(self.sparsity * (25 / sigma))))
        coefficients = self.fast_omp_batch(patches_normalized, dictionary, adaptive_sparsity)
        
        # Reconstruct denoised patches
        denoised_patches_normalized = coefficients @ dictionary
        denoised_patches = denoised_patches_normalized * patch_std + patch_mean
        
        # Reconstruct image
        denoised_image = self.reconstruct_image_fast(denoised_patches, indices, noisy_image.shape)
        
        # Optimized post-processing
        denoised_image = cv2.GaussianBlur(denoised_image.astype(np.float32), (3, 3), 0.5)
        
        return np.clip(denoised_image, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    data_dir = "data"
    train_clean_dir = os.path.join(data_dir, "train_imgs", "clean_imgs")
    test_noisy_dir = os.path.join(data_dir, "test_imgs", "noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs", "clean_imgs")
    base_output_dir = "results/optimized_ksvd"
    
    # Efficient parameter configurations
    configs = [
        # Best for CT scans - balanced quality/speed
        {'patch_size': 5, 'n_components': 32, 'sparsity': 3, 'n_iter': 5, 'use_global': True},
        # Faster but still good quality
        {'patch_size': 4, 'n_components': 16, 'sparsity': 2, 'n_iter': 3, 'use_global': True},
        # Highest quality but slower
        {'patch_size': 6, 'n_components': 64, 'sparsity': 4, 'n_iter': 7, 'use_global': True}
    ]
    
    all_results = []
    
    for config_idx, config in enumerate(configs):
        print(f"\nTesting configuration {config_idx + 1}/{len(configs)}")
        
        # Create config-specific output directory
        config_name = f"config{config_idx}_p{config['patch_size']}_n{config['n_components']}_s{config['sparsity']}"
        output_dir = os.path.join(base_output_dir, config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dictionary visualization path
        dict_vis_path = os.path.join(output_dir, f"dictionary_visualization.png")
        
        # Create denoiser
        denoiser = OptimizedKSVDDenoiser(**config)
        
        # Train global dictionary if needed and save visualization
        if config['use_global']:
            denoiser.train_global(train_clean_dir, save_dict_path=dict_vis_path)
        else:
            # For local dictionary, we'll save visualization after processing the first image
            first_image_path = sorted(glob.glob(os.path.join(test_noisy_dir, "*.png")))[0]
            first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
            if first_image is not None:
                # Extract patches and train local dictionary
                patches, _ = denoiser.extract_patches_fast(first_image)
                patch_mean = np.mean(patches, axis=1, keepdims=True)
                patch_std = np.std(patches, axis=1, keepdims=True) + 1e-8
                patches_normalized = (patches - patch_mean) / patch_std
                
                dictionary = denoiser.initialize_dictionary(patches_normalized)
                local_dict, _ = denoiser.ksvd_core(patches_normalized, dictionary, n_iter=3)
                
                # Save visualization
                denoiser.visualize_dictionary(local_dict, dict_vis_path)
        
        # Process test images
        results = denoiser.batch_process(test_noisy_dir, test_clean_dir, output_dir)
        
        if not results.empty:
            # Save results
            results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
            
            # Print summary
            print(f"Results for {config_name}:")
            summary = results.groupby(['NoiseType']).mean(numeric_only=True)
            print(summary[['PSNR', 'SSIM', 'Time']])
            print("-" * 50)
            
            all_results.append(results)
    
    # Combine and analyze results
    if all_results:
        combined_results = pd.concat(all_results)
        combined_results.to_csv(os.path.join(base_output_dir, "combined_results.csv"), index=False)
        
        # Final summary
        print("\nOverall Optimized K-SVD Results:")
        summary = combined_results.groupby(['DenoiserName']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])