import os
import cv2
import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from scipy.sparse.linalg import spsolve
from scipy import sparse
from utils.base_denoiser import BaseDenoiser

class InternalDictionaryLearningDenoiser(BaseDenoiser):
    def __init__(self, patch_size=8, n_components=100, alpha=1.0, max_iter=100, stride=4, lambda_tv=0.1):
        super().__init__(f"InternalDictLearning_p{patch_size}_n{n_components}_a{alpha}_tv{lambda_tv}")
        self.patch_size = patch_size
        self.n_components = n_components  # Number of dictionary atoms
        self.alpha = alpha  # Sparsity regularization
        self.max_iter = max_iter
        self.stride = stride  # Stride for patch extraction
        self.lambda_tv = lambda_tv  # Weight for TV regularization to handle noise
        self.dictionary = None
    
    def total_variation(self, img):
        """Calculate total variation - used to identify less noisy patches"""
        dx = np.diff(img, axis=0)
        dy = np.diff(img, axis=1)
        
        # Pad to original size
        dx = np.vstack((dx, np.zeros((1, img.shape[1]))))
        dy = np.hstack((dy, np.zeros((img.shape[0], 1))))
        
        return np.sqrt(dx**2 + dy**2)
    
    def extract_good_patches(self, noisy_image, n_patches=5000):
        """Extract patches from noisy image, preferring those with lower noise"""
        # Calculate total variation to find smoother areas (likely less noisy)
        tv_map = self.total_variation(noisy_image)
        tv_map_blurred = cv2.GaussianBlur(tv_map, (5, 5), 0)
        
        # Extract all possible patches with stride
        patches = []
        positions = []
        tv_values = []
        
        for i in range(0, noisy_image.shape[0] - self.patch_size + 1, self.stride):
            for j in range(0, noisy_image.shape[1] - self.patch_size + 1, self.stride):
                patch = noisy_image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch.flatten())
                positions.append((i, j))
                
                # Calculate average TV value for this patch
                tv_value = np.mean(tv_map_blurred[i:i+self.patch_size, j:j+self.patch_size])
                tv_values.append(tv_value)
        
        patches = np.array(patches)
        tv_values = np.array(tv_values)
        
        # Select patches with lower TV values (smoother, less noisy patches)
        if len(patches) > n_patches:
            idx = np.argsort(tv_values)[:n_patches]
            return patches[idx]
        else:
            return patches
    
    def train(self, train_noisy_dir, patches_per_image=2000):
        """Learn dictionary from multiple noisy training images"""
        print(f"Training dictionary from noisy images in {train_noisy_dir}")
        all_patches = []
        
        # Process all image files in the directory
        image_files = glob.glob(os.path.join(train_noisy_dir, "*.png"))
        if not image_files:
            raise ValueError(f"No PNG images found in {train_noisy_dir}")
            
        print(f"Found {len(image_files)} training images")
        
        for img_path in image_files:
            noisy_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if noisy_img is None:
                continue
                
            # Extract good patches from this noisy image
            patches = self.extract_good_patches(noisy_img, n_patches=patches_per_image)
            all_patches.append(patches)
            print(f"Extracted {len(patches)} patches from {os.path.basename(img_path)}")
        
        # Combine patches from all images
        if not all_patches:
            raise ValueError("No valid patches extracted from training images")
            
        combined_patches = np.vstack(all_patches)
        print(f"Total patches for dictionary learning: {len(combined_patches)}")
        
        # Normalize patches
        patches_mean = np.mean(combined_patches, axis=1).reshape(-1, 1)
        patches_centered = combined_patches - patches_mean
        
        # Learn dictionary using sklearn
        dict_learning = DictionaryLearning(
            n_components=min(self.n_components, len(combined_patches)),
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_algorithm='lars',
            transform_algorithm='lasso_lars',
            transform_alpha=self.alpha,
            random_state=42
        )
        
        dict_learning.fit(patches_centered)
        self.dictionary = dict_learning.components_
        print(f"Dictionary learning completed with {len(self.dictionary)} atoms")
        
        return self.dictionary
    
    def sparse_code(self, patches, dictionary):
        """Find sparse representation of patches using the dictionary"""
        patches_mean = np.mean(patches, axis=1).reshape(-1, 1)
        patches_centered = patches - patches_mean
        
        # Create sparse coding solver matrix
        D = dictionary
        DTD = D.dot(D.T)
        L = sparse.eye(D.shape[0]) * self.alpha
        
        # Solve for each patch
        codes = np.zeros((patches.shape[0], D.shape[0]))
        
        for i in range(patches.shape[0]):
            x = patches_centered[i].reshape(-1, 1)
            Dx = D.dot(x)
            # Solve: (D^T D + alpha*I) z = D^T x
            codes[i] = spsolve(DTD + L, Dx).flatten()
        
        return codes, patches_mean
    
    def denoise(self, noisy_image):
        """Denoise the image using pre-trained dictionary"""
        if self.dictionary is None:
            # If no dictionary has been pre-trained, learn one from this image
            self.dictionary = self.learn_dictionary_from_noisy(noisy_image)
        
        # Extract all patches from noisy image
        noisy_patches = extract_patches_2d(noisy_image, (self.patch_size, self.patch_size))
        noisy_patches_shape = noisy_patches.shape
        noisy_patches = noisy_patches.reshape(noisy_patches.shape[0], -1)
        
        # Find sparse representation
        codes, patches_mean = self.sparse_code(noisy_patches, self.dictionary)
        
        # Reconstruct patches
        reconstructed_patches = codes.dot(self.dictionary) + patches_mean
        reconstructed_patches = reconstructed_patches.reshape(noisy_patches_shape)
        
        # Reconstruct full image
        denoised_image = reconstruct_from_patches_2d(reconstructed_patches, noisy_image.shape)
        
        # Apply light TV denoising as post-processing to remove artifacts
        if self.lambda_tv > 0:
            denoised_image = self.apply_tv_postprocessing(denoised_image, noisy_image)
        
        # Clip values and convert to uint8
        denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
        
        return denoised_image
    
    def learn_dictionary_from_noisy(self, noisy_image):
        """Learn dictionary from the noisy image itself (fallback method)"""
        # Extract patches with lower noise levels
        patches = self.extract_good_patches(noisy_image)
        
        if len(patches) < self.n_components:
            # If too few patches, reduce dictionary size
            n_components = max(len(patches) // 2, 10)
            print(f"Warning: Reduced dictionary size to {n_components} due to limited patches")
        else:
            n_components = self.n_components
        
        # Normalize patches
        patches_mean = np.mean(patches, axis=1).reshape(-1, 1)
        patches_centered = patches - patches_mean
        
        # Learn dictionary using sklearn
        dict_learning = DictionaryLearning(
            n_components=n_components,
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_algorithm='lars',
            transform_algorithm='lasso_lars',
            transform_alpha=self.alpha,
            random_state=42
        )
        
        dict_learning.fit(patches_centered)
        return dict_learning.components_
    
    def apply_tv_postprocessing(self, denoised_image, noisy_image, iterations=5):
        """Apply total variation regularization as post-processing"""
        # Simple gradient descent for TV minimization
        img_float = denoised_image.astype(np.float32)
        
        for _ in range(iterations):
            # Calculate gradient of TV term
            dx_forward = np.diff(img_float, axis=0, append=img_float[-1:])
            dy_forward = np.diff(img_float, axis=1, append=img_float[:,-1:])
            
            dx_backward = np.diff(img_float, axis=0, prepend=img_float[:1])
            dy_backward = np.diff(img_float, axis=1, prepend=img_float[:,:1])
            
            # Calculate divergence
            div_x = dx_forward - dx_backward
            div_y = dy_forward - dy_backward
            
            # Update image (gradient descent step)
            grad = div_x + div_y
            img_float = img_float - self.lambda_tv * grad
            
            # Data fidelity term (don't stray too far from original)
            img_float = img_float + 0.1 * (noisy_image.astype(np.float32) - img_float)
        
        return img_float

if __name__ == "__main__":
    data_dir = os.path.join("data")
    train_noisy_dir = os.path.join(data_dir, "train_imgs/noisy_imgs")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/internal_dictionary_learning")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    # Parameter combinations
    param_sets = [
        {'patch_size': 6, 'n_components': 100, 'alpha': 0.5, 'lambda_tv': 0.05},
        {'patch_size': 8, 'n_components': 150, 'alpha': 1.0, 'lambda_tv': 0.1},
        {'patch_size': 10, 'n_components': 200, 'alpha': 1.5, 'lambda_tv': 0.15}
    ]
    
    all_results = []
    
    for params in param_sets:
        patch_size = params['patch_size']
        n_components = params['n_components']
        alpha = params['alpha']
        lambda_tv = params['lambda_tv']
        
        # Create output directory for this parameter set
        param_dir = os.path.join(base_output_dir, f"p{patch_size}_n{n_components}_a{alpha}_tv{lambda_tv}")
        os.makedirs(param_dir, exist_ok=True)
        
        # Create specific results file
        param_results_csv = os.path.join(param_dir, "results.csv")
        
        # Initialize denoiser
        denoiser = InternalDictionaryLearningDenoiser(
            patch_size=patch_size,
            n_components=n_components,
            alpha=alpha,
            lambda_tv=lambda_tv
        )
        
        try:
            # Train dictionary on noisy training images
            denoiser.train(train_noisy_dir, patches_per_image=2000)
            
            # Process test images using the trained dictionary
            results = denoiser.batch_process(test_noisy_dir, test_clean_dir, param_dir)
            
            if not results.empty:
                # Save individual results
                results.to_csv(param_results_csv, index=False)
                
                # Print individual summary
                print(f"Internal Dictionary Learning Results (p={patch_size}, n={n_components}, α={alpha}, tv={lambda_tv}):")
                summary = results.groupby(['NoiseType']).mean(numeric_only=True)
                print(summary[['PSNR', 'SSIM', 'Time']])
                print("-" * 50)
                
                all_results.append(results)
        except Exception as e:
            print(f"Error processing parameter set {params}: {e}")
    
    # Make sure we have results before trying to concatenate
    if all_results:
        # Create base output directory if it doesn't exist
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Combine and save all results
        combined_results = pd.concat(all_results)
        combined_results.to_csv(combined_results_csv, index=False)
        
        # Overall summary
        print("Overall Internal Dictionary Learning Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")