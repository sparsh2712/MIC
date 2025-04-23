import os
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils.base_denoiser import BaseDenoiser

class OptimizedBayesianMRFDenoiser(BaseDenoiser):
    def __init__(self, prior_type='huber', alpha=0.7, gamma=0.1, max_iterations=100):
        super().__init__(f"OptimizedBayesianMRF_{prior_type}_alpha{alpha}_gamma{gamma}")
        self.prior_type = prior_type
        self.alpha = alpha
        self.gamma = gamma
        self.max_iterations = max_iterations
    
    def objective_function(self, img_flat, shape, noisy_img_flat):
        """Combined objective function (negative log posterior)"""
        img = img_flat.reshape(shape)
        noisy_img = noisy_img_flat.reshape(shape)
        
        # Likelihood term (data fidelity)
        likelihood = np.sum((img - noisy_img)**2)
        
        # Prior term (regularization)
        prior = 0
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Calculate differences with neighbors
            shifted_img = np.roll(img, (dx, dy), axis=(0, 1))
            diff = img - shifted_img
            
            if self.prior_type == "quadratic":
                prior += np.sum(diff**2)
            elif self.prior_type == "huber":
                # Fixed Huber prior calculation - calculate both parts separately
                mask = np.abs(diff) <= self.gamma
                huber_small = 0.5 * (diff[mask]**2)
                huber_large = self.gamma * (np.abs(diff[~mask]) - 0.5 * self.gamma)
                
                # Add the sum of each part to the prior
                prior += np.sum(huber_small) + np.sum(huber_large)
            elif self.prior_type == "adaptive":
                prior += np.sum(self.gamma * np.abs(diff) - 
                              self.gamma**2 * np.log(1 + np.abs(diff)/self.gamma))
        
        # Combined posterior (weighted sum)
        return (1 - self.alpha) * likelihood + self.alpha * prior
    
    def objective_gradient(self, img_flat, shape, noisy_img_flat):
        """Gradient of the objective function"""
        img = img_flat.reshape(shape)
        noisy_img = noisy_img_flat.reshape(shape)
        
        # Likelihood gradient
        likelihood_grad = 2 * (img - noisy_img)
        
        # Prior gradient
        prior_grad = np.zeros_like(img)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            # Forward difference
            fwd_diff = img - np.roll(img, (-dx, -dy), axis=(0, 1))
            # Backward difference (already shifted)
            bwd_diff = img - np.roll(img, (dx, dy), axis=(0, 1))
            
            if self.prior_type == "quadratic":
                prior_grad += 2 * (fwd_diff + bwd_diff)
            elif self.prior_type == "huber":
                # Huber gradient for forward difference
                fwd_mask = np.abs(fwd_diff) <= self.gamma
                fwd_grad = np.zeros_like(fwd_diff)
                fwd_grad[fwd_mask] = fwd_diff[fwd_mask]
                fwd_grad[~fwd_mask] = self.gamma * np.sign(fwd_diff[~fwd_mask])
                
                # Huber gradient for backward difference
                bwd_mask = np.abs(bwd_diff) <= self.gamma
                bwd_grad = np.zeros_like(bwd_diff)
                bwd_grad[bwd_mask] = bwd_diff[bwd_mask]
                bwd_grad[~bwd_mask] = self.gamma * np.sign(bwd_diff[~bwd_mask])
                
                prior_grad += fwd_grad + bwd_grad
            elif self.prior_type == "adaptive":
                # Adaptive gradient
                fwd_grad = self.gamma * np.sign(fwd_diff) - \
                          (self.gamma**2 * np.sign(fwd_diff)) / (self.gamma + np.abs(fwd_diff))
                bwd_grad = self.gamma * np.sign(bwd_diff) - \
                          (self.gamma**2 * np.sign(bwd_diff)) / (self.gamma + np.abs(bwd_diff))
                
                prior_grad += fwd_grad + bwd_grad
        
        # Combined gradient
        total_grad = (1 - self.alpha) * likelihood_grad + self.alpha * prior_grad
        return total_grad.flatten()
    
    def denoise(self, noisy_image):
        """Main denoising function using L-BFGS-B optimizer"""
        # Convert to float for processing
        noisy_image_float = noisy_image.astype(np.float32)
        shape = noisy_image_float.shape
        
        # Initial guess is the noisy image
        x0 = noisy_image_float.flatten()
        noisy_flat = x0.copy()
        
        # Define bounds to keep pixel values in valid range
        bounds = [(0, 255) for _ in range(len(x0))]
        
        # Optimize using L-BFGS-B algorithm
        result = minimize(
            fun=self.objective_function,
            x0=x0,
            args=(shape, noisy_flat),
            method='L-BFGS-B',
            jac=self.objective_gradient,
            bounds=bounds,
            options={'maxiter': self.max_iterations, 'disp': False}
        )
        
        # Reshape the result back to image dimensions
        denoised_image_float = result.x.reshape(shape)
        
        # Clip and convert back to uint8
        denoised_image = np.clip(denoised_image_float, 0, 255).astype(np.uint8)
        return denoised_image

if __name__ == "__main__":
    data_dir = os.path.join("data")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/optimized_bayesian_mrf")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    # Parameter sets optimized for each prior type
    parameter_sets = [
        # Quadratic prior - only needs alpha (gamma not used)
        {'prior_type': 'quadratic', 'alpha': 0.3, 'gamma': 0.0},
        {'prior_type': 'quadratic', 'alpha': 0.5, 'gamma': 0.0},
        {'prior_type': 'quadratic', 'alpha': 0.7, 'gamma': 0.0},
        
        # Huber prior - needs both alpha and gamma
        # {'prior_type': 'huber', 'alpha': 0.5, 'gamma': 0.05},
        # {'prior_type': 'huber', 'alpha': 0.5, 'gamma': 0.1},
        # {'prior_type': 'huber', 'alpha': 0.7, 'gamma': 0.1},
        
        # # Adaptive prior - needs both alpha and gamma
        # {'prior_type': 'adaptive', 'alpha': 0.5, 'gamma': 0.1},
        # {'prior_type': 'adaptive', 'alpha': 0.7, 'gamma': 0.1},
        # {'prior_type': 'adaptive', 'alpha': 0.7, 'gamma': 0.2}
    ]
    
    all_results = []
    
    # Process each parameter set
    for params in parameter_sets:
        prior_type = params['prior_type']
        alpha = params['alpha']
        gamma = params['gamma']
        
        # Create specific output directory for this combination
        if prior_type == 'quadratic':
            param_dir = os.path.join(base_output_dir, f"{prior_type}_alpha{alpha}")
        else:
            param_dir = os.path.join(base_output_dir, f"{prior_type}_alpha{alpha}_gamma{gamma}")
        
        os.makedirs(param_dir, exist_ok=True)
        
        # Create specific results file
        param_results_csv = os.path.join(param_dir, "results.csv")
        
        denoiser = OptimizedBayesianMRFDenoiser(
            prior_type=prior_type, 
            alpha=alpha, 
            gamma=gamma
        )
        
        results = denoiser.batch_process(test_noisy_dir, test_clean_dir, param_dir)
        
        if not results.empty:
            # Save individual results
            results.to_csv(param_results_csv, index=False)
            
            # Print individual summary
            print(f"Optimized Bayesian MRF {prior_type}, α={alpha}" + 
                  (f", γ={gamma}" if prior_type != 'quadratic' else "") + 
                  " Results:")
            summary = results.groupby(['NoiseType']).mean(numeric_only=True)
            print(summary[['PSNR', 'SSIM', 'Time']])
            print("-" * 50)
            
            all_results.append(results)
    
    # Make sure we have results before trying to concatenate
    if all_results:
        # Create base output directory if it doesn't exist
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Combine and save all results
        combined_results = pd.concat(all_results)
        combined_results.to_csv(combined_results_csv, index=False)
        
        # Overall summary by denoiser name and noise type
        print("Overall Optimized Bayesian MRF Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
        
        # Additional summary by parameters (averaged across noise types)
        print("\nSummary by Parameters (averaged across noise types):")
        param_summary = combined_results.groupby(['DenoiserName']).mean(numeric_only=True)
        print(param_summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")