import os
import cv2
import numpy as np
import pandas as pd
from utils.base_denoiser import BaseDenoiser

class GaussianFilterDenoiser(BaseDenoiser):
    def __init__(self, kernel_size=3, sigma=1.0):
        super().__init__(f"GaussianFilter_{kernel_size}x{kernel_size}_sigma{sigma}")
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def denoise(self, noisy_image):
        # OpenCV's GaussianBlur requires kernel_size to be odd
        if self.kernel_size % 2 == 0:
            kernel_size = self.kernel_size + 1
        else:
            kernel_size = self.kernel_size
            
        return cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), self.sigma)

if __name__ == "__main__":
    data_dir = os.path.join("data")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/gaussian_filter")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    kernel_sizes = [3, 5, 7]
    sigma_values = [0.5, 1.0, 1.5]
    all_results = []
    
    for kernel_size in kernel_sizes:
        for sigma in sigma_values:
            # Create specific output directory for this combination
            kernel_sigma_dir = os.path.join(base_output_dir, f"{kernel_size}x{kernel_size}_sigma{sigma}")
            os.makedirs(kernel_sigma_dir, exist_ok=True)
            
            # Create specific results file
            kernel_results_csv = os.path.join(kernel_sigma_dir, "results.csv")
            
            denoiser = GaussianFilterDenoiser(kernel_size=kernel_size, sigma=sigma)
            results = denoiser.batch_process(test_noisy_dir, test_clean_dir, kernel_sigma_dir)
            
            if not results.empty:
                # Save individual results
                results.to_csv(kernel_results_csv, index=False)
                
                # Print individual summary
                print(f"Gaussian Filter {kernel_size}x{kernel_size}, σ={sigma} Results:")
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
        print("Overall Gaussian Filter Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
        
        # Additional summary by kernel size and sigma (averaged across noise types)
        print("\nSummary by Parameters (averaged across noise types):")
        param_summary = combined_results.groupby(['DenoiserName']).mean(numeric_only=True)
        print(param_summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")