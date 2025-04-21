import os
import cv2
import numpy as np
import pandas as pd
from utils.base_denoiser import BaseDenoiser

class MedianFilterDenoiser(BaseDenoiser):
    def __init__(self, kernel_size=3):
        super().__init__(f"MedianFilter_{kernel_size}x{kernel_size}")
        self.kernel_size = kernel_size
    
    def denoise(self, noisy_image):
        # OpenCV's medianBlur requires kernel_size to be odd and > 1
        if self.kernel_size % 2 == 0:
            kernel_size = self.kernel_size + 1
        else:
            kernel_size = self.kernel_size
            
        return cv2.medianBlur(noisy_image, kernel_size)

if __name__ == "__main__":
    data_dir = os.path.join("data")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/median_filter")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    kernel_sizes = [3, 5, 7]
    all_results = []
    
    for kernel_size in kernel_sizes:
        # Create specific output directory for this kernel size
        kernel_output_dir = os.path.join(base_output_dir, f"{kernel_size}x{kernel_size}")
        os.makedirs(kernel_output_dir, exist_ok=True)
        
        # Create specific results file for this kernel size
        kernel_results_csv = os.path.join(kernel_output_dir, "results.csv")
        
        denoiser = MedianFilterDenoiser(kernel_size=kernel_size)
        results = denoiser.batch_process(test_noisy_dir, test_clean_dir, kernel_output_dir)
        
        if not results.empty:
            # Save individual kernel results
            results.to_csv(kernel_results_csv, index=False)
            
            # Print individual kernel summary
            print(f"Median Filter {kernel_size}x{kernel_size} Results:")
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
        
        print("Overall Median Filter Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")