import os
import cv2
import numpy as np
import pandas as pd
from utils.base_denoiser import BaseDenoiser

class BilateralFilterDenoiser(BaseDenoiser):
    def __init__(self, diameter=9, sigma_color=75, sigma_space=75):
        super().__init__(f"BilateralFilter_d{diameter}_sColor{sigma_color}_sSpace{sigma_space}")
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
    def denoise(self, noisy_image):
        return cv2.bilateralFilter(noisy_image, self.diameter, self.sigma_color, self.sigma_space)

if __name__ == "__main__":
    data_dir = os.path.join("data")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/bilateral_filter")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    # Parameter sets for bilateral filter
    # Bilateral filter has three parameters:
    # - diameter: Diameter of each pixel neighborhood (must be odd)
    # - sigma_color: Filter sigma in the color space
    # - sigma_space: Filter sigma in the coordinate space
    
    diameters = [5, 9, 15]  # Diameter of the pixel neighborhood
    sigma_colors = [25, 50, 75]  # Filter sigma in the color space (higher preserves edges better)
    sigma_spaces = [25, 50, 75]  # Filter sigma in the coordinate space (higher blurs more)
    
    all_results = []
    
    # For medical images, test combinations that emphasize edge preservation
    test_combinations = [
        {'diameter': 5, 'sigma_color': 25, 'sigma_space': 25},
        {'diameter': 5, 'sigma_color': 75, 'sigma_space': 25},
        {'diameter': 9, 'sigma_color': 50, 'sigma_space': 50},
        {'diameter': 9, 'sigma_color': 75, 'sigma_space': 75},
        {'diameter': 15, 'sigma_color': 75, 'sigma_space': 50},
        {'diameter': 15, 'sigma_color': 50, 'sigma_space': 75}
    ]
    
    for params in test_combinations:
        diameter = params['diameter']
        sigma_color = params['sigma_color']
        sigma_space = params['sigma_space']
        
        # Create specific output directory for this combination
        param_dir = os.path.join(base_output_dir, f"d{diameter}_sColor{sigma_color}_sSpace{sigma_space}")
        os.makedirs(param_dir, exist_ok=True)
        
        # Create specific results file
        param_results_csv = os.path.join(param_dir, "results.csv")
        
        denoiser = BilateralFilterDenoiser(
            diameter=diameter,
            sigma_color=sigma_color,
            sigma_space=sigma_space
        )
        
        results = denoiser.batch_process(test_noisy_dir, test_clean_dir, param_dir)
        
        if not results.empty:
            # Save individual results
            results.to_csv(param_results_csv, index=False)
            
            # Print individual summary
            print(f"Bilateral Filter d={diameter}, σ_color={sigma_color}, σ_space={sigma_space} Results:")
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
        print("Overall Bilateral Filter Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
        
        # Additional summary by parameters (averaged across noise types)
        print("\nSummary by Parameters (averaged across noise types):")
        param_summary = combined_results.groupby(['DenoiserName']).mean(numeric_only=True)
        print(param_summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")