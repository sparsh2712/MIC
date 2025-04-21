import os
import cv2
import numpy as np
import pandas as pd
import pywt
from utils.base_denoiser import BaseDenoiser

class WaveletFilterDenoiser(BaseDenoiser):
    def __init__(self, wavelet='db4', level=3, threshold_type='soft', threshold_mode='bayes'):
        super().__init__(f"WaveletFilter_{wavelet}_L{level}_{threshold_type}_{threshold_mode}")
        self.wavelet = wavelet
        self.level = level
        self.threshold_type = threshold_type  # 'soft' or 'hard'
        self.threshold_mode = threshold_mode  # 'universal', 'bayes', or 'sure'
    
    def soft_threshold(self, coeffs, threshold):
        # Keep sign, but shrink magnitude towards zero
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def hard_threshold(self, coeffs, threshold):
        # Set coefficients below threshold to zero
        coeffs_thresh = coeffs.copy()
        coeffs_thresh[np.abs(coeffs) < threshold] = 0
        return coeffs_thresh
    
    def universal_threshold(self, detail_coeffs, sigma=None):
        # VisuShrink - Universal threshold
        if sigma is None:
            # Estimate noise from highest frequency subband
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        
        N = np.prod(detail_coeffs.shape)
        return sigma * np.sqrt(2 * np.log(N))
    
    def bayes_threshold(self, detail_coeffs):
        # BayesShrink - Adaptive threshold based on Bayesian estimation
        # Estimate noise variance using median absolute deviation
        sigma = np.median(np.abs(detail_coeffs)) / 0.6745
        
        # Variance of signal without noise
        var_x = max(0, np.var(detail_coeffs) - sigma**2)
        
        if var_x == 0:
            return np.max(np.abs(detail_coeffs))
        
        return (sigma**2) / np.sqrt(var_x)
    
    def sure_threshold(self, detail_coeffs):
        # SureShrink - Minimizes Stein's Unbiased Risk Estimator
        N = len(detail_coeffs.flatten())
        
        # Sort squared coefficients
        sq_coeffs = np.sort(np.abs(detail_coeffs.flatten())**2)
        
        # Calculate risk
        risk = np.zeros_like(sq_coeffs)
        for i in range(N):
            risk[i] = (N - 2*(i+1) + np.sum(sq_coeffs[:i+1]) + (N-i-1)*sq_coeffs[i])
        
        # Find threshold that minimizes risk
        min_idx = np.argmin(risk)
        return np.sqrt(sq_coeffs[min_idx])
    
    def apply_threshold(self, coeffs):
        # Apply thresholding to wavelet detail coefficients
        c_a, details = coeffs[0], coeffs[1:]
        
        thresholded_details = []
        for level_details in details:
            thresholded_level = []
            for detail_coeff in level_details:
                # Determine threshold value based on mode
                if self.threshold_mode == 'universal':
                    threshold = self.universal_threshold(detail_coeff)
                elif self.threshold_mode == 'bayes':
                    threshold = self.bayes_threshold(detail_coeff)
                elif self.threshold_mode == 'sure':
                    threshold = self.sure_threshold(detail_coeff)
                else:
                    raise ValueError(f"Unsupported threshold mode: {self.threshold_mode}")
                
                # Apply thresholding (soft or hard)
                if self.threshold_type == 'soft':
                    thresholded_coeff = self.soft_threshold(detail_coeff, threshold)
                elif self.threshold_type == 'hard':
                    thresholded_coeff = self.hard_threshold(detail_coeff, threshold)
                else:
                    raise ValueError(f"Unsupported threshold type: {self.threshold_type}")
                
                thresholded_level.append(thresholded_coeff)
            
            thresholded_details.append(tuple(thresholded_level))
        
        # Return approximation and thresholded details
        return [c_a] + thresholded_details
    
    def denoise(self, noisy_image):
        # Convert to float for better precision
        img_float = noisy_image.astype(np.float32)
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(img_float, self.wavelet, level=self.level)
        
        # Apply thresholding to wavelet coefficients
        thresholded_coeffs = self.apply_threshold(coeffs)
        
        # Reconstruct image from thresholded coefficients
        denoised_float = pywt.waverec2(thresholded_coeffs, self.wavelet)
        
        # Handle potential shape differences in reconstruction
        if denoised_float.shape != noisy_image.shape:
            denoised_float = denoised_float[:noisy_image.shape[0], :noisy_image.shape[1]]
        
        # Convert back to uint8
        denoised_image = np.clip(denoised_float, 0, 255).astype(np.uint8)
        
        return denoised_image

if __name__ == "__main__":
    data_dir = os.path.join("data")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/wavelet_filter")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    # Parameters optimized for CT scan denoising
    wavelet_families = ['db4', 'sym4', 'bior4.4']  # Good for preserving edges in medical images
    levels = [3, 4]  # Decomposition levels
    threshold_types = ['soft', 'hard']
    threshold_modes = ['bayes', 'universal']  # BayesShrink is often better for CT scans
    
    all_results = []
    
    # Test only selected combinations to avoid combinatorial explosion
    test_combinations = [
        # Recommended for CT scans: db4 wavelet with BayesShrink soft thresholding
        {'wavelet': 'db4', 'level': 3, 'threshold_type': 'soft', 'threshold_mode': 'bayes'},
        {'wavelet': 'db4', 'level': 4, 'threshold_type': 'soft', 'threshold_mode': 'bayes'},
        
        # Alternative wavelets with same thresholding
        {'wavelet': 'sym4', 'level': 3, 'threshold_type': 'soft', 'threshold_mode': 'bayes'},
        {'wavelet': 'bior4.4', 'level': 3, 'threshold_type': 'soft', 'threshold_mode': 'bayes'},
        
        # Test hard thresholding
        {'wavelet': 'db4', 'level': 3, 'threshold_type': 'hard', 'threshold_mode': 'bayes'},
        
        # Test universal thresholding
        {'wavelet': 'db4', 'level': 3, 'threshold_type': 'soft', 'threshold_mode': 'universal'},
    ]
    
    for params in test_combinations:
        # Create directory name from parameters
        dir_name = f"{params['wavelet']}_L{params['level']}_{params['threshold_type']}_{params['threshold_mode']}"
        param_dir = os.path.join(base_output_dir, dir_name)
        os.makedirs(param_dir, exist_ok=True)
        
        # Create specific results file
        param_results_csv = os.path.join(param_dir, "results.csv")
        
        denoiser = WaveletFilterDenoiser(**params)
        results = denoiser.batch_process(test_noisy_dir, test_clean_dir, param_dir)
        
        if not results.empty:
            # Save individual results
            results.to_csv(param_results_csv, index=False)
            
            # Print individual summary
            print(f"Wavelet Filter Results for {dir_name}:")
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
        print("Overall Wavelet Filter Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
        
        # Additional summary by parameter (averaged across noise types)
        print("\nSummary by Parameters (averaged across noise types):")
        param_summary = combined_results.groupby(['DenoiserName']).mean(numeric_only=True)
        print(param_summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")