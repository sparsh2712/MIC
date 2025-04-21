import os
import cv2
import numpy as np
import pandas as pd
from utils.base_denoiser import BaseDenoiser

class FourierFilterDenoiser(BaseDenoiser):
    def __init__(self, filter_type='gaussian', cutoff_freq=0.1):
        super().__init__(f"FourierFilter_{filter_type}_cutoff{cutoff_freq}")
        self.filter_type = filter_type
        self.cutoff_freq = cutoff_freq
    
    def create_filter(self, shape, cutoff, filter_type):
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols) * cols
        y = np.linspace(-0.5, 0.5, rows) * rows
        X, Y = np.meshgrid(x, y)
        
        # Calculate distance from center
        D = np.sqrt(X**2 + Y**2)
        
        # Max possible distance (used for normalization)
        D0 = cutoff * np.max(D)
        
        # Create filter based on specified type
        if filter_type == 'ideal':
            # Ideal low-pass filter (sharp cutoff)
            H = np.zeros(shape)
            H[D <= D0] = 1
        
        elif filter_type == 'butterworth':
            # Butterworth low-pass filter (smoother transition)
            n = 2  # Order of filter
            H = 1 / (1 + (D / D0) ** (2 * n))
        
        elif filter_type == 'gaussian':
            # Gaussian low-pass filter (gradual transition)
            H = np.exp(-(D**2) / (2 * (D0**2)))
        
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
        
        return H
    
    def denoise(self, noisy_image):
        # Convert to float for better precision in FFT
        img_float = noisy_image.astype(np.float32) / 255.0
        
        # Apply FFT to the image
        img_fft = np.fft.fft2(img_float)
        img_fft_shifted = np.fft.fftshift(img_fft)
        
        # Create the appropriate filter
        filter_mask = self.create_filter(
            img_float.shape, 
            self.cutoff_freq, 
            self.filter_type
        )
        
        # Apply filter in frequency domain
        img_filtered_shifted = img_fft_shifted * filter_mask
        img_filtered = np.fft.ifftshift(img_filtered_shifted)
        
        # Apply inverse FFT
        img_restored = np.fft.ifft2(img_filtered)
        img_restored = np.abs(img_restored)
        
        # Rescale to 0-255 and convert back to uint8
        img_restored = np.clip(img_restored * 255, 0, 255).astype(np.uint8)
        
        return img_restored

if __name__ == "__main__":
    data_dir = os.path.join("data")
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    base_output_dir = os.path.join("results/fourier_filter")
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")
    
    filter_types = ['ideal', 'butterworth', 'gaussian']
    cutoff_freqs = [0.1, 0.15, 0.2]
    all_results = []
    
    for filter_type in filter_types:
        for cutoff in cutoff_freqs:
            # Create specific output directory for this combination
            filter_dir = os.path.join(base_output_dir, f"{filter_type}_cutoff{cutoff}")
            os.makedirs(filter_dir, exist_ok=True)
            
            # Create specific results file
            filter_results_csv = os.path.join(filter_dir, "results.csv")
            
            denoiser = FourierFilterDenoiser(filter_type=filter_type, cutoff_freq=cutoff)
            results = denoiser.batch_process(test_noisy_dir, test_clean_dir, filter_dir)
            
            if not results.empty:
                # Save individual results
                results.to_csv(filter_results_csv, index=False)
                
                # Print individual summary
                print(f"Fourier Filter Type: {filter_type}, Cutoff: {cutoff} Results:")
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
        print("Overall Fourier Filter Denoising Results:")
        summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
        print(summary[['PSNR', 'SSIM', 'Time']])
        
        # Additional summary by filter type and cutoff (averaged across noise types)
        print("\nSummary by Parameters (averaged across noise types):")
        param_summary = combined_results.groupby(['DenoiserName']).mean(numeric_only=True)
        print(param_summary[['PSNR', 'SSIM', 'Time']])
    else:
        print("No results were generated. Please check if the input directories exist and contain images.")