import cv2
import numpy as np
import os
import pandas as pd
from utils.base_denoiser import BaseDenoiser

class MeanFilterDenoiser(BaseDenoiser):
    def __init__(self, kernel_size=3):
        super().__init__(f"MeanFilter_{kernel_size}x{kernel_size}")
        self.kernel_size = kernel_size
    
    def denoise(self, noisy_image):
        return cv2.blur(noisy_image, (self.kernel_size, self.kernel_size))

if __name__ == "__main__":
    data_dir = "data"
    test_noisy_dir = os.path.join(data_dir, "test_imgs/noisy_imgs")
    test_clean_dir = os.path.join(data_dir, "test_imgs/clean_imgs")
    output_dir = os.path.join("results/mean_filter")
    results_csv = os.path.join("results/mean_filter/results.csv")
    
    kernel_sizes = [3, 5, 7]
    all_results = []
    
    for kernel_size in kernel_sizes:
        denoiser = MeanFilterDenoiser(kernel_size=kernel_size)
        results = denoiser.batch_process(test_noisy_dir, test_clean_dir, output_dir)
        all_results.append(results)
    
    combined_results = pd.concat(all_results)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    combined_results.to_csv(results_csv, index=False)
    
    print("Mean Filter Denoising Results:")
    summary = combined_results.groupby(['DenoiserName', 'NoiseType']).mean(numeric_only=True)
    print(summary[['PSNR', 'SSIM', 'Time']])