import os
import numpy as np
import cv2
from abc import ABC, abstractmethod
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import pandas as pd
from time import time

class BaseDenoiser(ABC):
    def __init__(self, name):
        self.name = name
        self.metrics = ['MSE', 'PSNR', 'SSIM', 'RMSE', 'SNR', 'Time']
    
    @abstractmethod
    def denoise(self, noisy_image):
        pass
    
    def process_image(self, noisy_path, clean_path, output_dir):
        noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        
        if noisy_image is None or clean_image is None:
            return None
        
        start_time = time()
        denoised_image = self.denoise(noisy_image)
        end_time = time()
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(noisy_path)
            output_path = os.path.join(output_dir, f"{self.name}_{filename}")
            cv2.imwrite(output_path, denoised_image)
        
        return self.calculate_metrics(clean_image, denoised_image, end_time - start_time)
    
    def calculate_metrics(self, clean_image, denoised_image, processing_time):
        mse = np.mean((clean_image.astype(np.float64) - denoised_image.astype(np.float64)) ** 2)
        rmse = np.sqrt(mse)
        
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        ssim = structural_similarity(clean_image, denoised_image)
        
        signal_power = np.mean(clean_image.astype(np.float64) ** 2)
        noise_power = mse
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'PSNR': psnr,
            'SSIM': ssim,
            'SNR': snr,
            'Time': processing_time
        }
    
    def batch_process(self, noisy_dir, clean_dir, output_dir=None):
        results = []
        
        for filename in os.listdir(noisy_dir):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            noisy_path = os.path.join(noisy_dir, filename)
            base_name = os.path.splitext(filename)[0].split('_')[0]
            clean_path = os.path.join(clean_dir, f"{base_name}.png")
            
            if not os.path.exists(clean_path):
                continue
            
            metrics = self.process_image(noisy_path, clean_path, output_dir)
            if metrics:
                metrics['DenoiserName'] = self.name
                metrics['ImageName'] = filename
                metrics['NoiseType'] = filename.split('_')[-1].split('.')[0]
                results.append(metrics)
        
        return pd.DataFrame(results)