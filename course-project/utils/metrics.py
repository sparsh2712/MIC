import numpy as np
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def calculate_metrics(clean_image, denoised_image):
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
        'SNR': snr
    }

