import os
import random
import numpy as np
import cv2
from skimage import util

def add_gaussian_noise(image, mean=0, sigma=15):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float64)
    noisy_image_float = image.astype(np.float64) + noise
    noisy_image = np.clip(noisy_image_float, 0, 255).astype(np.uint8)
    return noisy_image, "gaussian"

def add_salt_pepper_noise(image, amount=0.05):
    noisy_image = util.random_noise(image, mode='s&p', amount=amount)
    noisy_image = np.array(255 * noisy_image, dtype=np.uint8)
    return noisy_image, "salt_pepper"

def add_poisson_noise(image):
    noisy_image = util.random_noise(image, mode='poisson')
    noisy_image = np.array(255 * noisy_image, dtype=np.uint8)
    return noisy_image, "poisson"

def add_speckle_noise(image):
    noisy_image = util.random_noise(image, mode='speckle')
    noisy_image = np.array(255 * noisy_image, dtype=np.uint8)
    return noisy_image, "speckle"

def add_rician_noise(image, sigma=25):
    noise_real = np.random.normal(0, sigma, image.shape).astype(np.float64)
    noise_imag = np.random.normal(0, sigma, image.shape).astype(np.float64)
    
    noisy_real = image.astype(np.float64) + noise_real
    noisy_imag = noise_imag
    
    noisy_image = np.sqrt(noisy_real**2 + noisy_imag**2)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image, "rician"

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    noise_functions = [
        add_gaussian_noise,
        add_salt_pepper_noise,
        add_poisson_noise,
        add_speckle_noise,
        add_rician_noise
    ]
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            
            image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            
            noise_func = random.choice(noise_functions)
            noisy_image, noise_type = noise_func(image)
            
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_{noise_type}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            cv2.imwrite(output_path, noisy_image)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python add_noise.py input_directory output_directory")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    process_images(input_dir, output_dir)