import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils.base_denoiser import BaseDenoiser
import time
import random
import matplotlib.pyplot as plt # Added import

class CTDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, patch_size=128):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.image_pairs = []

        # Ensure directories exist
        if not os.path.isdir(noisy_dir):
            raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")
        if not os.path.isdir(clean_dir):
            raise FileNotFoundError(f"Clean directory not found: {clean_dir}")

        for filename in os.listdir(noisy_dir):
            # Use lower() for case-insensitive extension checking
            if filename.lower().endswith('.png'):
                noisy_path = os.path.join(noisy_dir, filename)
                base_name = filename.split("_")[0]
                clean_path = os.path.join(clean_dir, f"{base_name}.png")
                if os.path.exists(clean_path):
                    self.image_pairs.append((noisy_path, clean_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.image_pairs[idx]
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

        if noisy_img is None:
            raise ValueError(f"Could not read noisy image: {noisy_path}")
        if clean_img is None:
            raise ValueError(f"Could not read clean image: {clean_path}")

        # Ensure images are large enough for patch extraction
        h, w = noisy_img.shape
        if h < self.patch_size or w < self.patch_size:
             # Option 1: Resize (might introduce artifacts)
             # noisy_img = cv2.resize(noisy_img, (max(w, self.patch_size), max(h, self.patch_size)), interpolation=cv2.INTER_CUBIC)
             # clean_img = cv2.resize(clean_img, (max(w, self.patch_size), max(h, self.patch_size)), interpolation=cv2.INTER_CUBIC)
             # Option 2: Skip this image pair (safer)
             print(f"Warning: Image {noisy_path} ({h}x{w}) is smaller than patch size ({self.patch_size}). Skipping.")
             # Return a dummy tensor or handle appropriately in the training loop / dataloader collate_fn
             # For simplicity here, we might get an error later if not handled.
             # A robust way is to filter these out in __init__ or use a custom collate_fn.
             # Let's try to return the next valid item instead for simplicity here
             return self.__getitem__((idx + 1) % len(self)) # Simple fix, might lead to duplicates if many small images

        h, w = noisy_img.shape # Re-evaluate shape if resizing was done

        # Random patch extraction
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        noisy_patch = noisy_img[y:y+self.patch_size, x:x+self.patch_size]
        clean_patch = clean_img[y:y+self.patch_size, x:x+self.patch_size]

        # Normalize to [-1, 1]
        noisy_patch = (noisy_patch.astype(np.float32) / 127.5) - 1.0
        clean_patch = (clean_patch.astype(np.float32) / 127.5) - 1.0

        # Convert to tensor and add channel dimension
        noisy_tensor = torch.from_numpy(noisy_patch).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean_patch).unsqueeze(0)

        return noisy_tensor, clean_tensor


# --- Generator Class (Corrected Decoder Dimensions) ---
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder for 128x128 patches
        self.encoder1 = self.conv_block(1, 32)      # Output: [B, 32, 64, 64]
        self.encoder2 = self.conv_block(32, 64)     # Output: [B, 64, 32, 32]
        self.encoder3 = self.conv_block(64, 128)    # Output: [B, 128, 16, 16]
        self.encoder4 = self.conv_block(128, 256)   # Output: [B, 256, 8, 8]

        # Bridge - no downsampling
        self.bridge = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )  # Output: [B, 512, 8, 8]

        # Decoder with skip connections (Corrected Input Channels)
        # Input to decoder4 is 'b': [B, 512, 8, 8] -> Output: [B, 256, 16, 16]
        self.decoder4 = self.upconv_block(512, 256)
        # Input to decoder3 is cat(d4, e3): [B, 256+128, 16, 16] = [B, 384, 16, 16] -> Output: [B, 128, 32, 32]
        self.decoder3 = self.upconv_block(384, 128) # Corrected: 512 -> 384
        # Input to decoder2 is cat(d3, e2): [B, 128+64, 32, 32] = [B, 192, 32, 32] -> Output: [B, 64, 64, 64]
        self.decoder2 = self.upconv_block(192, 64)  # Corrected: 256 -> 192
        # Input to decoder1 is cat(d2, e1): [B, 64+32, 64, 64] = [B, 96, 64, 64] -> Output: [B, 32, 128, 128]
        self.decoder1 = self.upconv_block(96, 32)   # Corrected: 128 -> 96

        # Final output layer (Input channels should match output of decoder1)
        self.output = nn.Sequential(
            # The input channels here (32) now correctly match the output of decoder1
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh() # Output is in [-1, 1] range
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels), # InstanceNorm is common in GANs
            nn.LeakyReLU(0.2, inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            # Kernel size 4, stride 2, padding 1 is common for doubling resolution
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) # ReLU is common in generator decoders
        )

    def forward(self, x):
        # Input: [B, 1, 128, 128]

        # Encoder path
        e1 = self.encoder1(x)    # [B, 32, 64, 64]
        e2 = self.encoder2(e1)   # [B, 64, 32, 32]
        e3 = self.encoder3(e2)   # [B, 128, 16, 16]
        e4 = self.encoder4(e3)   # [B, 256, 8, 8]

        # Bridge
        b = self.bridge(e4)      # [B, 512, 8, 8]

        # Decoder with skip connections (Corrected Concatenation)
        d4_out = self.decoder4(b)                 # [B, 256, 16, 16]
        # Concatenate d4 output with e3 (same spatial size)
        d3_in = torch.cat([d4_out, e3], dim=1)    # [B, 256+128, 16, 16] = [B, 384, 16, 16]

        d3_out = self.decoder3(d3_in)             # [B, 128, 32, 32]
        # Concatenate d3 output with e2
        d2_in = torch.cat([d3_out, e2], dim=1)    # [B, 128+64, 32, 32] = [B, 192, 32, 32]

        d2_out = self.decoder2(d2_in)             # [B, 64, 64, 64]
        # Concatenate d2 output with e1
        d1_in = torch.cat([d2_out, e1], dim=1)    # [B, 64+32, 64, 64] = [B, 96, 64, 64]

        d1_out = self.decoder1(d1_in)             # [B, 32, 128, 128]

        # Final output layer
        output = self.output(d1_out)              # [B, 1, 128, 128]
        return output

# --- Discriminator Class (No changes needed, seems dimensionally correct) ---
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Input: [B, 2, 128, 128] (Concatenated noisy + real/fake)

        self.model = nn.Sequential(
            # Layer 1: [B, 2, 128, 128] -> [B, 32, 64, 64]
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: [B, 32, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: [B, 128, 16, 16] -> [B, 256, 15, 15] (Stride 1, Padding 1)
            # Output size = floor((16 + 2*1 - 4)/1 + 1) = 15
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5 (Output): [B, 256, 15, 15] -> [B, 1, 14, 14] (PatchGAN output)
            # Output size = floor((15 + 2*1 - 4)/1 + 1) = 14
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid() # Output probability map
        )

    def forward(self, x):
        # Input x shape: [B, 2, 128, 128]
        return self.model(x) # Output shape: [B, 1, 14, 14]


# --- cGANDenoiser Class (No changes needed in structure, assuming BaseDenoiser exists) ---
class cGANDenoiser(BaseDenoiser):
    def __init__(self, generator_path):
        super().__init__("cGAN_denoiser")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        # Load weights - ensure the path exists
        if not os.path.exists(generator_path):
             raise FileNotFoundError(f"Generator weights not found at {generator_path}")
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.generator.eval() # Set to evaluation mode
        self.patch_size = 128
        self.stride = 64  # 50% overlap

    @torch.no_grad() # Decorator for inference mode
    def denoise(self, noisy_image):
        """Denoise a single grayscale image using patch-based processing."""
        if noisy_image.ndim != 2:
            raise ValueError(f"Input image must be 2D grayscale. Got shape {noisy_image.shape}")
        if noisy_image.dtype != np.uint8:
             # Attempt to convert if it's a different type but looks like grayscale
             if noisy_image.max() <= 1.0 and noisy_image.min() >= 0.0:
                 print("Warning: Input image seems to be float in [0,1]. Converting to uint8 [0,255].")
                 noisy_image = (noisy_image * 255).astype(np.uint8)
             elif noisy_image.max() <= 255 and noisy_image.min() >= 0:
                 print("Warning: Input image is not uint8. Converting to uint8.")
                 noisy_image = noisy_image.astype(np.uint8)
             else:
                 raise ValueError(f"Input image type not uint8. Got {noisy_image.dtype}")


        h, w = noisy_image.shape

        # If image is smaller than patch size, pad it
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h > 0 or pad_w > 0:
            # Pad with reflection to minimize edge artifacts
            noisy_image_padded = cv2.copyMakeBorder(noisy_image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            h_pad, w_pad = noisy_image_padded.shape
            print(f"Padded image from {h}x{w} to {h_pad}x{w_pad}")
        else:
            noisy_image_padded = noisy_image
            h_pad, w_pad = h, w


        # Initialize output image and weight map for the padded size
        denoised_image_padded = np.zeros_like(noisy_image_padded, dtype=np.float32)
        weight_map = np.zeros_like(noisy_image_padded, dtype=np.float32)

        # Create a weight matrix for smooth blending (Gaussian)
        blend_weight = np.ones((self.patch_size, self.patch_size), dtype=np.float32)
        center_x, center_y = self.patch_size // 2, self.patch_size // 2
        sigma = self.patch_size / 4 # Adjust sigma for desired falloff
        for i in range(self.patch_size):
            for j in range(self.patch_size):
                dist_sq = (i - center_y)**2 + (j - center_x)**2
                blend_weight[i, j] = np.exp(-dist_sq / (2 * sigma**2))
        blend_weight /= blend_weight.max() # Normalize weight to have max 1


        # Process patches with stride
        for i in range(0, h_pad - self.patch_size + 1, self.stride):
            for j in range(0, w_pad - self.patch_size + 1, self.stride):
                # Extract patch
                patch = noisy_image_padded[i:i+self.patch_size, j:j+self.patch_size]

                # Normalize to [-1, 1]
                patch_normalized = (patch.astype(np.float32) / 127.5) - 1.0
                patch_tensor = torch.from_numpy(patch_normalized).unsqueeze(0).unsqueeze(0).to(self.device)

                # Denoise patch
                denoised_patch_tensor = self.generator(patch_tensor)

                # Convert back to image range [0, 255]
                denoised_patch = denoised_patch_tensor.squeeze().cpu().numpy()
                denoised_patch = (denoised_patch + 1.0) * 127.5

                # Accumulate patch with weights
                denoised_image_padded[i:i+self.patch_size, j:j+self.patch_size] += denoised_patch * blend_weight
                weight_map[i:i+self.patch_size, j:j+self.patch_size] += blend_weight

        # Handle right edge (if needed)
        if (w_pad - self.patch_size) % self.stride != 0:
            j = w_pad - self.patch_size
            for i in range(0, h_pad - self.patch_size + 1, self.stride):
                 patch = noisy_image_padded[i:i+self.patch_size, j:j+self.patch_size]
                 patch_normalized = (patch.astype(np.float32) / 127.5) - 1.0
                 patch_tensor = torch.from_numpy(patch_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
                 denoised_patch_tensor = self.generator(patch_tensor)
                 denoised_patch = denoised_patch_tensor.squeeze().cpu().numpy()
                 denoised_patch = (denoised_patch + 1.0) * 127.5
                 denoised_image_padded[i:i+self.patch_size, j:j+self.patch_size] += denoised_patch * blend_weight
                 weight_map[i:i+self.patch_size, j:j+self.patch_size] += blend_weight

        # Handle bottom edge (if needed)
        if (h_pad - self.patch_size) % self.stride != 0:
            i = h_pad - self.patch_size
            for j in range(0, w_pad - self.patch_size + 1, self.stride):
                patch = noisy_image_padded[i:i+self.patch_size, j:j+self.patch_size]
                patch_normalized = (patch.astype(np.float32) / 127.5) - 1.0
                patch_tensor = torch.from_numpy(patch_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
                denoised_patch_tensor = self.generator(patch_tensor)
                denoised_patch = denoised_patch_tensor.squeeze().cpu().numpy()
                denoised_patch = (denoised_patch + 1.0) * 127.5
                denoised_image_padded[i:i+self.patch_size, j:j+self.patch_size] += denoised_patch * blend_weight
                weight_map[i:i+self.patch_size, j:j+self.patch_size] += blend_weight

        # Handle bottom-right corner (if needed)
        if (w_pad - self.patch_size) % self.stride != 0 and (h_pad - self.patch_size) % self.stride != 0:
            i = h_pad - self.patch_size
            j = w_pad - self.patch_size
            patch = noisy_image_padded[i:i+self.patch_size, j:j+self.patch_size]
            patch_normalized = (patch.astype(np.float32) / 127.5) - 1.0
            patch_tensor = torch.from_numpy(patch_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            denoised_patch_tensor = self.generator(patch_tensor)
            denoised_patch = denoised_patch_tensor.squeeze().cpu().numpy()
            denoised_patch = (denoised_patch + 1.0) * 127.5
            denoised_image_padded[i:i+self.patch_size, j:j+self.patch_size] += denoised_patch * blend_weight
            weight_map[i:i+self.patch_size, j:j+self.patch_size] += blend_weight

        # Avoid division by zero - replace zero weights with a small epsilon or 1
        weight_map[weight_map == 0] = 1e-8 # Use a small epsilon
        # Normalize by accumulated weights
        denoised_image_norm = np.divide(denoised_image_padded, weight_map)

        # Crop back to original size if padding was applied
        if pad_h > 0 or pad_w > 0:
            denoised_image_final = denoised_image_norm[0:h, 0:w]
        else:
            denoised_image_final = denoised_image_norm

        # Clip and convert back to uint8
        denoised_image_final = np.clip(denoised_image_final, 0, 255).astype(np.uint8)

        return denoised_image_final


# --- Training Function (Minor adjustments for clarity and robustness) ---
def train_cgan(train_noisy_dir, train_clean_dir, output_base_dir, num_epochs=50, batch_size=4, lr=0.0002, lambda_L1=100.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Create output directory for this training run
    output_dir = os.path.join(output_base_dir, f"cgan_train_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss functions
    criterion_GAN = nn.BCELoss() # Binary Cross Entropy for GAN loss
    criterion_L1 = nn.L1Loss()   # L1 Loss for image similarity

    # Optimizers (Using Adam as specified)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # Dataset and dataloader
    try:
        dataset = CTDataset(train_noisy_dir, train_clean_dir)
        if len(dataset) == 0:
             print("Error: No image pairs found. Check dataset paths and file naming.")
             return None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True) # Use num_workers
    except FileNotFoundError as e:
        print(f"Error initializing dataset: {e}")
        return None

    print(f"Found {len(dataset)} training image pairs.")

    # Lists to store losses for plotting
    history = {'epoch': [], 'batch': [], 'loss_D': [], 'loss_G': [], 'loss_G_GAN': [], 'loss_G_L1': []}

    # Training loop
    start_time = time.time()
    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss_G_agg = 0.0
        epoch_loss_D_agg = 0.0
        generator.train() # Set generator to training mode
        discriminator.train() # Set discriminator to training mode

        for i, (noisy_images, clean_images) in enumerate(dataloader):
            if noisy_images is None or clean_images is None: # Handle potential None from dataset error
                 print(f"Skipping batch {i} due to data loading issue.")
                 continue

            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)

            # Get batch size dynamically
            current_batch_size = noisy_images.size(0)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real images
            real_concat = torch.cat([noisy_images, clean_images], dim=1) # Shape: [B, 2, 128, 128]
            pred_real = discriminator(real_concat)           # Shape: [B, 1, 14, 14]
            # Create real labels (all ones) matching discriminator output shape
            real_labels = torch.ones_like(pred_real, device=device)
            loss_D_real = criterion_GAN(pred_real, real_labels)

            # Fake images
            generated_images = generator(noisy_images) # Shape: [B, 1, 128, 128]
            # Detach generated images to avoid backpropagating through generator here
            fake_concat = torch.cat([noisy_images, generated_images.detach()], dim=1) # Shape: [B, 2, 128, 128]
            pred_fake = discriminator(fake_concat)           # Shape: [B, 1, 14, 14]
            # Create fake labels (all zeros) matching discriminator output shape
            fake_labels = torch.zeros_like(pred_fake, device=device)
            loss_D_fake = criterion_GAN(pred_fake, fake_labels)

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()

            # Generate images again (no detach this time)
            generated_images_for_G = generator(noisy_images) # Shape: [B, 1, 128, 128]
            fake_concat_for_G = torch.cat([noisy_images, generated_images_for_G], dim=1) # Shape: [B, 2, 128, 128]
            pred_fake_for_G = discriminator(fake_concat_for_G) # Shape: [B, 1, 14, 14]

            # Generator aims to make discriminator output 'real' (all ones)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, real_labels) # Reuse real_labels

            # L1 loss (encourages similarity to clean images)
            loss_G_L1 = criterion_L1(generated_images_for_G, clean_images)

            # Total generator loss
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # Record losses for batch
            loss_D_item = loss_D.item()
            loss_G_item = loss_G.item()
            epoch_loss_D_agg += loss_D_item
            epoch_loss_G_agg += loss_G_item

            history['epoch'].append(epoch + 1)
            history['batch'].append(i + 1)
            history['loss_D'].append(loss_D_item)
            history['loss_G'].append(loss_G_item)
            history['loss_G_GAN'].append(loss_G_GAN.item())
            history['loss_G_L1'].append(loss_G_L1.item())


            if (i + 1) % 50 == 0: # Print progress every 50 batches
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}/{len(dataloader)}] "
                      f"[D loss: {loss_D_item:.4f}] [G loss: {loss_G_item:.4f} "
                      f"(GAN: {loss_G_GAN.item():.4f}, L1: {loss_G_L1.item():.4f})]")

        # --- End of Epoch ---
        avg_loss_G = epoch_loss_G_agg / len(dataloader)
        avg_loss_D = epoch_loss_D_agg / len(dataloader)
        epoch_time = time.time() - epoch_start_time

        print("-" * 60)
        print(f"End of Epoch {epoch+1}/{num_epochs} \t Time: {epoch_time:.2f}s")
        print(f"  Average D loss: {avg_loss_D:.4f}")
        print(f"  Average G loss: {avg_loss_G:.4f}")
        print("-" * 60)

        # Save model checkpoint periodically
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = os.path.join(output_dir, f"generator_epoch_{epoch+1}.pth")
            torch.save(generator.state_dict(), checkpoint_path)
            print(f"Saved generator checkpoint to {checkpoint_path}")

            # Optional: Save discriminator too
            # torch.save(discriminator.state_dict(), os.path.join(output_dir, f"discriminator_epoch_{epoch+1}.pth"))

            # Optional: Save some sample images
            if 'generated_images' in locals() and current_batch_size > 0: # Check if var exists
                sample_idx = 0 # Save first image of the last batch
                noisy_sample = (noisy_images[sample_idx].squeeze().cpu().numpy() + 1.0) * 127.5
                clean_sample = (clean_images[sample_idx].squeeze().cpu().numpy() + 1.0) * 127.5
                gen_sample = (generated_images[sample_idx].squeeze().detach().cpu().numpy() + 1.0) * 127.5

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(noisy_sample, cmap='gray')
                axes[0].set_title('Noisy Patch')
                axes[0].axis('off')
                axes[1].imshow(gen_sample, cmap='gray')
                axes[1].set_title(f'Generated Patch (Epoch {epoch+1})')
                axes[1].axis('off')
                axes[2].imshow(clean_sample, cmap='gray')
                axes[2].set_title('Clean Patch')
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"sample_epoch_{epoch+1}.png"), dpi=150)
                plt.close(fig)


    # --- End of Training ---
    total_training_time = time.time() - start_time
    print(f"Finished Training. Total time: {total_training_time:.2f}s")

    # Save final model
    final_model_path = os.path.join(output_dir, "generator_final.pth")
    torch.save(generator.state_dict(), final_model_path)
    print(f"Saved final generator model to {final_model_path}")

    # --- Plotting and Saving Losses ---
    loss_df = pd.DataFrame(history)
    loss_df.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)

    # Calculate epoch averages for cleaner plots
    epoch_avg_losses = loss_df.groupby('epoch').mean(numeric_only=True)

    plt.figure(figsize=(12, 8))

    # Plot average D and G loss per epoch
    plt.subplot(2, 1, 1)
    plt.plot(epoch_avg_losses.index, epoch_avg_losses['loss_G'], 'b-', label='Avg Generator Loss (G)', linewidth=2)
    plt.plot(epoch_avg_losses.index, epoch_avg_losses['loss_D'], 'r-', label='Avg Discriminator Loss (D)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average cGAN Training Losses per Epoch')
    plt.legend()
    plt.grid(True)

    # Plot average G (GAN) and G (L1) loss per epoch
    plt.subplot(2, 1, 2)
    plt.plot(epoch_avg_losses.index, epoch_avg_losses['loss_G_GAN'], 'g-', label='Avg Generator GAN Loss', linewidth=2)
    plt.plot(epoch_avg_losses.index, epoch_avg_losses['loss_G_L1'], 'm-', label=f'Avg Generator L1 Loss (scaled by {lambda_L1})', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss Component')
    plt.title('Average Generator Loss Components per Epoch')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_plot_epochs.png'), dpi=300)
    print(f"Saved loss plot to {os.path.join(output_dir, 'loss_plot_epochs.png')}")
    plt.close()


    return generator, final_model_path # Return trained generator and path

# --- Main Execution Block ---
if __name__ == "__main__":
    # Define directories (MODIFY THESE PATHS)
    # It's safer to use absolute paths or ensure relative paths are correct
    base_data_dir = "data" # Assume 'data' folder is in the same directory as the script
    output_base_dir = "results" # Assume 'results' folder is in the same directory

    train_noisy_dir = os.path.join(base_data_dir, "dl_train_imgs", "noisy_imgs")
    train_clean_dir = os.path.join(base_data_dir, "dl_train_imgs", "clean_imgs")
    test_noisy_dir = os.path.join(base_data_dir, "test_imgs", "noisy_imgs")
    test_clean_dir = os.path.join(base_data_dir, "test_imgs", "clean_imgs")

    # Create base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)

    # Train the cGAN
    print("Starting cGAN training...")
    # Pass the base output directory, the function will create a timestamped subfolder
    trained_generator, final_generator_path = train_cgan(
        train_noisy_dir,
        train_clean_dir,
        output_base_dir, # Pass the base dir here
        num_epochs=50, # Adjust as needed
        batch_size=4,  # Adjust based on GPU memory
        lr=0.0002,
        lambda_L1=100.0
    )

    if trained_generator and final_generator_path:
        print(f"\nTraining finished. Final model saved at: {final_generator_path}")

        # Evaluate the trained model
        print("\nEvaluating trained model...")
        # Use the path to the final saved model
        denoiser = cGANDenoiser(final_generator_path)

        # Define the output directory for evaluation results (inside the training run folder)
        eval_output_dir = os.path.dirname(final_generator_path) # Get the timestamped folder

        # Ensure test directories exist before evaluation
        if not os.path.isdir(test_noisy_dir) or not os.path.isdir(test_clean_dir):
             print(f"Warning: Test directories not found. Skipping evaluation.")
             print(f"  Checked noisy: {test_noisy_dir}")
             print(f"  Checked clean: {test_clean_dir}")
        else:
             # Assuming BaseDenoiser has a 'batch_process' method that
             # takes noisy_dir, clean_dir, output_dir and returns a pandas DataFrame
             try:
                 results_df = denoiser.batch_process(test_noisy_dir, test_clean_dir, eval_output_dir)

                 # Save results DataFrame to CSV
                 results_csv = os.path.join(eval_output_dir, "evaluation_results.csv")
                 results_df.to_csv(results_csv, index=False)
                 print(f"Evaluation results saved to {results_csv}")

                 # Print summary (check if DataFrame is not empty)
                 if not results_df.empty:
                     print("\ncGAN Denoising Evaluation Summary:")
                     # Ensure columns exist before grouping/calculating mean
                     if all(col in results_df.columns for col in ['NoiseType', 'PSNR', 'SSIM', 'Time']):
                         summary = results_df.groupby(['NoiseType']).mean(numeric_only=True)
                         print(summary[['PSNR', 'SSIM', 'Time']])
                     else:
                         print("Could not generate summary, expected columns (NoiseType, PSNR, SSIM, Time) not found in results.")
                     print("\nFull Evaluation Results:")
                     print(results_df)
                 else:
                     print("Evaluation produced no results.")

             except AttributeError:
                  print("Error: The 'cGANDenoiser' (or its parent 'BaseDenoiser')")
                  print("       does not have a 'batch_process' method implemented.")
                  print("       Cannot perform batch evaluation.")
             except Exception as e:
                  print(f"An error occurred during evaluation: {e}")

    else:
        print("Training did not complete successfully. Skipping evaluation.")