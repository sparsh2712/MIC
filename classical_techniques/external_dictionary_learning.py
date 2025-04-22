import os
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import DictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from scipy.sparse.linalg import spsolve
from scipy import sparse
import glob
from time import time # Import time for simple duration tracking if needed

# Assuming 'utils.base_denoiser' exists and defines BaseDenoiser correctly
# If not, you might need to create a placeholder or the actual class
try:
    from utils.base_denoiser import BaseDenoiser
except ImportError:
    print("Warning: utils.base_denoiser not found. Using a placeholder BaseDenoiser.")
    # Placeholder BaseDenoiser if the original is not available
    class BaseDenoiser:
        def __init__(self, name):
            self.name = name
            print(f"Placeholder BaseDenoiser initialized with name: {self.name}")

        def calculate_metrics(self, clean_img, denoised_img):
            # Placeholder metrics - replace with actual calculations (e.g., PSNR, SSIM)
            print("Placeholder: Calculating metrics...")
            psnr = 30.0 # Dummy value
            ssim = 0.9  # Dummy value
            return psnr, ssim

        def batch_process(self, noisy_dir, clean_dir, output_dir):
            print(f"\nStarting batch processing...")
            print(f"  Noisy images directory: {noisy_dir}")
            print(f"  Clean images directory: {clean_dir}")
            print(f"  Output directory for denoised images: {output_dir}")

            results = []
            noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.png"))) # Ensure consistent order

            if not noisy_files:
                print(f"Warning: No noisy images found in {noisy_dir}")
                return pd.DataFrame()

            for noisy_path in noisy_files:
                base_name = os.path.basename(noisy_path)
                # Infer noise type from filename if possible (e.g., 'image_gaussian_sigma25.png')
                parts = base_name.split('_')
                noise_type = "Unknown"
                if len(parts) > 1:
                     # Attempt to find common noise types in filename parts
                    potential_noise = parts[1].split('.')[0] # Handle cases like 'gaussian.png'
                    if potential_noise.lower() in ['gaussian', 'salt', 'pepper', 'sp', 'poisson']:
                         noise_type = potential_noise
                    elif potential_noise.startswith('sigma'): # e.g. sigma25
                         noise_type = f"Gaussian_{potential_noise}"


                print(f"\nProcessing image: {base_name} (Noise Type: {noise_type})")

                noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
                if noisy_img is None:
                    print(f"  Warning: Could not read noisy image: {noisy_path}")
                    continue

                # Corresponding clean image path
                clean_path = os.path.join(clean_dir, base_name) # Assumes clean images have the same names
                clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
                if clean_img is None:
                    print(f"  Warning: Could not find or read corresponding clean image: {clean_path}")
                    # Decide if you want to skip or continue without metrics
                    # continue

                start_time = time()
                denoised_img = self.denoise(noisy_img) # Call the specific denoise method
                end_time = time()
                processing_time = end_time - start_time
                print(f"  Denoising finished in {processing_time:.4f} seconds.")

                # Save denoised image
                denoised_output_path = os.path.join(output_dir, base_name)
                cv2.imwrite(denoised_output_path, denoised_img)
                print(f"  Denoised image saved to: {denoised_output_path}")

                # Calculate metrics if clean image is available
                psnr, ssim = -1.0, -1.0 # Default values if clean image missing
                if clean_img is not None:
                   try:
                       # Make sure calculate_metrics exists and works
                       psnr, ssim = self.calculate_metrics(clean_img, denoised_img)
                       print(f"  Metrics calculated: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
                   except AttributeError:
                       print("  Warning: `calculate_metrics` method not found in BaseDenoiser. Skipping metrics.")
                   except Exception as e:
                       print(f"  Error calculating metrics for {base_name}: {e}")

                results.append({
                    'DenoiserName': self.name,
                    'FileName': base_name,
                    'NoiseType': noise_type,
                    'PSNR': psnr,
                    'SSIM': ssim,
                    'Time': processing_time
                })

            print("\nBatch processing finished.")
            return pd.DataFrame(results)

# --- Your ExternalDictionaryLearningDenoiser Class ---

class ExternalDictionaryLearningDenoiser(BaseDenoiser):
    def __init__(self, patch_size=8, n_components=256, alpha=1.0, max_iter=100, n_training_patches=10000):
        super().__init__(f"ExternalDictLearning_p{patch_size}_n{n_components}_a{alpha}")
        self.patch_size = patch_size
        self.n_components = n_components  # Number of dictionary atoms
        self.alpha = alpha  # Sparsity regularization
        self.max_iter = max_iter
        self.n_training_patches = n_training_patches
        self.dictionary = None
        # Print parameters on initialization
        print(f"Initializing ExternalDictionaryLearningDenoiser:")
        print(f"  Patch Size: {self.patch_size}")
        print(f"  Dictionary Components (Atoms): {self.n_components}")
        print(f"  Sparsity Alpha: {self.alpha}")
        print(f"  Max Iterations (Dict Learn): {self.max_iter}")
        print(f"  Target Training Patches: {self.n_training_patches}")

    def train(self, clean_images_dir):
        """Train dictionary from clean images"""
        print(f"\n--- Starting Dictionary Training ---")
        print(f"Using clean images from: {clean_images_dir}")
        start_train_time = time()

        # Collect patches from clean images
        all_patches = []
        image_files = glob.glob(os.path.join(clean_images_dir, "*.png"))

        if not image_files:
             raise ValueError(f"No PNG images found for training in directory: {clean_images_dir}")

        print(f"Found {len(image_files)} potential training images.")
        patches_per_image = max(1, self.n_training_patches // len(image_files)) # Ensure at least 1 patch per image
        print(f"Attempting to extract approx. {patches_per_image} patches per image.")

        actual_patches_collected = 0
        for img_path in image_files:
            # print(f"  Reading image: {os.path.basename(img_path)}") # Can be verbose
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  Warning: Could not read image {img_path}, skipping.")
                continue

            # Extract patches from this image
            try:
                patches = extract_patches_2d(img, (self.patch_size, self.patch_size),
                                             max_patches=patches_per_image, random_state=42) # Added random state for consistency
                if patches.shape[0] > 0:
                    all_patches.append(patches)
                    actual_patches_collected += patches.shape[0]
                # print(f"    Extracted {patches.shape[0]} patches.") # Can be verbose
            except Exception as e:
                 print(f"  Warning: Error extracting patches from {img_path}: {e}")


        # Combine patches from all images
        if not all_patches:
            raise ValueError("No valid patches could be extracted from the training images.")

        patches = np.vstack([p.reshape(p.shape[0], -1) for p in all_patches])
        print(f"Total patches collected for training: {patches.shape[0]}")

        # Normalize patches (mean subtraction)
        print("Normalizing patches (mean subtraction)...")
        patches_mean = np.mean(patches, axis=1, keepdims=True) # Keep dims for broadcasting
        patches_centered = patches - patches_mean
        print("Patch normalization complete.")

        # Learn dictionary using sklearn's dictionary learning
        print(f"Learning dictionary from {patches.shape[0]} patches...")
        print(f"  Parameters: n_components={self.n_components}, alpha={self.alpha}, max_iter={self.max_iter}")
        dict_learning = DictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha, # Regularization parameter for the dictionary learning
            max_iter=self.max_iter,
            fit_algorithm='lars',       # Algorithm to fit the dictionary
            transform_algorithm='lasso_lars', # Algorithm to find sparse codes during training/transform
            transform_alpha=self.alpha, # Regularization for the sparse coding step within fit
            random_state=42,
            verbose=1 # Add verbosity to sklearn's DictionaryLearning
        )

        dict_learning.fit(patches_centered)
        self.dictionary = dict_learning.components_
        end_train_time = time()
        print(f"Dictionary learning completed in {end_train_time - start_train_time:.2f} seconds.")
        print(f"Learned dictionary shape: {self.dictionary.shape}") # (n_components, n_features=patch_size*patch_size)
        print("--- Dictionary Training Finished ---")

        return self.dictionary

    def sparse_code(self, patches, dictionary):
        """Find sparse representation of patches using the dictionary"""
        # This method is called per image during denoise, print statements here can be very verbose.
        # Add prints only if debugging this specific part.
        # print(f"  Calculating sparse codes for {patches.shape[0]} patches...")
        patches_mean = np.mean(patches, axis=1, keepdims=True) # Keep dims
        patches_centered = patches - patches_mean

        # Using sklearn's transform method is often more optimized and robust
        # than manual solving, especially for algorithms like LassoLARS.
        # Re-initialize a transform object (or use the trained one if settings match)
        # Note: The alpha here controls the sparsity of the *codes*, not the dictionary learning itself.
        # We use self.alpha consistent with the __init__ and training transform_alpha.
        sparse_coder = DictionaryLearning(
             n_components=self.n_components,
             alpha=self.alpha, # Use the same alpha for consistency, or define a separate transform_alpha if needed
             fit_algorithm='lars', # Not used for transform, but needed for init
             transform_algorithm='lasso_lars',
             transform_alpha=self.alpha, # Controls sparsity of the codes
             random_state=42
        )
        # Set the learned dictionary to the coder
        sparse_coder.components_ = dictionary

        # Calculate sparse codes
        codes = sparse_coder.transform(patches_centered)
        # print(f"  Sparse coding complete. Code shape: {codes.shape}")

        # --- Manual Sparse Coding (Alternative - keep for reference or specific needs) ---
        # print("  Using manual sparse coding solver...")
        # D = dictionary.T # Dictionary atoms as columns (n_features x n_components)
        # DTD = D.T @ D    # (n_components x n_components)
        # L = sparse.eye(self.n_components) * self.alpha # Regularization matrix

        # codes = np.zeros((patches.shape[0], self.n_components))
        # for i in range(patches.shape[0]):
        #     x = patches_centered[i].reshape(-1, 1) # (n_features x 1)
        #     DTx = D.T @ x # (n_components x 1)
        #     # Solve: (D^T D + alpha*I) z = D^T x
        #     # Need try-except block for potential solver issues
        #     try:
        #         codes[i] = spsolve(DTD + L, DTx) # Solves for z (sparse code)
        #     except Exception as e:
        #         print(f"    Warning: spsolve failed for patch {i}: {e}. Setting code to zero.")
        #         codes[i] = np.zeros(self.n_components)
        # print("  Manual sparse coding complete.")
        # --- End Manual Sparse Coding ---

        return codes, patches_mean

    def denoise(self, noisy_image):
        """Denoise using learned dictionary"""
        # This is called within batch_process, which already prints image info
        print(f"  Starting denoising process for the current image...")
        denoise_start_time = time()

        if self.dictionary is None:
            raise ValueError("Dictionary not trained. Call train() first.")

        # 1. Extract patches from noisy image
        print(f"    Extracting patches (size: {self.patch_size}x{self.patch_size})...")
        noisy_patches = extract_patches_2d(noisy_image, (self.patch_size, self.patch_size))
        noisy_patches_shape = noisy_patches.shape
        noisy_patches_flat = noisy_patches.reshape(noisy_patches.shape[0], -1)
        print(f"    Extracted {noisy_patches_flat.shape[0]} patches.")

        # 2. Find sparse representation
        print(f"    Finding sparse representations using the learned dictionary (shape: {self.dictionary.shape})...")
        codes, patches_mean = self.sparse_code(noisy_patches_flat, self.dictionary)
        print(f"    Calculated sparse codes (shape: {codes.shape}).")

        # 3. Reconstruct patches from codes and dictionary
        print("    Reconstructing patches from codes and dictionary...")
        reconstructed_patches_flat = codes.dot(self.dictionary) + patches_mean # Add back mean
        reconstructed_patches = reconstructed_patches_flat.reshape(noisy_patches_shape)
        print("    Patch reconstruction complete.")

        # 4. Reconstruct full image from patches
        print("    Reconstructing full image from overlapping patches...")
        denoised_image = reconstruct_from_patches_2d(reconstructed_patches, noisy_image.shape)
        print("    Full image reconstruction complete.")

        # 5. Clip values and convert to uint8
        print("    Clipping values to [0, 255] and converting to uint8.")
        denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)

        denoise_end_time = time()
        print(f"  Denoising for this image finished in {denoise_end_time - denoise_start_time:.4f} seconds.")

        return denoised_image

if __name__ == "__main__":
    print("--- Script Execution Started ---")

    # Define directories
    base_data_dir = "data" # Base directory for data
    train_clean_dir = os.path.join(base_data_dir, "train_imgs", "clean_imgs")  # Directory with clean training images
    test_noisy_dir = os.path.join(base_data_dir, "test_imgs", "noisy_imgs")
    test_clean_dir = os.path.join(base_data_dir, "test_imgs", "clean_imgs")
    base_output_dir = os.path.join("results", "external_dictionary_learning") # Base for results
    combined_results_csv = os.path.join(base_output_dir, "combined_results.csv")

    print("Configuration:")
    print(f"  Training Data (Clean): {train_clean_dir}")
    print(f"  Test Data (Noisy):   {test_noisy_dir}")
    print(f"  Test Data (Clean):   {test_clean_dir}")
    print(f"  Base Output Dir:     {base_output_dir}")

    # Parameter combinations to test
    param_sets = [
        {'patch_size': 6, 'n_components': 128, 'alpha': 0.5},
        {'patch_size': 8, 'n_components': 256, 'alpha': 1.0},
        # {'patch_size': 10, 'n_components': 512, 'alpha': 1.5} # Uncomment to add more sets
    ]
    print(f"\nParameter sets to evaluate: {param_sets}")

    all_results = [] # List to store DataFrames from each parameter set

    # Loop through each parameter set
    for i, params in enumerate(param_sets):
        print(f"\n===== Processing Parameter Set {i+1}/{len(param_sets)}: {params} =====")
        patch_size = params['patch_size']
        n_components = params['n_components']
        alpha = params['alpha']

        # Create specific output directory for this parameter set's results and denoised images
        param_dir_name = f"p{patch_size}_n{n_components}_a{alpha}"
        param_output_dir = os.path.join(base_output_dir, param_dir_name)
        os.makedirs(param_output_dir, exist_ok=True)
        print(f"Output directory for this set: {param_output_dir}")

        # Define path for the CSV results for this specific parameter set
        param_results_csv = os.path.join(param_output_dir, "results.csv")

        # Initialize the denoiser with current parameters
        print("\nInitializing denoiser instance...")
        denoiser = ExternalDictionaryLearningDenoiser(
            patch_size=patch_size,
            n_components=n_components,
            alpha=alpha
            # max_iter and n_training_patches use defaults unless specified in params
        )

        try:
            # Train dictionary on clean images
            print(f"\nStarting training phase for parameter set {i+1}...")
            denoiser.train(train_clean_dir) # Train method now has internal prints

            # Process test images using the trained denoiser
            print(f"\nStarting testing (batch processing) phase for parameter set {i+1}...")
            # The batch_process method (from placeholder or actual BaseDenoiser) will handle image loops and denoise calls
            results_df = denoiser.batch_process(test_noisy_dir, test_clean_dir, param_output_dir) # Pass output dir for denoised images

            if not results_df.empty:
                # Save results for this specific parameter set
                print(f"\nSaving results for this parameter set to: {param_results_csv}")
                results_df.to_csv(param_results_csv, index=False)

                # Print summary for this parameter set
                print(f"\n--- Summary for Parameter Set: {params} ---")
                # Group by NoiseType if column exists, otherwise just print means
                if 'NoiseType' in results_df.columns:
                     summary = results_df.groupby(['NoiseType']).agg(
                         Avg_PSNR=('PSNR', 'mean'),
                         Avg_SSIM=('SSIM', 'mean'),
                         Avg_Time=('Time', 'mean'),
                         Num_Images=('FileName', 'count') # Count images per noise type
                     )
                     # Calculate overall mean row
                     overall_mean = results_df[['PSNR', 'SSIM', 'Time']].mean().rename(lambda x: f'Avg_{x}')
                     overall_mean['Num_Images'] = results_df['FileName'].count()
                     summary.loc['Overall Avg'] = overall_mean # Add overall average row
                else:
                     # Fallback if NoiseType wasn't determined
                     summary = results_df[['PSNR', 'SSIM', 'Time']].mean().to_frame('Average')

                # Format summary for printing
                pd.options.display.float_format = '{:.4f}'.format
                print(summary)
                print("-" * 60)

                all_results.append(results_df) # Add the DataFrame to the list for final combination
            else:
                print("No results generated for this parameter set (batch_process returned empty).")

        except FileNotFoundError as fnf_error:
             print(f"\nError processing parameter set {params}: {fnf_error}")
             print("Please ensure the specified image directories exist and contain images.")
        except ValueError as val_error:
             print(f"\nError processing parameter set {params}: {val_error}")
        except Exception as e:
            print(f"\nAn unexpected error occurred processing parameter set {params}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for unexpected errors

    # --- Post-Loop Processing ---
    print("\n===== Finished processing all parameter sets =====")

    # Make sure we have results before trying to concatenate
    if all_results:
        # Create base output directory if it doesn't exist (important if only errors occurred before)
        os.makedirs(base_output_dir, exist_ok=True)

        # Combine and save all results
        print("\nCombining results from all successful parameter sets...")
        combined_results = pd.concat(all_results, ignore_index=True) # ignore_index resets index for the combined df
        print(f"Saving combined results to: {combined_results_csv}")
        combined_results.to_csv(combined_results_csv, index=False)

        # Print Overall summary grouped by DenoiserName (which includes params) and NoiseType
        print("\n--- Overall Combined Denoising Results Summary ---")
        if 'DenoiserName' in combined_results.columns and 'NoiseType' in combined_results.columns:
            overall_summary = combined_results.groupby(['DenoiserName', 'NoiseType']).agg(
                 Avg_PSNR=('PSNR', 'mean'),
                 Avg_SSIM=('SSIM', 'mean'),
                 Avg_Time=('Time', 'mean'),
                 Num_Images=('FileName', 'count')
            )
             # Add overall average per denoiser
            overall_avg_per_denoiser = combined_results.groupby('DenoiserName')[['PSNR', 'SSIM', 'Time']].mean()
            overall_avg_per_denoiser['Num_Images'] = combined_results.groupby('DenoiserName')['FileName'].count()
             # Add multi-index level for alignment if needed or just print separately
            print("\nOverall Averages per Denoiser Configuration:")
            print(overall_avg_per_denoiser)
            print("\nDetailed Averages per Denoiser and Noise Type:")
            print(overall_summary)

        else:
             print("Combined results DataFrame structure doesn't allow detailed grouping. Printing overall mean:")
             print(combined_results[['PSNR', 'SSIM', 'Time']].mean())

    else:
        print("\nNo results were successfully generated across any parameter sets.")
        print("Please check the input directories, image files, and potential errors reported above.")

    print("\n--- Script Execution Finished ---")