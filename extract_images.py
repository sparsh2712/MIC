import os
import shutil
import time 

input_dir = "/Users/sparsh/Desktop/MIC/results/cgan_train"
output_dir = os.path.expanduser("~/Desktop/slide_results/gan")
target_numbers = ["9", "145", "258", "274", "374"]

os.makedirs(output_dir, exist_ok=True)

found_count = 0

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith('.png'):
            file_path = os.path.join(root, file)
            parts = file.split('_')           
            for part in parts:
                if part in target_numbers:
                    dest_path = os.path.join(output_dir, file)
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied: {file}")
                    found_count += 1
                    break

print(f"Done! Copied {found_count} images")