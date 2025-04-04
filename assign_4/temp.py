import os
import numpy as np
from utils.io_utils import load_mat_file
import matplotlib.pyplot as plt

# Load hand data and convert to list of shapes
data = load_mat_file(os.path.join('assign4_data', 'hands2D.mat'))
img = data["shapes"]
os.makedirs('hand_shapes', exist_ok=True)

for i in range(img.shape[0]):
    coords = img[i]
    x = coords[:, 0]
    y = coords[:, 1]

    plt.figure(figsize=(4, 4))
    plt.plot(x, y, marker='o')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'hand_shapes/hand_{i}.png', dpi=150)
    plt.close()