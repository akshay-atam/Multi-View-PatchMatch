import numpy as np
import matplotlib.pyplot as plt

# change the file structure as in depthmaps
file_paths = ["depthmaps/herz-jesu/SSD/depth_0.npy",
              "depthmaps/herz-jesu/SSD/depth_1.npy",
              "depthmaps/herz-jesu/SSD/depth_2.npy",
              "depthmaps/herz-jesu/SSD/depth_3.npy",
              "depthmaps/herz-jesu/SSD/depth_4.npy",
              "depthmaps/herz-jesu/SSD/depth_5.npy",
              "depthmaps/herz-jesu/SSD/depth_6.npy",
              "depthmaps/herz-jesu/SSD/depth_7.npy",
              "depthmaps/herz-jesu/SSD/depth_8.npy",
              "depthmaps/herz-jesu/SSD/depth_9.npy",]

num_files = len(file_paths)

# Jet
fig, axes = plt.subplots(2, num_files // 2, figsize=(12, 8))

for i, file_path in enumerate(file_paths):
    data = np.load(file_path)
    row = i // (num_files // 2)
    col = i % (num_files // 2)
    
    axes[row, col].imshow(data, cmap='jet')
    axes[row, col].set_title(f"File {i+1} (Jet Colormap)")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# Gray
fig, axes = plt.subplots(2, num_files // 2, figsize=(12, 8))

for i, file_path in enumerate(file_paths):
    data = np.load(file_path)
    row = i // (num_files // 2)
    col = i % (num_files // 2)
    
    axes[row, col].imshow(data, cmap='gray')
    axes[row, col].set_title(f"File {i+1} (Gray Colormap)")
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()