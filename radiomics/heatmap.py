import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load mammogram image
# -------------------------------
img_path = "data\\2ce50597d3a7711d30b9ec3a0aa25623.png"   # <--- change this
img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)

# -------------------------------
# 2. Lattice parameters
# -------------------------------
patch_size = 40
stride = 40

# -------------------------------
# 3. Compute per-patch feature
# -------------------------------
feature_map = []

for y in range(0, img.shape[0] - patch_size + 1, stride):
    row_values = []
    for x in range(0, img.shape[1] - patch_size + 1, stride):
        
        patch = img[y:y+patch_size, x:x+patch_size]
        
        # Example feature: mean intensity
        feature_value = patch.mean()
        
        # You can replace above with ANY feature later
        row_values.append(feature_value)
    
    feature_map.append(row_values)

feature_map = np.array(feature_map)

# -------------------------------
# 4. Plot heatmap
# -------------------------------
plt.figure(figsize=(6,6))
plt.imshow(feature_map, cmap='inferno')
plt.colorbar(label='Mean Intensity')
# plt.title("Lattice Feature Heatmap")
plt.axis('off')

# Save heatmap
plt.savefig("heatmap_output.png", dpi=300, bbox_inches='tight')
plt.show()
