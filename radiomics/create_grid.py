import numpy as np
from PIL import Image
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# ---------------------------------------------------------------
# 1. Load mammogram
# ---------------------------------------------------------------
img_path = "data\\2ce50597d3a7711d30b9ec3a0aa25623.png"   # <-- replace with your file path
img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)

print("Image shape:", img.shape)

# ---------------------------------------------------------------
# 2. Lattice (patch) parameters
# ---------------------------------------------------------------
patch_size = 40
stride = 40   # same as patch size → non-overlapping grid

# ---------------------------------------------------------------
# 3. Extract patches & coordinates
# ---------------------------------------------------------------
patches = []
coords = []

for y in range(0, img.shape[0] - patch_size + 1, stride):
    for x in range(0, img.shape[1] - patch_size + 1, stride):
        patch = img[y:y+patch_size, x:x+patch_size]
        patches.append(patch)
        coords.append((x, y))

print(f"Total patches extracted: {len(patches)}")

# ---------------------------------------------------------------
# 4. Initialize PyRadiomics Extractor
# ---------------------------------------------------------------
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()
extractor.disableAllImageTypes()
extractor.enableImageTypeByName('Original')

# ---------------------------------------------------------------
# 5. Extract radiomic features patch-by-patch
# ---------------------------------------------------------------
feature_rows = []

for i, patch in enumerate(patches):

    # Convert patch to SimpleITK image
    patch_img = sitk.GetImageFromArray(patch)

    # Mask → full patch is ROI
    mask = np.ones_like(patch, dtype=np.uint8)
    patch_mask = sitk.GetImageFromArray(mask)

    # Extract features
    features = extractor.execute(patch_img, patch_mask)

    # Convert dict → single row
    row = {k: v for k, v in features.items()}
    row['x'], row['y'] = coords[i]  # include patch coordinates
    feature_rows.append(row)

    print(f"Processed patch {i+1}/{len(patches)}")

# ---------------------------------------------------------------
# 6. Convert to DataFrame & save
# ---------------------------------------------------------------
df = pd.DataFrame(feature_rows)
df.to_csv("radiomics_lattice_features.csv", index=False)

print("Saved radiomics features to radiomics_lattice_features.csv")
print("DataFrame shape:", df.shape)
