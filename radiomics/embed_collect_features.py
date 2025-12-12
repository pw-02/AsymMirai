import os
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
from skimage.io import imsave
from radiomics import featureextractor
import pandas as pd
from numpy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

def show_overlay(image, mask):
    plt.figure(figsize=(8,8))
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.3)
    plt.show()


# ======================================================
# CUSTOM TEXTURE FEATURES
# ======================================================

def autocorrelation_decay(img, mask=None, max_radius=128):
    img = img.astype(np.float32)

    # Ensure mask matches image shape
    if mask is not None:
        if mask.shape != img.shape:
            raise ValueError(f"Mask {mask.shape} != image {img.shape}")
        img = img * mask.astype(np.float32)

    # Normalize
    img = img - np.mean(img)

    # FFT autocorrelation
    F = fft2(img)
    acf = fftshift(ifft2(np.abs(F)**2).real)
    acf = acf / (acf.max() + 1e-8)

    H, W = acf.shape
    cy, cx = H//2, W//2
    y, x = np.indices((H, W))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Radial ACF
    radial_acf = []
    for R in range(1, min(max_radius, min(H, W)//2)):
        ring = (r >= R-0.5) & (r < R+0.5)
        if np.any(ring):
            radial_acf.append(acf[ring].mean())
        else:
            radial_acf.append(np.nan)

    radial_acf = np.array(radial_acf)

    # Remove NaNs
    valid = ~np.isnan(radial_acf)
    radii = np.arange(len(radial_acf))[valid]
    values = radial_acf[valid]

    # Fit exponential decay
    logv = np.log(values + 1e-8)
    coeffs = np.polyfit(radii, logv, 1)
    decay_rate = -coeffs[0]

    return decay_rate

def lacunarity(img, mask=None, box_sizes=[2,4,8,16,32]):
    if mask is not None:
        img = img * mask

    img = img.astype(np.float32)
    results = {}

    for r in box_sizes:
        patches = []
        for i in range(0, img.shape[0]-r, r):
            for j in range(0, img.shape[1]-r, r):
                patch = img[i:i+r, j:j+r]
                patches.append(np.sum(patch))

        patches = np.array(patches)
        L = np.var(patches) / (np.mean(patches)**2 + 1e-8)
        results[f"lac_{r}"] = L

    return results


def fractal_dimension_binary(mask):
    Z = mask > 0
    p = min(Z.shape)
    sizes = 2**np.arange(int(np.log2(p)), 1, -1)

    counts = []
    for S in sizes:
        count = 0
        for i in range(0, Z.shape[0], S):
            for j in range(0, Z.shape[1], S):
                if np.any(Z[i:i+S, j:j+S]):
                    count += 1
        counts.append(count)

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


# ======================================================
# IMAGE LOADING
# ======================================================

def dicom_to_2d_image(dicom_path):
    sitk_img = sitk.ReadImage(dicom_path)
    arr3d = sitk.GetArrayFromImage(sitk_img)
    arr2d = arr3d[0].astype(np.float32)

    norm = (arr2d - arr2d.min()) / (arr2d.max() - arr2d.min() + 1e-8)
    return arr2d, norm


# ======================================================
# MASKING
# ======================================================

import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

def remove_pectoral_muscle(img, mask):
    """
    Automatically detect and remove the pectoral muscle in MLO mammograms.
    Handles both left and right MLO views.
    """

    H, W = img.shape

    # --- 1. Determine whether muscle is left or right ---
    left_brightness  = img[0:H//4, 0:W//4].mean()
    right_brightness = img[0:H//4, W//2:W].mean()

    if left_brightness > right_brightness:
        roi = img[:H//3, :W//2]       # muscle left
        x_offset = 0
    else:
        roi = img[:H//3, W//2:W]      # muscle right
        x_offset = W//2

    # --- 2. Edge detection ---
    edges = canny(roi, sigma=2)

    # --- 3. Hough transform ---
    hspace, angles, dists = hough_line(edges)
    h_peaks, angle_peaks, dist_peaks = hough_line_peaks(hspace, angles, dists)

    # If no line found → skip muscle removal
    if len(angle_peaks) == 0:
        return mask

    angle = angle_peaks[0]
    dist  = dist_peaks[0]

    # --- 4. Convert Hough line into full-image coordinates ---
    y0 = 0
    x0 = (dist - y0 * np.sin(angle)) / np.cos(angle) + x_offset

    y1 = H//3
    x1 = (dist - y1 * np.sin(angle)) / np.cos(angle) + x_offset

    # --- 5. Create region ABOVE the detected pectoral line ---
    rr, cc = np.indices(mask.shape)

    # Line equation
    slope = (x1 - x0) / (y1 - y0 + 1e-8)
    intercept = x0

    muscle_region = cc < (slope * (rr - y0) + intercept)

    # Restrict to upper part only (pectoral area)
    muscle_region = muscle_region & (rr < H//3)

    # --- 6. Remove muscle from mask ---
    cleaned_mask = mask.copy()
    cleaned_mask[muscle_region] = 0

    return cleaned_mask.astype(np.uint8)


def create_mask(img_norm, min_size=5000):
    thresh = threshold_otsu(img_norm)
    mask = img_norm > thresh

    label_img = label(mask)
    regions = regionprops(label_img)
    largest = regions[np.argmax([r.area for r in regions])].label
    mask = (label_img == largest)

    mask = remove_small_objects(mask, min_size)
    mask = binary_closing(mask, disk(10))
    mask = binary_opening(mask, disk(5))
    mask = convex_hull_image(mask)

    print(np.unique(mask))

    return mask.astype(np.uint8)


def save_inputs(image_2d, mask_2d, base_name="case"):
    img_sitk = sitk.GetImageFromArray(image_2d)
    image_path = f"{base_name}_image.nii.gz"
    sitk.WriteImage(img_sitk, image_path)

    mask_path = f"{base_name}_mask.png"
    imsave(mask_path, mask_2d.astype(np.uint8))

    return image_path, mask_path


# ======================================================
# RADIOMICS
# ======================================================

def extract_radiomics_features(image_path, mask_path):
    extractor = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25,
        normalize=True,
        resampledPixelSpacing=None,
        interpolator="sitkBSpline"
    )
    
    result = extractor.execute(image_path, mask_path)
    return {k: v for k, v in result.items() if "diagnostics" not in k}


# ======================================================
# MAIN PIPELINE
# ======================================================

def process_mammogram(dicom_path, base_name="example"):
    print(f"Processing {dicom_path}")

    image, norm = dicom_to_2d_image(dicom_path)
    mask = create_mask(norm)

    # Detect and remove pectoral muscle (MLO only)
    # mask = remove_pectoral_muscle(norm, mask)

    show_overlay(image, mask)

    img_file, mask_file = save_inputs(image, mask, base_name)

    radiomics = extract_radiomics_features(img_file, mask_file)

    # CUSTOM FEATURES
    acd = autocorrelation_decay(image, mask)
    fd = fractal_dimension_binary(mask)
    lac = lacunarity(image, mask)

    custom = {
        "path": dicom_path,
        "acf_decay": acd,
        "fractal_dimension_mask": fd,
    }
    custom.update(lac)

    # Merge everything
    all_features = { **custom, **radiomics }

    #cleanup temp files
    os.remove(img_file) if os.path.exists(img_file) else None
    os.remove(mask_file) if os.path.exists(mask_file) else None

    return all_features


def process_list_of_files(dicom_list, output_csv="radiomics_features.csv"):
    all_data = []
    counter = 0
    for dicom_rel_path in dicom_list:

        dicom_path = '/media/pwatters/WD_BLACK/MammoDataset/EMBED/' + dicom_rel_path

        features = process_mammogram(dicom_path, base_name=os.path.splitext(os.path.basename(dicom_path))[0])
        all_data.append(features)

        #save after evey 500 files
        if len(all_data) % 500 == 0:
            df = pd.DataFrame(all_data)
            df.to_csv(output_csv, index=False)
            print(f"Saved intermediate {output_csv} after {len(all_data)} files")
        
        #save file path to files processed txt file to avoid reprocessing
        with open("radiomics/files_processed.txt", 'a') as f:
            f.write(dicom_rel_path + '\n')
        counter += 1
        print(f"Processed {counter} files.")

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")


# ======================================================
if __name__ == "__main__":
        input_file = "radiomics/files_to_process.txt"
        with open(input_file, 'r') as f:
            dicom_files = [line.strip() for line in f.readlines()]
        
        process_list_of_files(dicom_files)