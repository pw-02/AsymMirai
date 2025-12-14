import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, binary_opening, disk
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

from numpy.fft import fft2, ifft2, fftshift
from radiomics import featureextractor
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import numpy as np
# -----------------------------
# CONFIG
# -----------------------------

TEST_MODE = False      # set True for quick debugging on few images
DEBUG_PLOT = False    # show intermediate images for debugging

if TEST_MODE:
    BASE_DIR = ""
    INPUT_FILE = "radiomics/files_to_process_test.txt"
    PROCESSED_FILE = "radiomics/files_to_process_test_processed.txt"
    OUTPUT_CSV = "radiomics/files_to_process_test_features.csv"
else:
    BASE_DIR = "/media/pwatters/WD_BLACK/MammoDataset/EMBED/" 
    # INPUT_FILE = "radiomics/files_to_process_mlo.txt"
    # PROCESSED_FILE = "radiomics/files_processed_mlo.txt"
    # OUTPUT_CSV = "radiomics/radiomics_features_mlo.csv"
    INPUT_FILE = "radiomics/density_4_files.txt"
    PROCESSED_FILE = "radiomics/density_4_files_processed.txt"
    OUTPUT_CSV = "radiomics/radiomics_features_density_4.csv"
    



          # set True for single-image debugging
N_WORKERS = max(1, cpu_count() - 1) if not TEST_MODE else 1
# N_WORKERS = 1
FLUSH_EVERY = 200 if not TEST_MODE else 1

# Radiomics speed control:
# If you enable only a few feature classes, runtime drops a lot.
ENABLE_LIMITED_RADIOMICS = True
ENABLED_CLASSES = ["firstorder", "glcm", "glrlm"]  # adjust as needed


# -----------------------------
# RADIOMICS EXTRACTOR (init once)
# -----------------------------
EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(
    # binWidth=25,
    binCount=128,
    normalize=True,
    resampledPixelSpacing=None,
    interpolator="sitkBSpline",
)

if ENABLE_LIMITED_RADIOMICS:
    EXTRACTOR.disableAllFeatures()
    for cls in ENABLED_CLASSES:
        EXTRACTOR.enableFeatureClassByName(cls)


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

# -----------------------------
# CUSTOM FEATURES
# -----------------------------
import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def autocorrelation_decay(
    img,
    mask,
    spacing=(1.0, 1.0),
    max_radius=None,
):
    img = img.astype(np.float32)
    mask = mask.astype(bool)

    if mask.shape != img.shape:
        raise ValueError("Mask and image shape mismatch")

    if mask.sum() < 20:
        return np.nan

    # Normalize INSIDE mask
    vals = img[mask]
    img = img.copy()
    img[mask] = (vals - vals.mean()) / (vals.std() + 1e-8)
    img[~mask] = 0.0

    # Autocorrelation
    F = fft2(img)
    acf = fftshift(ifft2(np.abs(F) ** 2).real)
    acf /= acf.max() + 1e-8

    H, W = acf.shape
    cy, cx = H // 2, W // 2

    y, x = np.indices((H, W))
    dy = (y - cy) * spacing[0]
    dx = (x - cx) * spacing[1]
    r = np.sqrt(dx**2 + dy**2)

    maxR = max_radius or min(H, W) * min(spacing) / 4

    # Radial bins
    dr = min(spacing)
    bins = np.arange(0, maxR, dr)
    radial = np.zeros(len(bins) - 1)

    for i in range(len(bins) - 1):
        ring = (r >= bins[i]) & (r < bins[i + 1])
        if np.any(ring):
            radial[i] = acf[ring].mean()
        else:
            radial[i] = np.nan

    valid = (~np.isnan(radial)) & (radial > 0)
    if valid.sum() < 5:
        return np.nan

    radii = bins[:-1][valid]
    logv = np.log(radial[valid])

    slope = np.polyfit(radii, logv, 1)[0]
    return float(-slope)



def lacunarity(img, mask=None, box_sizes=(2, 4, 8, 16, 32)):
    # Lightweight version, still has loops but ok compared to radiomics
    if mask is not None:
        img = img * mask

    img = img.astype(np.float32)
    results = {}

    H, W = img.shape
    for r in box_sizes:
        if r <= 0 or r > min(H, W):
            continue

        # crop so reshape works cleanly
        h = (H // r) * r
        w = (W // r) * r
        if h == 0 or w == 0:
            continue

        cropped = img[:h, :w]
        # reshape into blocks and sum
        blocks = cropped.reshape(h // r, r, w // r, r).sum(axis=(1, 3))
        patches = blocks.flatten()

        mu = patches.mean()
        var = patches.var()
        L = float(var / (mu * mu + 1e-8))
        results[f"lac_{r}"] = L

    return results


def fractal_dimension_binary(mask):
    Z = mask > 0
    p = min(Z.shape)
    if p < 4:
        return float("nan")

    sizes = 2 ** np.arange(int(np.log2(p)), 1, -1)
    counts = []

    for S in sizes:
        # count non-empty boxes of size S
        count = 0
        for i in range(0, Z.shape[0], S):
            for j in range(0, Z.shape[1], S):
                if np.any(Z[i:i + S, j:j + S]):
                    count += 1
        counts.append(count)

    sizes = np.array(sizes, dtype=np.float32)
    counts = np.array(counts, dtype=np.float32)

    coeffs = np.polyfit(np.log(sizes + 1e-8), np.log(counts + 1e-8), 1)
    return float(-coeffs[0])




def compute_lbp(image, mask, radius=1, n_points=8, method="uniform"):
    lbp = local_binary_pattern(image, n_points, radius, method)
    lbp_roi = lbp[mask > 0]

    return {
        "LBP_mean": np.mean(lbp_roi),
        "LBP_std": np.std(lbp_roi),
        "LBP_entropy": -np.sum(
            np.histogram(lbp_roi, bins=32, density=True)[0]
            * np.log2(np.histogram(lbp_roi, bins=32, density=True)[0] + 1e-10)
        ),
    }


# -----------------------------
# IMAGE LOADING / MASKING
# -----------------------------
# def dicom_to_2d_image(dicom_path):
#     # SimpleITK ReadImage is fine; pydicom often slower unless you optimize.
#     sitk_img = sitk.ReadImage(dicom_path)
#     arr3d = sitk.GetArrayFromImage(sitk_img)
#     arr2d = arr3d[0].astype(np.float32)

#     mn, mx = float(arr2d.min()), float(arr2d.max())
#     norm = (arr2d - mn) / (mx - mn + 1e-8)
#     return arr2d, norm

def dicom_to_2d_image(dicom_path):
    sitk_img = sitk.ReadImage(dicom_path)
    arr3d = sitk.GetArrayFromImage(sitk_img)
    arr2d = arr3d[0].astype(np.float32)

    # Clip to a fixed, meaningful range
    arr2d = np.clip(arr2d, np.percentile(arr2d, 1), np.percentile(arr2d, 99))

    # Scale using a fixed denominator
    norm = (arr2d - arr2d.min()) / (arr2d.max() - arr2d.min() + 1e-8)

    return arr2d, norm


def create_mask(img_norm, min_size=5000):
    thresh = threshold_otsu(img_norm)
    mask = img_norm > thresh

    label_img = label(mask)
    regions = regionprops(label_img)
    if not regions:
        return np.zeros_like(img_norm, dtype=np.uint8)

    largest = regions[np.argmax([r.area for r in regions])].label
    mask = (label_img == largest)

    mask = remove_small_objects(mask, min_size)
    mask = binary_closing(mask, disk(10))
    mask = binary_opening(mask, disk(5))
    mask = convex_hull_image(mask)

    return mask.astype(np.uint8)


# -----------------------------
# RADIOMICS (in-memory)
# -----------------------------
def extract_radiomics_features(image_np, mask_np, spacing=(1.0, 1.0)):
    image_sitk = sitk.GetImageFromArray(image_np.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray((mask_np > 0).astype(np.uint8))

    image_sitk.SetSpacing(spacing)
    mask_sitk.SetSpacing(spacing)

    result = EXTRACTOR.execute(image_sitk, mask_sitk)
    
    return {
        k: v for k, v in result.items()
        if not k.startswith("diagnostics")
    }


# -----------------------------
# SINGLE FILE PROCESSOR
# -----------------------------
def process_one(rel_path):
    """
    Worker-safe function: takes a relative DICOM path, returns dict of features.
    If error occurs, returns dict with 'error' field.
    """
    dicom_path = os.path.join(BASE_DIR, rel_path)

    try:
        image, norm = dicom_to_2d_image(dicom_path)


        mask = create_mask(norm)

        # If mask empty, skip
        if mask.sum() == 0:
            return {"path": rel_path, "error": "empty_mask"}
        if DEBUG_PLOT:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            # --- 1. Image ---
            axes[0].imshow(norm, cmap="gray")
            axes[0].set_title("Image (normalized)")
            axes[0].axis("off")

            # --- 2. Mask ---
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Mask")
            axes[1].axis("off")

            # --- 3. Overlay (BEST diagnostic view) ---
            axes[2].imshow(norm, cmap="gray")
            axes[2].contour(mask, levels=[0.5], colors="red", linewidths=1)

            # axes[2].imshow(mask, cmap="Reds", alpha=0.35)
            axes[2].set_title("Overlay (image + mask)")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()
        
        #remove_pectoral_muscle(image, create_mask(norm))

        # if DEBUG_PLOT:
        #     plt.title("Mask after muscle removal")
        #     plt.imshow(mask, cmap="gray")
        #     plt.axis("off")
        #     plt.show()
        # 
        #  # radiomics = extract_radiomics_features(image, mask)
        features_list = extract_radiomics_features(norm, mask)

        if TEST_MODE:
            # df = pd.DataFrame(features_list)
            # print(df.shape)
            # print(df.isna().sum().sort_values(ascending=False).head())

            print(f"Radiomics features for {rel_path}:")
            for k, v in features_list.items():
                print(f"  {k}: {v}")

        # custom
        acd = autocorrelation_decay(norm, mask)
        fd = fractal_dimension_binary(mask)
        lac = lacunarity(norm, mask)

        out = {
            "path": rel_path,
            "acf_decay": acd,
            "fractal_dimension_mask": fd,
        }
        out.update(lac)
        out.update(features_list)

        #cleanup NaNs/Infs

        return out

    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return {"path": rel_path, "error": f"{type(e).__name__}: {e}"}


# -----------------------------
# I/O HELPERS
# -----------------------------
def load_file_list(input_file, processed_file):
    with open(input_file, "r") as f:
        all_files = [line.strip() for line in f if line.strip()]

    processed = set()
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed = set(line.strip() for line in f if line.strip())

    todo = [p for p in all_files if p not in processed]
    return todo


def append_processed(processed_file, rel_paths):
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    with open(processed_file, "a") as f:
        for p in rel_paths:
            f.write(p + "\n")


def flush_csv(rows, output_csv):
    df = pd.DataFrame(rows)
    # append mode if exists
    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(output_csv, index=False)


# -----------------------------
# MAIN
# -----------------------------
def main():
    todo = load_file_list(INPUT_FILE, PROCESSED_FILE)
    print(f"Files to process (excluding already processed): {len(todo)}")
    if len(todo) == 0:
        print("Nothing to do.")
        return

    results_buffer = []
    processed_buffer = []

    # multiprocessing
    with Pool(processes=N_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(process_one, todo), total=len(todo)):
            if res is None:
                print("Warning: got None result from worker.")
            
            results_buffer.append(res)
            processed_buffer.append(res["path"])

            if len(results_buffer) >= FLUSH_EVERY:
                flush_csv(results_buffer, OUTPUT_CSV)
                append_processed(PROCESSED_FILE, processed_buffer)
                results_buffer.clear()
                processed_buffer.clear()

    # final flush
    if results_buffer:
        flush_csv(results_buffer, OUTPUT_CSV)
        append_processed(PROCESSED_FILE, processed_buffer)

    print(f"✔ Done. Output: {OUTPUT_CSV}")
    print(f"✔ Processed list updated: {PROCESSED_FILE}")

    # Optional: quick error summary
    try:
        df = pd.read_csv(OUTPUT_CSV)
        if "error" in df.columns:
            n_err = df["error"].notna().sum()
            print(f"Errors recorded: {n_err}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
