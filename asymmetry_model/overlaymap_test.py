import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG (EDIT THESE)
# -----------------------------
IMAGE_PATH = "C:\\Users\\pw\\Desktop\\17c3d032d611eb3d1baf3434d62483a1.png"     # path to your mammogram image
HEATMAP_PATH = None                 # set to .npy file if you have one
USE_FAKE_HEATMAP = True             # set False if loading real heatmap
HEATMAP_SHAPE = (32, 32)             # only used if fake heatmap


# -----------------------------
# LOAD IMAGE
# -----------------------------
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

img = img.astype(np.float32)
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
img_rgb = np.repeat(img[:, :, None], 3, axis=2)


# -----------------------------
# LOAD OR CREATE HEATMAP
# -----------------------------
if USE_FAKE_HEATMAP:
    heatmap = np.random.rand(*HEATMAP_SHAPE).astype(np.float32)
else:
    heatmap = np.load(HEATMAP_PATH).astype(np.float32)

heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)


# -----------------------------
# UPSAMPLE HEATMAP TO IMAGE SIZE
# -----------------------------
heatmap_up = cv2.resize(
    heatmap,
    (img.shape[1], img.shape[0]),
    interpolation=cv2.INTER_LINEAR   # use INTER_NEAREST for blocky MIRAI-style
)


# -----------------------------
# APPLY COLORMAP
# -----------------------------
heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap_up),
    cv2.COLORMAP_JET
)
heatmap_color = heatmap_color[..., ::-1] / 255.0  # BGR → RGB


# -----------------------------
# OVERLAY
# -----------------------------
overlay = 0.6 * img_rgb + 0.4 * heatmap_color
overlay = np.clip(overlay, 0, 1)


# -----------------------------
# DISPLAY
# -----------------------------
plt.figure(figsize=(6, 8))
plt.imshow(overlay)
plt.axis("off")
plt.title("Mammogram Heatmap Overlay")
plt.show()
