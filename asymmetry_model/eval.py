import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
ASYMMETRY_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, ASYMMETRY_MODEL)

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from asymmetry_model.mirai_metadatasets3 import MiraiMetadatasetS3
from embed_explore import resize_and_normalize, crop

def get_centroid_activation(heatmap: torch.Tensor, threshold: float = 0.02):
    """
    Finds the centroid of the contiguous region near the max activation.
    """
    hmap = heatmap.detach().cpu()
    H, W = hmap.shape
    max_val = torch.max(hmap)
    mask = (max_val - hmap) <= threshold

    coords = mask.nonzero(as_tuple=False).tolist()
    visited = set()
    components = []
    for r, c in coords:
        if (r, c) in visited:
            continue

        stack = [(r, c)]
        component = []

        while stack:
            rr, cc = stack.pop()
            if (rr, cc) in visited:
                continue

            visited.add((rr, cc))
            component.append((rr, cc))

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = rr + dr, cc + dc
                    if (nr, nc) in coords and (nr, nc) not in visited:
                        stack.append((nr, nc))

        components.append(component)
    
    # take largest connected component
    largest = max(components, key=len)
    ys, xs = zip(*largest)

    return float(np.mean(ys)), float(np.mean(xs))

def resize_and_normalize(img, use_crop=False):
    """
    Normalizes and resizes a single image.
    Output: torch.FloatTensor [3, 1664, 2048]
    """
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)

    img = torch.tensor(img, dtype=torch.float32)

    if torch.sum(img) == 0:
        img = img.unsqueeze(0).repeat(3, 1, 1)
        img = img.unsqueeze(0)
        return F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]

    img = (img - img_mean) / img_std

    if use_crop:
        img = crop(img)

    if img.ndim == 2:
        img = img.unsqueeze(0)

    img = img.repeat(3, 1, 1)
    img = img.unsqueeze(0)

    img = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)
    return img[0]





def main(align_images=False, 
         use_crop=False, 
         batch_size=1, 
         max_workers=0,
         print_every=50, 
         save_every=200):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = torch.load(
        "snapshots/trained_asymmirai.pt",
        map_location=device,
        weights_only=False,
    )
    
    model = model.eval().to(device)
    model.latent_h = 5
    model.latent_w = 5
    model.topk_for_heatmap = None
    model.topk_weights = torch.tensor([1.0], device=device)
    model.use_bn = False
    model.learned_asym_mean = model.initial_asym_mean
    model.learned_asym_std = model.initial_asym_std

      # Dataset
    # --------------------
    val_df = pd.read_csv(
        "data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_with_demographics.csv"
    )

    val_dataset = MiraiMetadatasetS3(
        val_df,
        resizer=lambda x: resize_and_normalize(x, use_crop),
        mode="val",
        align_images=align_images,
        s3_bucket="embdedpng",
        multiple_pairs_per_exam=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(max_workers, batch_size),
        pin_memory=torch.cuda.is_available(),
    )

    # Storag
    eids = []
    pred_neg = []
    pred_pos = []

    y_cc, x_cc = [], []
    y_mlo, x_mlo = [], []

     # --------------------
    # Inference
    # --------------------
    with torch.no_grad():
        for idx, sample in enumerate(val_loader):
            (
                eid,
                label,
                l_cc_img,
                l_cc_path,
                r_cc_img,
                r_cc_path,
                l_mlo_img,
                l_mlo_path,
                r_mlo_img,
                r_mlo_path,
            ) = sample

            l_cc_img = l_cc_img.to(device)
            r_cc_img = r_cc_img.to(device)
            l_mlo_img = l_mlo_img.to(device)
            r_mlo_img = r_mlo_img.to(device)

            output, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

            batch_sz = output.size(0)

            eids.extend(eid.cpu().numpy().tolist())
            pred_neg.extend(output[:, 0].cpu().numpy().tolist())
            pred_pos.extend(output[:, 1].cpu().numpy().tolist())

            for c in range(2):  # 0 = CC, 1 = MLO
                heatmaps = other[c]["heatmap"]
                for i in range(batch_sz):
                    cy, cx = get_centroid_activation(heatmaps[i])
                    if c == 0:
                        y_cc.append(cy)
                        x_cc.append(cx)
                    else:
                        y_mlo.append(cy)
                        x_mlo.append(cx)

            if (idx + 1) % print_every == 0:
                print(f"[{idx + 1}/{len(val_loader)}] processed")

            if (idx + 1) % save_every == 0:
                pd.DataFrame(
                    {
                        "exam_id": eids,
                        "prediction_neg": pred_neg,
                        "prediction_pos": pred_pos,
                        "y_argmin_cc": y_cc,
                        "x_argmin_cc": x_cc,
                        "y_argmin_mlo": y_mlo,
                        "x_argmin_mlo": x_mlo,
                    }
                ).to_csv("tmp_val_run.csv", index=False)
                print(f"Saved partial CSV at {idx + 1}")

    # --------------------
    # Final save
    # --------------------
    df = pd.DataFrame(
        {
            "exam_id": eids,
            "prediction_neg": pred_neg,
            "prediction_pos": pred_pos,
            "y_argmin_cc": y_cc,
            "x_argmin_cc": x_cc,
            "y_argmin_mlo": y_mlo,
            "x_argmin_mlo": x_mlo,
        }
    )
    df.to_csv("tmp_val_run.csv", index=False)
    print("Final CSV saved.")


if __name__ == "__main__":
    main(
        align_images=False,
        use_crop=False,
        batch_size=1,
        max_workers=0,
        print_every=1,
        save_every=10,
    )