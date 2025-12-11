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



def get_centroid_activation(array, threshold=0.02):
    print(array.shape)
    h, w = array.shape
    am = torch.argmax(array)
    am_h, am_w = am // w, am % w

    # Candidate pixels close to the max
    candidate_locations = list(((array[am_h, am_w] - array) <= threshold).nonzero())

    contiguous_w_max = [torch.tensor([am_h, am_w])]
    added_new = True

    while added_new:
        added_new = False
        to_move = []
        for cl_ind, cl in enumerate(candidate_locations):
            for contig in contiguous_w_max:
                if abs(cl[0] - contig[0]) <= 1 and abs(cl[1] - contig[1]) <= 1:
                    if not (cl[0] == contig[0] and cl[1] == contig[1]):
                        to_move.append(cl_ind)
                        added_new = True
                        break

        for index in sorted(to_move, reverse=True):
            contiguous_w_max.append(candidate_locations[index])
            del candidate_locations[index]

    # Remove duplicated max
    if len(contiguous_w_max) > 1:
        del contiguous_w_max[0]

    # Means
    h_mean, w_mean = 0.0, 0.0
    for cm in contiguous_w_max:
        h_mean += cm[0].item()
        w_mean += cm[1].item()
    h_mean /= len(contiguous_w_max)
    w_mean /= len(contiguous_w_max)

    print(h_mean, w_mean, contiguous_w_max)
    return (h_mean, w_mean)



def resize_and_normalize(img, use_crop=False):
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)

    if np.sum(img) == 0:
        img = torch.tensor(img).expand(1, 3, *img.shape).float()
        return F.interpolate(img, size=target_size, mode='bilinear')[0]

    dummy_batch_dim = False
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
        dummy_batch_dim = True

    with torch.no_grad():
        if use_crop:
            img = crop(torch.tensor((img - img_mean) / img_std))
        else:
            img = torch.tensor((img - img_mean) / img_std)

        img = img.expand(1, 3, *img.shape).float()
        img_resized = F.interpolate(img, size=target_size, mode='bilinear')

    return img_resized[0] if dummy_batch_dim else img_resized[0]



def main(align_images=False, use_crop=False, batch_size=1, max_workers=0,
         print_every=50, save_every=200):

    device = 0
    torch.cuda.set_device(device)

    # Load model
    model = torch.load(
        'snapshots/trained_asymmirai.pt',
        weights_only=False,
        map_location=torch.device(f'cuda:{device}')
    )
    model.eval()
    model.latent_h = 5
    model.latent_w = 5
    model.topk_for_heatmap = None
    model.topk_weights = torch.tensor([1]).cuda()
    model.use_bn = False
    model.learned_asym_mean = model.initial_asym_mean
    model.learned_asym_std = model.initial_asym_std

    # Load dataset
    val_df = pd.read_csv('data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW_test.csv')
    val_dataset = MiraiMetadatasetS3(
        val_df, resizer=resize_and_normalize,
        mode="val", align_images=align_images,
        multiple_pairs_per_exam=False
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=min(max_workers, batch_size)
    )

    # Storage
    eids_for_epoch = []
    predictions = []
    centroids_h_cc_for_epoch = []
    centroids_w_cc_for_epoch = []
    centroids_h_mlo_for_epoch = []
    centroids_w_mlo_for_epoch = []

    with torch.no_grad():
        for index, sample in enumerate(val_dataloader):
            eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample

            # Move to GPU
            l_cc_img = l_cc_img.cuda()
            r_cc_img = r_cc_img.cuda()
            l_mlo_img = l_mlo_img.cuda()
            r_mlo_img = r_mlo_img.cuda()
            label = label.cuda()

            output, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)

            # Accumulate model outputs
            eids_for_epoch.extend(eid.numpy().tolist())
            predictions.extend(output.detach().cpu().numpy().tolist())

            # Heatmaps (c = 0 CC, c = 1 MLO)
            for c in range(2):
                for i in range(l_cc_img.size(0)):
                    heatmap = other[c]['heatmap'][i]
                    cy, cx = get_centroid_activation(heatmap)

                    if c == 0:
                        centroids_h_cc_for_epoch.append(cy)
                        centroids_w_cc_for_epoch.append(cx)
                    else:
                        centroids_h_mlo_for_epoch.append(cy)
                        centroids_w_mlo_for_epoch.append(cx)

            # Progress print
            if (index + 1) % print_every == 0:
                print(f"[{index + 1}/{len(val_dataloader)}] processed")

            # Periodic CSV save
            if (index + 1) % save_every == 0:
                df = pd.DataFrame({
                    'exam_id': eids_for_epoch,
                    'prediction_neg': list(1 - np.array(predictions)),
                    'prediction_pos': predictions,
                    'y_argmin_cc': centroids_h_cc_for_epoch,
                    'x_argmin_cc': centroids_w_cc_for_epoch,
                    'y_argmin_mlo': centroids_h_mlo_for_epoch,
                    'x_argmin_mlo': centroids_w_mlo_for_epoch
                })
                df.to_csv('tmp_val_run.csv', index=False)
                print(f"Saved partial CSV at sample {index + 1}")

    # Final save
    df = pd.DataFrame({
        'exam_id': eids_for_epoch,
        'prediction_neg': list(1 - np.array(predictions)),
        'prediction_pos': predictions,
        'y_argmin_cc': centroids_h_cc_for_epoch,
        'x_argmin_cc': centroids_w_cc_for_epoch,
        'y_argmin_mlo': centroids_h_mlo_for_epoch,
        'x_argmin_mlo': centroids_w_mlo_for_epoch
    })
    df.to_csv('tmp_val_run.csv', index=False)
    print("Final CSV saved.")



if __name__ == "__main__":
    main(
        align_images=False,
        use_crop=False,
        batch_size=1,
        max_workers=0,
        print_every=1,   # print every 50 iterations
        save_every=10    # write CSV every 200 iterations
    )
