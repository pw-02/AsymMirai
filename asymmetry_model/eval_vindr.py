#!/usr/bin/env python3
"""
AsymMirai + Mirai Full Pipeline
Cleaned & structured version of original Jupyter notebook.

Includes:
    - data loading & filtering
    - AsymMirai inference
    - Mirai inference (via subprocess)
    - merging predictions
    - multi-year AUC computation
    - centroid & shift analysis
    - risk change analysis
    - bootstrap PR AUC
    - demographic analysis
    - plotting utilities

NOTE:
    All file paths remain placeholders exactly as in the original notebook.
"""

# ======================
# Standard Library
# ======================
import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import copy
# ======================
# Third-Party Libraries
# ======================
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sklearn
import sklearn.metrics

import tqdm
from tqdm import tqdm as tq
import pyroc

import sys, os

# Add project root to Python path -> gives access to onconet/ and asymmetry_model/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# Add asymmetry_model folder so local imports work
ASYMMETRY_MODEL = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, ASYMMETRY_MODEL)

import pandas as pd
import torch
from torch.utils.data import DataLoader

# Now imports work correctly:
from asymmetry_model.mirai_metadataset import MiraiMetadataset
from asymmetry_model.embed_explore import resize_and_normalize

print("Working directory:", os.getcwd())
# ============================================================
# Section 2 — Utility Functions
# ============================================================

def safe_apply(fun, array):
    """
    Safely apply a function to an array.
    Returns NaN if array is empty.
    """
    if len(array) == 0:
        return np.nan
    try:
        val = fun(array)
        return val
    except Exception:
        return np.nan


def hex_to_tuple(color, alpha=0.2):
    """
    Convert a matplotlib hex color to RGBA tuple with transparency.
    """
    color = color[1:]
    return tuple([int(color[i:i+2], 16) / 255 for i in (0, 2, 4)] + [alpha])


def str_to_arr(arr_str):
    """
    Convert serialized array from CSV back into a numpy array.

    Handles cases where strings have odd whitespace or brackets.
    """
    arr_str = re.sub(r"\s+", ",", arr_str)
    arr_str = arr_str.replace("[,", "[")
    try:
        return np.array(eval(arr_str))
    except Exception:
        return None


# ============================================================
# Section 3 — Data Loading & Filtering
# ============================================================

def get_incomplete_exams(metadata_frame: pd.DataFrame) -> List[int]:
    """
    Identify exams missing required CC/MLO views for left or right breast.
    """
    incomplete_exams = []

    for eid in tq(metadata_frame['exam_id'].unique(),
                  total=metadata_frame['exam_id'].nunique(),
                  desc="Checking incomplete exams"):

        cur_exam = metadata_frame[metadata_frame['exam_id'] == eid]

        patient_exam = {
            "MLO": {"L": None, "R": None},
            "CC": {"L": None, "R": None}
        }

        for view in patient_exam.keys():

            def indices_for_side_view(side):
                return np.logical_and(cur_exam['view'] == view,
                                      cur_exam['laterality'] == side)

            # Missing LEFT view entirely?
            if len(cur_exam[indices_for_side_view('L')]['file_path']) == 0:
                incomplete_exams.append(eid)
                continue  # skip right check

            # Missing RIGHT side?
            for laterality in ['L', 'R']:
                if len(cur_exam[indices_for_side_view(laterality)]['file_path']) == 0:
                    incomplete_exams.append(eid)
                    continue

    return list(set(incomplete_exams))


def load_and_prepare_dataset():
    """
    Load dataset CSV and apply filtering logic exactly as in original notebook.
    Paths remain placeholders — modify by hand.
    """
    # Load raw dataset (placeholder path)
    test_df = pd.read_csv(r'C:\Users\pw\projects\AsymMirai\data\breast-level_annotations.csv')
    root_png_dir = r'D:\archive\images_png'  # --- IGNORE ---
    print("Columns:", test_df.columns)

    # Filter to 2D images
    filtered = test_df[test_df['split'] == 'test'].copy()

    # Standardize column names consistent w/ notebook logic
    filtered['exam_id'] = filtered['study_id']
    filtered['patient_id'] = filtered['series_id']
    filtered['laterality'] = filtered['laterality']
    filtered['view'] = filtered['view_position']
    # filtered['file_path'] = filtered['png_path']
    
    # ---- 3. Add file_path column in one vectorized call (no loop needed) ----
    filtered["file_path"] = (
        filtered["exam_id"].apply(lambda eid: os.path.join(root_png_dir, str(eid)))
        + os.sep
        + filtered["image_id"].astype(str) 
        + ".png"
    )

    

    # Remove incomplete exams
    incomplete = get_incomplete_exams(filtered)
    filtered = filtered[~filtered['exam_id'].isin(incomplete)]

    print("Unique patients after incomplete exam removal:",
          filtered['patient_id'].nunique())
    print("Unique exams after removal:", filtered['exam_id'].nunique())

    # Remove diagnostic exams
    filtered_input_df = filtered[filtered['desc'].str.contains('screen', case=False)] if 'desc' in filtered.columns else filtered
    print("After screening filter — patients:",
          filtered_input_df['patient_id'].nunique())
    print("Exams:", filtered_input_df['exam_id'].nunique())

    # # Load Mirai form data (placeholder path)
    # mirai_form_df = pd.read_csv('../2_10_mirai_form_extended_cohorts_1-2_with_matches.csv')
    # # Merge via original notebook logic
    # filtered = filtered_input_df.merge(mirai_form_df, on='file_path', how='inner')

    # print("After merging w/ Mirai form:",
    #       filtered['exam_id'].nunique(), "exams")

    return filtered


# ============================================================
# Section 4 — AsymMirai Utilities + Inference
# ============================================================

def resize_and_normalize(img, use_crop=False):
    """
    Resize + normalize mammography image exactly like the notebook version.
    Matches expected behavior for Mirai + AsymMirai datasets.

    NOTE: crop() is imported from embed_explore if available.
    """
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)
    dummy_batch_dim = False

    # If blank image, special handling
    if np.sum(img) == 0:
        img = torch.tensor(img).expand(1, 3, *img.shape).float()
        return F.interpolate(img, size=target_size, mode='bilinear')[0]

    # Ensure batch dimension
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
        dummy_batch_dim = True

    with torch.no_grad():
        norm = (img - img_mean) / img_std
        norm = torch.tensor(norm)

        if use_crop:
            norm = crop(norm)

        # Expand to 3-channel image
        norm = norm.expand(1, 3, *norm.shape).float()

        # Interpolate to MIRAI target size
        img_resized = F.interpolate(norm, size=target_size, mode='bilinear')

    return img_resized[0] if dummy_batch_dim else img_resized[0]


def get_centroid_activation(array, threshold=0.02):
    """
    Identify centroid of contiguous activations near the max response
    in a 2D feature map.
    """
    h, w = array.shape
    max_idx = torch.argmax(array)
    max_h, max_w = max_idx // w, max_idx % w

    # candidate pixels within threshold of the max
    candidate_locations = [
        idx for idx in (array[max_h, max_w] - array <= threshold).nonzero()
    ]

    contiguous = [torch.tensor([max_h, max_w])]
    added_new = True

    while added_new:
        added_new = False
        to_add = []

        for i, loc in enumerate(candidate_locations):
            for cont in contiguous:
                if abs(loc[0] - cont[0]) <= 1 and abs(loc[1] - cont[1]) <= 1:
                    if not (loc[0] == cont[0] and loc[1] == cont[1]):
                        if i not in to_add:
                            to_add.append(i)
                        added_new = True

        # move found locations into contiguous list
        for index in sorted(to_add, reverse=True):
            contiguous.append(candidate_locations[index])
            del candidate_locations[index]

    # first item is the max, double-count fix
    if len(contiguous) > 1:
        contiguous.pop(0)

    # compute centroid
    h_mean = np.mean([c[0].item() for c in contiguous]) if contiguous else max_h.item()
    w_mean = np.mean([c[1].item() for c in contiguous]) if contiguous else max_w.item()

    return (h_mean, w_mean)


def run_validation(model, val_df):
    """
    Run AsymMirai model on all missing exams and compute:
        - prediction_pos
        - centroid coordinates for CC and MLO

    Writes intermediate results to tmp_val_run.csv.
    """
    torch.cuda.set_device(0)
    model.eval()

    # Fix model latent dims as in notebook
    model.latent_h = 5
    model.latent_w = 5
    model.topk_for_heatmap = None
    model.topk_weights = torch.tensor([1]).cuda()
    model.use_bn = False
    model.learned_asym_mean = model.initial_asym_mean
    model.learned_asym_std = model.initial_asym_std

    from mirai_metadataset import MiraiMetadataset
    val_dataset = MiraiMetadataset(
        val_df,
        resizer=resize_and_normalize,
        mode='val',
        align_images=False,
        multiple_pairs_per_exam=False
    )

    batch_size = 1
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(0, batch_size)
    )

    eids = []
    preds = []
    cc_y, cc_x = [], []
    mlo_y, mlo_x = [], []

    with torch.no_grad():
        for idx, sample in enumerate(val_loader):

            (eid, label, l_cc_img, l_cc_path,
             r_cc_img, r_cc_path,
             l_mlo_img, l_mlo_path,
             r_mlo_img, r_mlo_path) = sample

            l_cc_img, r_cc_img = l_cc_img.cuda(), r_cc_img.cuda()
            l_mlo_img, r_mlo_img = l_mlo_img.cuda(), r_mlo_img.cuda()
            label = label.cuda()

            output, other = model(l_cc_img, r_cc_img,
                                  l_mlo_img, r_mlo_img)

            # keep predictions
            eids += list(eid.numpy())
            preds += list(output.detach().cpu().numpy())

            # centroid extraction for CC + MLO
            for c in range(2):
                for i in range(batch_size):
                    heatmap = other[c]['heatmap'][i]
                    cy, cx = get_centroid_activation(heatmap)
                    if c == 0:
                        cc_y.append(cy)
                        cc_x.append(cx)
                    else:
                        mlo_y.append(cy)
                        mlo_x.append(cx)

            # save running CSV
            df = pd.DataFrame({
                'exam_id': eids,
                'prediction_neg': list(1 - np.array(preds)),
                'prediction_pos': preds,
                'y_argmin_cc': cc_y,
                'x_argmin_cc': cc_x,
                'y_argmin_mlo': mlo_y,
                'x_argmin_mlo': mlo_x
            })

            df.to_csv('tmp_val_run.csv', index=False)
            print(f"Processed batch {idx}")

    return df

# ============================================================
# Section 5 — Mirai Inference (via subprocess)
# ============================================================

def prepare_mirai_input_csv(val_df: pd.DataFrame,
                            output_csv: str = "./tmp_val_input_for_mirai_2.csv") -> str:
    """
    Create the metadata CSV that MIRAI expects.
    Follows notebook's exact column ordering and filtering.
    """
    df = val_df.copy()
    df['split_group'] = 'test'

    # MIRAI requires explicit follow-up time for negative cases
    df['years_to_last_followup'] = df.get('years_to_last_followup', pd.Series([100] * len(df)))

    cols = [
        'exam_id', 'patient_id', 'laterality', 'view',
        'file_path', 'years_to_cancer', 'years_to_last_followup',
        'split_group'
    ]

    df[cols].to_csv(output_csv, index=False)
    print(f"[Mirai] Prepared input CSV at: {output_csv}")
    return output_csv


def run_mirai_inference(input_csv: str,
                        output_csv: str = "./tmp_val_predictions_for_mirai.csv",
                        gpu: int = 0):
    """
    Execute MIRAI via subprocess to match the notebook's `%run ./scripts/main.py ...`
    """
    cmd = [
        "python", "./scripts/main.py",
        "--model_name", "mirai_full",
        "--img_encoder_snapshot", "./snapshots/mgh_mammo_MIRAI_Base_May20_2019.p",
        "--transformer_snapshot", "./snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p",
        "--callibrator_snapshot", "./snapshots/callibrators/MIRAI_FULL_PRED_RF.callibrator.p",
        "--batch_size", "2",
        "--dataset", "csv_mammo_risk_all_full_future",
        "--img_mean", "7699.5",
        "--img_size", "2294", "1914",
        "--img_std", "11765.06",
        "--metadata_path", input_csv,
        "--test",
        "--prediction_save_path", output_csv,
        "--results_path", output_csv,
        "--cuda",
        "--num_gpus", "1",
        "--test"
    ]

    print("[Mirai] Running MIRAI model...")
    subprocess.run(cmd, check=True)
    print("[Mirai] Finished MIRAI inference.")
    return output_csv


def load_mirai_predictions(pred_csv: str,
                           filtered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load MIRAI predictions and attach exam_id based on file_path lookups.
    """
    mirai_preds = pd.read_csv(pred_csv, header=None)

    # create columns for 1–5 year risks
    for i in range(5):
        mirai_preds[f'year_{i+1}_risk'] = mirai_preds[4 + i]

    # extract exam_id by matching file_path
    def get_exam_id(row):
        file_path = row[0]
        result = filtered_df[filtered_df['file_path'] == file_path]['exam_id']
        return result.values[0] if len(result) > 0 else None

    mirai_preds['exam_id'] = mirai_preds.apply(get_exam_id, axis=1)
    return mirai_preds


# ============================================================
# Section 6 — Merging AsymMirai + Mirai + Metadata
# ============================================================

def correct_y_argmin_cc(row):
    """
    Fix CC argmin: if stored as serialized array, extract real value.
    """
    if isinstance(row['y_argmin_cc'], float):
        return int(row['y_argmin_cc'])

    arr = str_to_arr(row['y_argmin_cc'])
    if arr is None:
        return np.nan
    return arr[int(row['x_argmin_cc'])]


def correct_y_argmin_mlo(row):
    """
    Fix MLO argmin: if stored as serialized array, extract.
    """
    if isinstance(row['y_argmin_mlo'], float):
        return int(row['y_argmin_mlo'])

    arr = str_to_arr(row['y_argmin_mlo'])
    if arr is None:
        return np.nan
    return arr[int(row['x_argmin_mlo'])]


def get_cancer_label(row, max_followup=10, mode='censor_time'):
    """
    Recreates the cancer labeling logic from notebook:
        - censor time
        - binary cancer label
    """
    any_cancer = row["years_to_cancer"] < max_followup

    if any_cancer:
        censor_time = int(row["years_to_cancer"])
    else:
        censor_time = int(min(row["years_to_last_followup"], max_followup))

    if mode == 'censor_time':
        return censor_time
    else:
        return any_cancer


def merge_predictions(filtered_df: pd.DataFrame,
                      asym_preds: pd.DataFrame,
                      mirai_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Merge filtered metadata with AsymMirai + MIRAI predictions.
    Apply correction transforms & compute censor labels.
    """
    # 1. Merge asymmetry predictions
    merged = filtered_df.merge(asym_preds, on='exam_id', suffixes=['', '_asym'])
    # 2. Merge MIRAI predictions
    merged = merged.merge(mirai_preds, on='exam_id', suffixes=['', '_mirai'])

    # Fix y_argmin arrays → scalar
    merged['y_argmin_cc'] = merged.apply(correct_y_argmin_cc, axis=1)
    merged['y_argmin_mlo'] = merged.apply(correct_y_argmin_mlo, axis=1)

    # Canonical naming for later shift computations
    merged['mlo_y_argmin'] = merged['y_argmin_mlo']
    merged['mlo_x_argmin'] = merged['x_argmin_mlo']
    merged['cc_y_argmin'] = merged['y_argmin_cc']
    merged['cc_x_argmin'] = merged['x_argmin_cc']

    # Risk score normalization
    merged['asymmetries'] = merged['prediction_pos']
    merged['mlo_asym'] = merged['asymmetries']
    merged['cc_asym'] = merged['asymmetries']

    # Compute censor_time and binary cancer label
    merged['censor_time'] = merged.apply(get_cancer_label,
                                         args=(10, 'censor_time'), axis=1)
    merged['any_cancer'] = merged.apply(get_cancer_label,
                                        args=(10, 'any_cancer'), axis=1)

    # Simplify to exam-level
    simplified = merged.drop_duplicates(['exam_id']).copy()

    return merged, simplified


# ============================================================
# Section 7 — AUC Utilities & Evaluation
# ============================================================

def include_exam_and_label(censor_time, gold, followup):
    """
    Determines whether to include an exam for ROC evaluation
    and assigns label:
        valid_pos = cancer & censor_time <= followup
        valid_neg = censor_time >= followup
    """
    valid_pos = gold and censor_time <= followup
    valid_neg = censor_time >= followup
    include = valid_pos or valid_neg
    label = valid_pos
    return include, label


def get_arrs_for_auc(probs, censor_times, golds, followup):
    """
    Convert raw probability + censor information into
    arrays suitable for sklearn's ROC.
    """
    probs_eval, labels_eval = [], []

    for p, c, g in zip(probs, censor_times, golds):
        include, label = include_exam_and_label(c, g, followup)
        if include:
            # flatten array predictions
            if isinstance(p, (list, np.ndarray)) and not isinstance(p, float):
                p = float(p[1])
            probs_eval.append(p)
            labels_eval.append(label)

    return probs_eval, labels_eval


def compute_auc_for_asymm(probs, censor_times, golds,
                          followup, calculate_curve=False):
    """
    Legacy AUC implementation from notebook.
    Computes both ROC AUC and AP AUC.
    """
    probs_eval, labels_eval = get_arrs_for_auc(probs, censor_times, golds, followup)

    try:
        auc = sklearn.metrics.roc_auc_score(labels_eval, probs_eval, average='samples')
        avg_prec = sklearn.metrics.average_precision_score(labels_eval, probs_eval,
                                                           average='samples')
        if calculate_curve:
            fpr, tpr, _ = sklearn.metrics.roc_curve(labels_eval, probs_eval)
            plt.plot(fpr, tpr)
            plt.title(f"Year {followup + 1} AsymMirai ROC Curve")
            plt.show()
    except Exception as e:
        print("AUC calculation failed:", e)
        auc, avg_prec = float('nan'), float('nan')

    return auc, avg_prec, labels_eval

# ============================================================
# Section 8 — Centroid Shift Analysis
# ============================================================

def compute_centroid_shifts(df_simplified: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CC/MLO/total centroid shifts across longitudinal exams.

    Adds columns:
        - centroid_mlo_shift_med
        - centroid_cc_shift_med
        - centroid_total_shift_med
        - centroid_mlo_shift_mean
        - centroid_cc_shift_mean
        - centroid_total_shift_mean
        - centroid_total_shifts (stringified list)
        - centroid_total_shift_min
        - centroid_total_shift_max
        - centroid_total_shift_from_last
        - prev_exam_time_delta_mon
        - num_prev_exams
    """

    merged = df_simplified.copy()

    for pid in tq(merged['patient_id'].unique(),
                  total=merged['patient_id'].nunique(),
                  desc="Computing centroid shifts"):

        patient = merged[merged['patient_id'] == pid].sort_values(
            'years_to_last_followup', ascending=False)

        exam_ids = patient['exam_id'].unique()

        for eid in exam_ids:
            cur_exam = patient[patient['exam_id'] == eid]
            prev_exams = patient[
                patient['years_to_last_followup'] > cur_exam['years_to_last_followup'].item()
            ]

            merged.loc[merged['exam_id'] == eid, 'num_prev_exams'] = prev_exams['exam_id'].nunique()

            mlo_shifts = []
            cc_shifts = []
            total_shifts = []

            previous_exam_time_delta = float('inf')
            previous_exam_total_shift = np.nan

            # Coordinates of current exam
            old_x_mlo, old_y_mlo = cur_exam['mlo_x_argmin'].item(), cur_exam['mlo_y_argmin'].item()
            old_x_cc, old_y_cc = cur_exam['cc_x_argmin'].item(), cur_exam['cc_y_argmin'].item()

            for prev_eid in prev_exams['exam_id'].unique():
                prev_exam = prev_exams[prev_exams['exam_id'] == prev_eid]

                # Coordinates for previous exam
                new_x_mlo, new_y_mlo = prev_exam['mlo_x_argmin'].item(), prev_exam['mlo_y_argmin'].item()
                new_x_cc, new_y_cc = prev_exam['cc_x_argmin'].item(), prev_exam['cc_y_argmin'].item()

                # Compute Euclidean shifts
                mlo_shift = np.sqrt((old_x_mlo - new_x_mlo)**2 + (old_y_mlo - new_y_mlo)**2)
                cc_shift = np.sqrt((old_x_cc - new_x_cc)**2 + (old_y_cc - new_y_cc)**2)
                total_shift = np.sqrt(mlo_shift**2 + cc_shift**2)

                mlo_shifts.append(mlo_shift)
                cc_shifts.append(cc_shift)
                total_shifts.append(total_shift)

                # Compute time delta for this pair of exams (months)
                time_delta = (prev_exam['years_to_last_followup'].item() -
                              cur_exam['years_to_last_followup'].item()) * 12

                if time_delta < previous_exam_time_delta:
                    previous_exam_time_delta = time_delta
                    previous_exam_total_shift = total_shift

            query = merged['exam_id'] == eid

            # Median/Mean shifts
            merged.loc[query, 'centroid_mlo_shift_med'] = safe_apply(np.median, mlo_shifts)
            merged.loc[query, 'centroid_cc_shift_med'] = safe_apply(np.median, cc_shifts)
            merged.loc[query, 'centroid_total_shift_med'] = safe_apply(np.median, total_shifts)

            merged.loc[query, 'centroid_mlo_shift_mean'] = safe_apply(np.mean, mlo_shifts)
            merged.loc[query, 'centroid_cc_shift_mean'] = safe_apply(np.mean, cc_shifts)
            merged.loc[query, 'centroid_total_shift_mean'] = safe_apply(np.mean, total_shifts)

            # Range of shifts
            merged.loc[query, 'centroid_total_shifts'] = np.nan if len(total_shifts) == 0 else str(total_shifts)

            merged.loc[query, 'centroid_total_shift_min'] = safe_apply(np.min, total_shifts)
            merged.loc[query, 'centroid_total_shift_max'] = safe_apply(np.max, total_shifts)

            # Nearest prior exam
            merged.loc[query, 'centroid_total_shift_from_last'] = previous_exam_total_shift
            merged.loc[query, 'prev_exam_time_delta_mon'] = previous_exam_time_delta

    return merged


def compute_risk_deltas(df_simplified: pd.DataFrame) -> pd.DataFrame:
    """
    Compute absolute and signed deltas for MIRAI year_1..year_5 risk +
    asymmetry deltas. Matches second large loop in notebook.

    Adds columns:
        - year_k_risk_delta
        - year_k_risk_delta_med
        - year_k_risk_delta_mean
        - asymmetries_delta
        - prev_exam_time_delta_mon (for delta context)
    """

    merged = df_simplified.copy()

    for pid in tq(merged['patient_id'].unique(),
                  total=merged['patient_id'].nunique(),
                  desc="Computing risk deltas"):

        patient = merged[merged['patient_id'] == pid].sort_values(
            'years_to_last_followup', ascending=False)
        exam_ids = patient['exam_id'].unique()

        for eid in exam_ids:
            cur = patient[patient['exam_id'] == eid]
            prev = patient[
                patient['years_to_last_followup'] >
                cur['years_to_last_followup'].item()
            ]

            time_delta_closest = float('inf')
            y1_d, y2_d, y3_d, y4_d, y5_d = np.nan, np.nan, np.nan, np.nan, np.nan
            a_d = np.nan

            year_1_deltas, year_2_deltas, year_3_deltas = [], [], []
            year_4_deltas, year_5_deltas, asym_deltas = [], [], []

            for prev_eid in prev['exam_id'].unique():
                old = cur
                new = prev[prev['exam_id'] == prev_eid]

                td = (new['years_to_last_followup'].item() -
                      old['years_to_last_followup'].item()) * 12

                # absolute change calculations
                year_1_deltas.append(abs(new['year_1_risk'].item() - old['year_1_risk'].item()))
                year_2_deltas.append(abs(new['year_2_risk'].item() - old['year_2_risk'].item()))
                year_3_deltas.append(abs(new['year_3_risk'].item() - old['year_3_risk'].item()))
                year_4_deltas.append(abs(new['year_4_risk'].item() - old['year_4_risk'].item()))
                year_5_deltas.append(abs(new['year_5_risk'].item() - old['year_5_risk'].item()))
                asym_deltas.append(abs(new['asymmetries'].item() - old['asymmetries'].item()))

                # closest previous exam
                if td < time_delta_closest:
                    y1_d = new['year_1_risk'].item() - old['year_1_risk'].item()
                    y2_d = new['year_2_risk'].item() - old['year_2_risk'].item()
                    y3_d = new['year_3_risk'].item() - old['year_3_risk'].item()
                    y4_d = new['year_4_risk'].item() - old['year_4_risk'].item()
                    y5_d = new['year_5_risk'].item() - old['year_5_risk'].item()
                    a_d = new['asymmetries'].item() - old['asymmetries'].item()

                    time_delta_closest = td

            # Store computed fields
            q = merged['exam_id'] == eid

            merged.loc[q, 'year_1_risk_delta'] = y1_d
            merged.loc[q, 'year_2_risk_delta'] = y2_d
            merged.loc[q, 'year_3_risk_delta'] = y3_d
            merged.loc[q, 'year_4_risk_delta'] = y4_d
            merged.loc[q, 'year_5_risk_delta'] = y5_d
            merged.loc[q, 'asymmetries_delta'] = a_d

            merged.loc[q, 'year_1_risk_delta_med'] = safe_apply(np.median, year_1_deltas)
            merged.loc[q, 'year_2_risk_delta_med'] = safe_apply(np.median, year_2_deltas)
            merged.loc[q, 'year_3_risk_delta_med'] = safe_apply(np.median, year_3_deltas)
            merged.loc[q, 'year_4_risk_delta_med'] = safe_apply(np.median, year_4_deltas)
            merged.loc[q, 'year_5_risk_delta_med'] = safe_apply(np.median, year_5_deltas)
            merged.loc[q, 'asymmetries_delta_med'] = safe_apply(np.median, asym_deltas)

            merged.loc[q, 'year_1_risk_delta_mean'] = safe_apply(np.mean, year_1_deltas)
            merged.loc[q, 'year_2_risk_delta_mean'] = safe_apply(np.mean, year_2_deltas)
            merged.loc[q, 'year_3_risk_delta_mean'] = safe_apply(np.mean, year_3_deltas)
            merged.loc[q, 'year_4_risk_delta_mean'] = safe_apply(np.mean, year_4_deltas)
            merged.loc[q, 'year_5_risk_delta_mean'] = safe_apply(np.mean, year_5_deltas)
            merged.loc[q, 'asymmetries_delta_mean'] = safe_apply(np.mean, asym_deltas)

            merged.loc[q, 'prev_exam_time_delta_mon'] = time_delta_closest

    return merged


# ============================================================
# Section 9 — Bootstrap PR AUC Analysis
# ============================================================

def bootstrap_pr_auc(df_simplified: pd.DataFrame,
                      shift_values: List[float],
                      shift_col: str = 'centroid_total_shift_from_last',
                      year: int = 5,
                      n_bootstraps: int = 2000) -> pd.DataFrame:
    """
    Perform bootstrap PR AUC analysis for a list of shift thresholds.

    Args:
        df_simplified:
            Exam-level dataframe with predictions + labels.
        shift_values:
            List of raw shift thresholds (already converted into % shift).
        shift_col:
            Column defining shift constraint (from_last, mean, etc.)
        year:
            Follow-up year (0–4 corresponding to 1–5 years).
        n_bootstraps:
            Number of bootstrap samples.

    Returns:
        DataFrame indexed by shift threshold, containing:
            - full_auc
            - ci_low
            - ci_high
            - n_patients
    """

    results = []

    for shift_val in shift_values:
        subset = df_simplified[df_simplified[shift_col] <= shift_val]

        # Extract appropriate arrays for this year
        probs, labels = get_arrs_for_auc(
            subset['prediction_pos'],
            subset['censor_time'],
            subset['any_cancer'],
            followup=year
        )

        # Convert labels for PR curve: sklearn convention wants positives = 1
        labels_arr = 1 - np.array(labels)

        # Compute full-set PR AUC
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_arr, probs)
        full_auc = sklearn.metrics.auc(recall, precision)

        # Bootstrap sampling
        boot = []
        for _ in tq(range(n_bootstraps), desc=f"Bootstrap shift {shift_val:.3f}"):
            sample = subset.sample(frac=1, replace=True)
            p, l = get_arrs_for_auc(
                sample['prediction_pos'],
                sample['censor_time'],
                sample['any_cancer'],
                followup=year
            )
            l = 1 - np.array(l)
            pr, rc, _ = sklearn.metrics.precision_recall_curve(l, p)
            boot.append(sklearn.metrics.auc(rc, pr))

        boot_s = pd.Series(boot)
        ci_low = boot_s.quantile(0.025)
        ci_high = boot_s.quantile(0.975)

        results.append({
            'shift': shift_val,
            'full_auc': full_auc,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_patients': subset['patient_id'].nunique()
        })

    return pd.DataFrame(results)
# ============================================================

# ============================================================
# Section 10 — Demographic Subgroup Analysis
# ============================================================

# def demographic_auc_analysis(df_simplified: pd.DataFrame,
#                              subgroup_query: pd.Series,
#                              years: int = 5) -> pd.DataFrame:
#     """
#     Compute AUC and CI for a demographic subgroup across 1–5 years.

#     Args:
#         df_simplified:
#             Full exam-level dataframe.
#         subgroup_query:
#             Boolean mask defining subgroup membership.
#         years:
#             Number of prediction years (default: 5).

#     Returns:
#         DataFrame with:
#             - year
#             - auc
#             - ci_low
#             - ci_high
#             - n_patients
#             - n_exams
#     """

#     subset = df_simplified[subgroup_query]
#     results = []

#     print("\n========== Demographic Subgroup Analysis ==========\n")
#     print(f"Subset size: {subset['patient_id'].nunique()} patients, "
#           f"{subset['exam_id'].nunique()} exams\n")

#     for year in range(years):
#         probs, labels = get_arrs_for_auc(
#             subset['prediction_pos'],
#             subset['censor_time'],
#             subset['any_cancer']_]()

# ============================================================
# Section 11 — Plotting Utilities
# ============================================================

def plot_roc_curves(probs_list, labels_list, legends, title="ROC Curves"):
    """
    Plot multiple ROC curves on the same figure.
    Expects lists of (probs, labels).
    """
    plt.figure(figsize=(10, 8))

    for probs, labels, legend in zip(probs_list, labels_list, legends):
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
        plt.plot(fpr, tpr, label=legend)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title(title)
    plt.show()


def plot_auc_vs_shift(auc_df: pd.DataFrame,
                      x_col: str = 'thresh',
                      label: str = 'AUC vs Shift',
                      percent_transform: bool = False):
    """
    Plot AUC and confidence intervals vs. shift threshold.
    Supports shift → percent conversion for readability.
    """
    x = auc_df[x_col]
    if percent_transform:
        x = x * (5 / 64 * 100)

    plt.figure(figsize=(12, 8))
    plt.errorbar(
        x,
        auc_df['auc'],
        yerr=[auc_df['ci_high'] - auc_df['auc']],
        elinewidth=1,
        capsize=3,
        fmt='-o'
    )

    plt.xlabel(f"{x_col} ({'%' if percent_transform else 'raw'})")
    plt.ylabel("AUC")
    plt.title(label)
    plt.grid(True)
    plt.show()


def plot_selected_shift_curves(df_simplified: pd.DataFrame,
                               metric_list: List[str],
                               shift_values: List[float],
                               year: int = 5):
    """
    Recreate the large MIRAI–AsymMirai multi-curve plots in the notebook.
    """
    for metric in metric_list:
        plt.figure(figsize=(12, 8))
        legend = []

        for shift_val in shift_values:
            subset = df_simplified[df_simplified[metric] <= shift_val]

            probs, labels = get_arrs_for_auc(
                subset['prediction_pos'],
                subset['censor_time'],
                subset['any_cancer'],
                followup=year
            )

            fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs)
            df_tmp = pd.DataFrame({'AsymMirai': probs})
            roc = pyroc.ROC(labels, df_tmp)

            auc = roc.auc[0, 0]
            ci = roc.ci(0.05)[:, 0]

            label_str = (
                f"Max shift {shift_val*(5/64*100):.1f}%\n"
                f"AUC {auc:.2f} ({ci[0]:.2f}, {ci[1]:.2f}), "
                f"{subset['patient_id'].nunique()} pts"
            )
            legend.append(label_str)

            plt.plot(fpr, tpr)

        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(legend, prop={'size': 14})
        plt.title(f"ROC Curves for Metric: {metric}")
        plt.show()
        plt.clf()


def plot_bootstrap_pr_curves(results_df: pd.DataFrame,
                             title="Bootstrap PR AUC vs Shift"):
    """
    Plot PR AUC full-set values and confidence intervals
    from bootstrap PR analysis.
    """
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        results_df['shift'] * (5 / 64 * 100),
        results_df['full_auc'],
        yerr=[
            results_df['full_auc'] - results_df['ci_low'],
            results_df['ci_high'] - results_df['full_auc']
        ],
        capsize=3,
        fmt='o-'
    )

    plt.xlabel("Shift Threshold (%)")
    plt.ylabel("PR AUC")
    plt.title(title)
    plt.grid(True)
    plt.show()


# ============================================================
# Section 12 — Main Function Stub
# ============================================================

def main():
    """
    Main entry point for the AsymMirai + MIRAI full pipeline.

    Does:
        - Load and filter dataset
        - Check if predictions already exist
        - Run AsymMirai for missing exams
        - Run MIRAI for missing exams
        - Merge predictions
        - Compute centroid shifts
        - Compute risk deltas
        - Save intermediate outputs

    Does NOT:
        - Run AUC analysis
        - Run bootstrap PR analysis
        - Run demographic analysis
        - Run plotting routines

    Those are available as separate functions you can call manually.
    """

    print("\n=== Loading and filtering dataset ===")
    filtered_df = load_and_prepare_dataset()

    # ---------------------------------------------------------
    # AsymMirai Predictions
    # ---------------------------------------------------------
    print("\n=== Loading AsymMirai predictions ===")
    # try:
    #     asym_preds = pd.read_csv('asymmirai_predictions.csv')
    # except:
    #     print("ERROR: Could not load AsymMirai prediction CSV.")
    #     print("Please update the placeholder path in the script.")
    #     return

    # missing_asym = filtered_df[
    #     ~filtered_df['exam_id'].isin(asym_preds['exam_id'])
    # ]['exam_id'].unique()
    
    model = torch.load(
        'trained_asymmirai.pt',
        weights_only=False,
        map_location=torch.device('cuda:0')
    )
    new_preds = run_validation(model, filtered_df)
    asym_preds = pd.concat([asym_preds, new_preds])

    # if len(missing_asym) > 0:
    #     print(f"AsymMirai: {len(missing_asym)} exams missing — running inference...")
    #     model = torch.load(
    #         './asymmetry_model/training_preds/full_model_epoch_40_3_11_corrected_flex.pt',
    #         map_location=torch.device('cuda:6')
    #     )
    #     val_df = filtered_df[filtered_df['exam_id'].isin(missing_asym)]
    #     new_preds = run_validation(model, val_df)
    #     asym_preds = pd.concat([asym_preds, new_preds])
    # else:
    #     print("AsymMirai predictions complete.")

    # ---------------------------------------------------------
    # MIRAI Predictions
    # ---------------------------------------------------------
    print("\n=== Loading MIRAI predictions ===")
    try:
        mirai_preds = pd.read_csv('/PATH/TO/MIRAI/PREDICTIONS', header=None)
    except:
        print("ERROR: Could not load MIRAI prediction CSV.")
        print("Please update the placeholder path in the script.")
        return

    # Add risk columns to existing MIRAI preds
    for i in range(5):
        mirai_preds[f'year_{i+1}_risk'] = mirai_preds[4 + i]

    # Identify missing MIRAI exams
    missing_mirai = filtered_df[
        ~filtered_df['exam_id'].isin(mirai_preds[mirai_preds.columns[-1]])
    ]['exam_id'].unique()  # last column should be exam_id

    if len(missing_mirai) > 0:
        print(f"MIRAI: {len(missing_mirai)} exams missing — running MIRAI inference...")
        val_df = filtered_df[filtered_df['exam_id'].isin(missing_mirai)]
        input_path = prepare_mirai_input_csv(val_df)
        out_path = run_mirai_inference(input_path)
        new_mirai = load_mirai_predictions(out_path, filtered_df)
        mirai_preds = pd.concat([mirai_preds, new_mirai])
    else:
        print("MIRAI predictions complete.")

    # ---------------------------------------------------------
    # Merge Prediction DataFrames
    # ---------------------------------------------------------
    print("\n=== Merging predictions ===")
    merged_df, simplified_df = merge_predictions(filtered_df, asym_preds, mirai_preds)

    # Save intermediate merged data
    merged_df.to_csv('merged_full_df.csv', index=False)
    simplified_df.to_csv('merged_exam_level_df.csv', index=False)
    print("Saved: merged_full_df.csv, merged_exam_level_df.csv")

    # ---------------------------------------------------------
    # Compute Shifts & Risk Deltas
    # ---------------------------------------------------------
    print("\n=== Computing centroid shifts ===")
    simplified_df = compute_centroid_shifts(simplified_df)

    print("\n=== Computing risk deltas ===")
    simplified_df = compute_risk_deltas(simplified_df)

    # Save enriched dataframe
    simplified_df.to_csv('merged_exam_level_with_shifts.csv', index=False)
    print("Saved: merged_exam_level_with_shifts.csv")

    print("\n=== Pipeline Setup Complete ===")
    print("Heavy analyses (AUC, plots, bootstrap, demographics) can now be")
    print("run via the functions provided in this script.")


def run_full_analysis(
        simplified_df_path="merged_exam_level_with_shifts.csv",
        output_dir="analysis_outputs",
        run_auc=True,
        run_shift_auc=False,
        run_bootstrap_pr=False,
        run_demographics=False,
        year=5
    ):
    """
    Run ALL heavy analysis steps that `main()` intentionally does NOT run.

    Includes:
        - Multi-year ROC AUC analysis
        - Shift-restricted performance curves
        - Bootstrap PR AUC analysis
        - Demographic subgroup AUC analysis
        - All plotting routines

    Parameters:
        simplified_df_path : str
            Path to the precomputed exam-level dataframe with shifts & deltas.

        output_dir : str
            Directory to save plots + CSV outputs.

        run_auc : bool
            Run basic multi-year AUC evaluations.

        run_shift_auc : bool
            Run AUC curves stratified by centroid shifts.

        run_bootstrap_pr : bool
            Perform bootstrap precision-recall AUC analysis.

        run_demographics : bool
            Compute AUCs on demographic subgroups.

        year : int
            Follow-up year for some analyses (default: 5).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Loading simplified exam-level dataframe ===")
    df = pd.read_csv(simplified_df_path)

    # ============================================
    # 1. Basic AUC curves (1–5 years)
    # ============================================
    if run_auc:
        print("\n=== Running multi-year AUC analysis ===")
        auc_results = []
        for y in range(5):
            probs, labels = get_arrs_for_auc(
                df['prediction_pos'], df['censor_time'], df['any_cancer'], followup=y
            )
            roc = pyroc.ROC(labels, pd.DataFrame({'AsymMirai': probs}))
            auc_results.append({
                'year': y+1,
                'auc': roc.auc[0, 0],
                'ci_low': roc.ci(0.05)[0, 0],
                'ci_high': roc.ci(0.05)[1, 0]
            })

        auc_df = pd.DataFrame(auc_results)
        auc_df.to_csv(os.path.join(output_dir, "asymmirai_auc_results.csv"), index=False)
        print("Saved AUC results.")

    # ============================================
    # 2. Shift-restricted AUC curves
    # ============================================
    if run_shift_auc:
        print("\n=== Running shift-restricted AUC analysis ===")
        shift_metrics = ["centroid_total_shift_from_last",
                         "centroid_total_shift_med",
                         "centroid_total_shift_mean"]

        for metric in shift_metrics:
            vals = sorted(df[metric].dropna().unique())
            aucs = []
            for v in vals[:-1]:
                subset = df[df[metric] <= v]
                probs, labels = get_arrs_for_auc(
                    subset['prediction_pos'], subset['censor_time'],
                    subset['any_cancer'], followup=year
                )
                roc = pyroc.ROC(labels, pd.DataFrame({'AsymMirai': probs}))
                aucs.append({
                    'shift': v,
                    'auc': roc.auc[0, 0],
                    'ci_low': roc.ci(0.05)[0, 0],
                    'ci_high': roc.ci(0.05)[1, 0],
                    'n_patients': subset['patient_id'].nunique()
                })

            metric_df = pd.DataFrame(aucs)
            metric_df.to_csv(os.path.join(output_dir, f"shift_auc_{metric}.csv"), index=False)
            print(f"Saved shift AUC results for {metric}.")

    # ============================================
    # 3. Bootstrap PR AUC analysis
    # ============================================
    if run_bootstrap_pr:
        print("\n=== Running bootstrap PR AUC analysis ===")
        shift_vals = [i / (5/64 * 100) for i in [40, 50, 60, 80, 100]]

        pr_df = bootstrap_pr_auc(df, shift_vals, year=year)
        pr_df.to_csv(os.path.join(output_dir, "bootstrap_pr_auc.csv"), index=False)
        print("Saved bootstrap PR AUC results.")

        plot_bootstrap_pr_curves(pr_df,
                                 title="Bootstrap PR AUC vs Shift Threshold")
        plt.savefig(os.path.join(output_dir, "bootstrap_pr_auc_plot.png"), dpi=300)



    # # ============================================
    # # 4. Demographic subgroup analysis
    # # ============================================
    # if run_demographics:
    #     print("\n=== Running demographic subgroup analysis ===")
    #     subgroup_df = example_ethnicity_subgroup(df)
    #     subgroup_df.to_csv(os.path.join(output_dir, "demographic_auc_results.csv"), index=False)
    #     print("Saved demographic AUC results.")

    print("\n=== FULL ANALYSIS COMPLETE ===")


# ============================================================
# Section 13 — Script Entry Point
# ============================================================

if __name__ == "__main__":
    main()
    print("\n=== Running full analysis suite ===")
    run_full_analysis()
