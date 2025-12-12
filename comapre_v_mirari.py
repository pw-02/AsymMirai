import ast
import re
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import sys
sys.path.append('./asymmetry_model')
import sklearn
from torch.utils.data import DataLoader
from asymmetry_model.mirai_metadatasets3 import MiraiMetadatasetS3
import torch.nn.functional as F
from asymmetry_model.embed_explore import crop
import torch
import tqdm
import pyroc

def get_centroid_activation(array, threshold=0.02):
    print(array.shape)
    h, w = array.shape
    am = torch.argmax(array)
    am_h, am_w = am // w, am % w
    
    # Grab all the locations at which activation is within threshold of the max
    candidate_locations = list(((array[am_h, am_w] - array) <= threshold).nonzero())
    
    # First, we're going to grab all the locations that are contiguous with the max
    added_new = True
    contiguous_w_max = [torch.tensor([am_h, am_w])]
    
    while added_new:
        added_new = False
        to_move = []
        for cl_ind, cl in enumerate(candidate_locations):
            for contig_ind, contig in enumerate(contiguous_w_max):
                if abs(cl[0] - contig[0]) <= 1 and abs(cl[1] - contig[1]) <= 1:
                    if abs(cl[0] - contig[0]) == 0 and abs(cl[1] - contig[1]) == 0:
                        continue
                    if cl_ind not in to_move:
                        to_move.append(cl_ind)
                    added_new = True
        
        for index in sorted(to_move, reverse=True):
            contiguous_w_max.append(candidate_locations[index])
            del candidate_locations[index]
            
    # This is a bit of a hack, but the true max gets double counted,
    # so this removes the first time we counted it
    if len(contiguous_w_max) > 1:
        del contiguous_w_max[0]
        
    h_mean, w_mean = 0.0, 0.0
    for cm in contiguous_w_max:
        h_mean += cm[0].item()
        w_mean += cm[1].item()
    print(h_mean, w_mean, contiguous_w_max)
    h_mean /= len(contiguous_w_max)
    w_mean /= len(contiguous_w_max)
    
    return (h_mean, w_mean)



def resize_and_normalize(img, use_crop=False):
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)
    dummy_batch_dim = False

    if np.sum(img) == 0:
        img = torch.tensor(img).expand(1, 3, *img.shape)\
                        .type(torch.FloatTensor)
        return F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')[0]

    # Adding a dummy batch dimension if necessary
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
        dummy_batch_dim = True

    with torch.no_grad():
        if use_crop:
            img = crop(torch.tensor((img - img_mean)/img_std))
        else:
            img = torch.tensor((img - img_mean)/img_std)
        img = img.expand(1, 3, *img.shape)\
                        .type(torch.FloatTensor)
        img_resized = F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')
    #img_resized = img

    if dummy_batch_dim:
        return img_resized[0]
    else:
        return img_resized[0]
    
def run_validation(model, val_df):
    #'exam_id', 'prediction_neg', 'prediction_pos', 'y_argmin_cc',
    #   'x_argmin_cc', 'y_argmin_mlo', 'x_argmin_mlo'

    torch.cuda.set_device(0)
    model = model.eval()
    model.latent_h = 5
    model.latent_w = 5
    model.topk_for_heatmap = None
    model.topk_weights = torch.tensor([1]).cuda()
    model.use_bn = False
    model.learned_asym_mean = model.initial_asym_mean
    model.learned_asym_std = model.initial_asym_std

    batch_size = 1

    val_dataset = MiraiMetadatasetS3(val_df, resizer=resize_and_normalize,
                                     s3_bucket='embdedpng',
                                       mode='val', align_images=False, multiple_pairs_per_exam=False)#, use_crop=use_crop)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=min(10, batch_size))

    with torch.no_grad():
        eids_for_epoch = []
        centroids_h_cc_for_epoch = []
        centroids_w_cc_for_epoch = []
        centroids_h_mlo_for_epoch = []
        centroids_w_mlo_for_epoch = []
        predictions = []

        for index, sample in enumerate(val_dataloader):

            eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
            l_cc_img, r_cc_img, l_mlo_img, r_mlo_img = l_cc_img.cuda(), r_cc_img.cuda(), l_mlo_img.cuda(), r_mlo_img.cuda()
            label = label.cuda()

            output, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)
            eids_for_epoch = eids_for_epoch + list(eid.numpy())
            predictions = predictions + list(output.detach().cpu().numpy())
            for c in range(2):
                for i in range(batch_size):
                    heatmap = other[c]['heatmap'][i]
                    centroid = get_centroid_activation(heatmap)
                    if c == 0:
                        centroids_h_cc_for_epoch.append(centroid[0])
                        centroids_w_cc_for_epoch.append(centroid[1])
                    else:
                        centroids_h_mlo_for_epoch.append(centroid[0])
                        centroids_w_mlo_for_epoch.append(centroid[1])
            df = pd.DataFrame({
                'exam_id': eids_for_epoch,
                'prediction_neg': list(1 - np.array(predictions)),
                'prediction_pos': predictions,
                'y_argmin_cc': centroids_h_cc_for_epoch,
                'x_argmin_cc': centroids_w_cc_for_epoch,
                'y_argmin_mlo': centroids_h_mlo_for_epoch,
                'x_argmin_mlo': centroids_w_mlo_for_epoch
            })
            print(index)
            df.to_csv('tmp_val_run.csv', index=False)
    return df


def construct_dataset(asym_data_file, mirai_data_file):
    filtered_input_df = pd.read_csv(asym_data_file)
    print(filtered_input_df['patient_id'].unique().shape)
    print(filtered_input_df['exam_id'].unique().shape)

    mirai_form_df = pd.read_csv(mirai_data_file)
    # filtered = filtered_input_df.join(mirai_form_df, lsuffix='', rsuffix='_mirai', on='exam_id')
    filtered = filtered_input_df.merge(mirai_form_df, on='file_path', how='inner')
    print(filtered['patient_id'].unique().shape)
    print(filtered[filtered['years_to_cancer'] < 99]['patient_id'].unique().shape)
    print(filtered['exam_id'].unique().shape)
    for lower, upper in [(0, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 200)]:
        print(lower, upper)
        age_grp = filtered[(filtered['age_at_study'] >= lower) 
                                    & (filtered['age_at_study'] < upper)]
        print(age_grp['exam_id'].unique().shape)
        print(age_grp[age_grp['years_to_cancer'] < 99]['exam_id'].unique().shape)
    filtered.drop_duplicates(['patient_id']).groupby('ETHNICITY_DESC').count()
    filtered['exam_id_y'].unique().shape
    filtered['exam_id'] = filtered['exam_id_y']
    filtered['patient_id'] = filtered['patient_id_y']
    filtered['view'] = filtered['view_y']
    filtered['laterality'] = filtered['laterality_y']
    filtered['file_path'] = filtered['file_path_y']
    return filtered



def compute_auc_for_asymm(probs, censor_times, golds, followup, calculate_curve=False):
    def include_exam_and_determine_label(censor_time, gold):
        valid_pos = gold and censor_time <= followup
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        return included, label

    probs_for_eval, golds_for_eval = [], []
    
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(censor_time, gold)
        if include:
            probs_for_eval.append(prob_arr)
            golds_for_eval.append(label)
    try:
        auc = sklearn.metrics.roc_auc_score(golds_for_eval, probs_for_eval, average='samples')
        avg_precision = sklearn.metrics.average_precision_score(golds_for_eval, probs_for_eval, average='samples')
        if calculate_curve:
                fpr, tpr, thresh = sklearn.metrics.roc_curve(golds_for_eval, probs_for_eval)
                plt.plot(fpr, tpr)
                plt.title(f"Year {followup + 1} Asymmetry ROC Curve")
                plt.show()
    except Exception as e:
        print("Failed to calculate AUC because {}".format(e))
        auc, avg_precision = ['NA']*2

    return auc, avg_precision, golds_for_eval

def get_label(row, max_followup=10, mode='censor_time'):
    any_cancer = row["years_to_cancer"] < max_followup
    cancer_key = "years_to_cancer"

    y =  any_cancer

    if y:
        censor_time = int(row[cancer_key])

    else:
        censor_time = int(min(row["years_to_last_followup"], max_followup))# - 1)

    #y_mask = np.array([1] * (censor_time+1) + [0]* (self.args.max_followup - (censor_time+1) ))
    #assert len(y_mask) == self.args.max_followup
    if mode == 'censor_time':
        return censor_time
    else:
        return any_cancer
    
def str_to_arr(arr_str):
    #if type(arr_str) is not str:
    #    return None
    arr_str = re.sub('\s+', ',', arr_str)
    arr_str = arr_str.replace('[,', '[')
    return np.array(eval(arr_str))

def correct_y_argmin_cc(row):
    if type(row['y_argmin_cc']) is float:
        return int(row['y_argmin_cc'])
    armin_arr = str_to_arr(row['y_argmin_cc'])
    #if armin_arr is None:
    #    return None
    argmin = armin_arr[int(row['x_argmin_cc'])]
    return argmin

def correct_y_argmin_mlo(row):
    if type(row['y_argmin_mlo']) is float:
        return int(row['y_argmin_mlo'])
    armin_arr = str_to_arr(row['y_argmin_mlo'])
    #if armin_arr is None:
    #    return None
    argmin = armin_arr[int(row['x_argmin_mlo'])]
    return argmin

def get_arrs_for_auc(probs, censor_times, golds, followup):
    
    def include_exam_and_determine_label(censor_time, gold):
        valid_pos = gold and censor_time <= followup
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        # if label == True:
        #     pass
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(censor_time, gold)
        # if gold == True:
        #     pass
        # if label == True:
        #     pass
        if include:
            probs_for_eval.append(prob_arr)
            golds_for_eval.append(label)
    return probs_for_eval, golds_for_eval


def craete_asum_auc_plot(merged_df_filtered: pd.DataFrame):

    # merged_df_filtered = filtered.merge(asym_preds, on='exam_id', suffixes=['', '_asym'])

    merged_df_filtered['exam_id'].unique().shape
    res = merged_df_filtered.apply(get_label, args=(10, 'censor_time'), axis=1)
    merged_df_filtered['censor_time'] = res
    res = merged_df_filtered.apply(get_label, args=(10, 'any_cancer'), axis=1)
    merged_df_filtered['any_cancer'] = res
    merged_df_filtered_simplified = merged_df_filtered.drop_duplicates(['exam_id'])

    merged_df_filtered_copy = merged_df_filtered.copy()
    merged_df_filtered_copy.to_csv('tmp_merged_df_filtered_copy.csv', index=False)

    merged_df_filtered_copy = pd.read_csv('tmp_merged_df_filtered_copy.csv')
    
    print(merged_df_filtered_copy['prediction_pos'])


    merged_df_filtered_simplified['y_argmin_cc'] = merged_df_filtered_simplified.apply(correct_y_argmin_cc, axis=1)
    merged_df_filtered_simplified['y_argmin_mlo'] = merged_df_filtered_simplified.apply(correct_y_argmin_mlo, axis=1)

    merged_df_filtered_simplified['mlo_y_argmin'] = merged_df_filtered_simplified['y_argmin_mlo']
    merged_df_filtered_simplified['mlo_x_argmin'] = merged_df_filtered_simplified['x_argmin_mlo']
    merged_df_filtered_simplified['cc_y_argmin'] = merged_df_filtered_simplified['y_argmin_cc']
    merged_df_filtered_simplified['cc_x_argmin'] = merged_df_filtered_simplified['x_argmin_cc']

    merged_df_filtered_simplified['asymmetries'] = merged_df_filtered_simplified['prediction_pos']
    merged_df_filtered_simplified['mlo_asym'] = merged_df_filtered_simplified['asymmetries']
    merged_df_filtered_simplified['cc_asym'] = merged_df_filtered_simplified['asymmetries']

    legend = []
    for year in range(5):
        # probs_mirai, labels = get_arrs_for_auc(merged_df_filtered_simplified[f'year_{year+1}_risk'],
        #                                 merged_df_filtered_simplified['censor_time'],
        #                                 merged_df_filtered_simplified['any_cancer'],
        #                                 year)
        probs_asymmirai, labels = get_arrs_for_auc(merged_df_filtered_simplified[f'prediction_pos'],
                                        merged_df_filtered_simplified['censor_time'],
                                        merged_df_filtered_simplified['any_cancer'],
                                        year)
        for i, v in enumerate(probs_asymmirai):
            # Convert "[x, y]" string into real list
            if isinstance(v, str):
                lst = ast.literal_eval(v)
                probs_asymmirai[i] = float(lst[1])   # pos class
            else:
                probs_asymmirai[i] = float(v)
    
        # Build dataframe with ONLY AsymMirai
        df = pd.DataFrame({'AsymMirai': probs_asymmirai})

        # ROC object
        roc = pyroc.ROC(labels, df)

        # Compute sklearn ROC curve for plotting
        fpr, tpr, _ = sklearn.metrics.roc_curve(labels, probs_asymmirai)
        matplotlib.rc('xtick', labelsize=19) 
        matplotlib.rc('ytick', labelsize=19) 
        plt.rcParams.update({'axes.labelsize': 40})
        # Plot
        plt.plot(fpr, tpr)

        # # Extract AUC + CI
        auc = roc.auc[0]
        # ci_low, ci_high =  roc.ci(0.05)[:, 0][0], roc.ci(0.05)[:, 0][1]
        # try:
        #     legend.append(f"Year {year+1} AUC: {auc:.2f} ({ci_low:.2f}, {ci_high:.2f})")
        # except:
        #     legend.append(f"Year {year+1} AUC: {auc} ({ci_low}, {ci_high})")
        legend.append(f"Year {year+1} AUC: {auc}")

        print(f"AsymMirai year {year+1} 95% CI:", roc.ci(0.05)[0])

       
        # df = pd.DataFrame({
        #     'AsymMirai': probs_asymmirai,
        #     'Mirai': probs_asymmirai
        # })
        # roc = pyroc.ROC(labels,
        #                 df)
        
        # fpr, tpr, thresh = sklearn.metrics.roc_curve(labels, probs_asymmirai)
        # matplotlib.rc('xtick', labelsize=19) 
        # matplotlib.rc('ytick', labelsize=19) 
        # plt.rcParams.update({'axes.labelsize': 40})
        
        # plt.plot(fpr, tpr)
        # auc = roc.auc[0, 1]
        # legend.append("Year {:.0f} AUC: {:.2f} ({:.2f}, {:.2f})".format(
        #     year+1, auc, roc.ci(0.05)[:, 1][0], roc.ci(0.05)[:, 1][1])
        # )
        # print(f'AsymMirai year {year + 1} 95% CI: \t', roc.ci(0.05)[:, 0])
        #print(f'Mirai year {year + 1} 95% CI: \t\t', roc.ci(0.05)[:, 1])
    plt.legend(legend, prop={'size': 20})
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    return merged_df_filtered_simplified

def safe_apply(fun, array):
    if len(array) == 0:
        return np.nan
    else:
        val = fun(array)
        if val == np.nan:
            print(array, val, fun)
        return val


def hex_to_tuple(color, alpha=0.2):
    color = color[1:]
    return tuple([int(color[i:i+2], 16) / 255 for i in (0, 2, 4)] + [alpha])



def test(merged_df_filtered_simplified: pd.DataFrame):
    for pid in merged_df_filtered_simplified['patient_id'].unique():
        cur_patient = merged_df_filtered_simplified[merged_df_filtered_simplified['patient_id'] == pid]
        cur_patient = cur_patient.sort_values('years_to_last_followup', ascending=False)
        
        # For each exam with a following exam
        for i in range(cur_patient.shape[0]):
            mlo_shifts = []
            cc_shifts = []
            total_shifts = []
            previous_exam_time_delta = []
            
            cur_exam = cur_patient[cur_patient['exam_id'] == cur_patient['exam_id'].unique()[i]]
            prev_exams = cur_patient[cur_patient['years_to_last_followup'] > cur_exam['years_to_last_followup'].values[0]]
            merged_df_filtered_simplified.loc[merged_df_filtered_simplified['exam_id'] == cur_exam['exam_id'].values[0], 'num_prev_exams'] = len(prev_exams['exam_id'].unique())

            previous_exam_time_delta = np.inf
            previous_exam_shift = np.nan

            for cur_eid in prev_exams['exam_id'].unique():
                e_other = prev_exams[prev_exams['exam_id'] == cur_eid]


                old_x_mlo, old_y_mlo = cur_exam['x_argmin_mlo'].values[0], cur_exam['y_argmin_mlo'].values[0]
                old_x_cc, old_y_cc = cur_exam['x_argmin_cc'].values[0], cur_exam['y_argmin_cc'].values[0]

                new_x_mlo, new_y_mlo = e_other['x_argmin_mlo'].values[0], e_other['y_argmin_mlo'].values[0]
                new_x_cc, new_y_cc = e_other['x_argmin_cc'].values[0], e_other['y_argmin_cc'].values[0]

                mlo_shift = ((old_x_mlo - new_x_mlo) ** 2 + (old_y_mlo - new_y_mlo) ** 2) ** 0.5
                mlo_shifts.append(mlo_shift)
                
                cc_shift = ((old_x_cc - new_x_cc) ** 2 + (old_y_cc - new_y_cc) ** 2) ** 0.5
                cc_shifts.append(cc_shift)
                
                total_shift = (mlo_shift ** 2 + cc_shift ** 2) ** 0.5
                total_shifts.append(total_shift)

                this_exam_time_delta = (e_other['years_to_last_followup'].item() - cur_exam['years_to_last_followup'].item()) * 12

                if this_exam_time_delta < previous_exam_time_delta:
                    previous_exam_time_delta = this_exam_time_delta
                    previous_exam_shift = total_shift
                    
            exam_query = merged_df_filtered_simplified['exam_id'] == cur_exam['exam_id'].values[0]
            merged_df_filtered_simplified.loc[exam_query, 'centroid_mlo_shift_med'] = safe_apply(np.median, mlo_shifts)
            merged_df_filtered_simplified.loc[exam_query, 'centroid_cc_shift_med'] = safe_apply(np.median, cc_shifts)
            merged_df_filtered_simplified.loc[exam_query, 'centroid_total_shift_med'] = safe_apply(np.median, total_shifts)

            merged_df_filtered_simplified.loc[exam_query, 'centroid_mlo_shift_mean'] = safe_apply(np.mean, mlo_shifts)
            merged_df_filtered_simplified.loc[exam_query, 'centroid_cc_shift_mean'] = safe_apply(np.mean, cc_shifts)
            merged_df_filtered_simplified.loc[exam_query, 'centroid_total_shift_mean'] = safe_apply(np.mean, total_shifts)

            merged_df_filtered_simplified.loc[exam_query, 'centroid_total_shifts'] = np.nan if len(total_shifts) == 0 else str(total_shifts)

            merged_df_filtered_simplified.loc[exam_query, 'centroid_total_shift_min'] = safe_apply(np.min, total_shifts)
            merged_df_filtered_simplified.loc[exam_query, 'centroid_total_shift_max'] = safe_apply(np.max, total_shifts)
            merged_df_filtered_simplified.loc[exam_query, 'centroid_total_shift_from_last'] = previous_exam_shift
            merged_df_filtered_simplified.loc[exam_query, 'prev_exam_time_delta_mon'] = previous_exam_time_delta

        
    for pid in merged_df_filtered_simplified['patient_id'].unique():
        cur_patient = merged_df_filtered_simplified[merged_df_filtered_simplified['patient_id'] == pid]
        cur_patient = cur_patient.sort_values('years_to_last_followup', ascending=False)
        
        # For each exam with a following exam
        for i in range(cur_patient.shape[0]):
            year_1_deltas = []
            year_2_deltas = []
            year_3_deltas = []
            year_4_deltas = []
            year_5_deltas = []
            asym_deltas = []
            previous_exam_time_delta = []
            
            cur_exam = cur_patient[cur_patient['exam_id'] == cur_patient['exam_id'].unique()[i]]
            prev_exams = cur_patient[cur_patient['years_to_last_followup'] > cur_exam['years_to_last_followup'].values[0]]
            merged_df_filtered_simplified.loc[merged_df_filtered_simplified['exam_id'] == cur_exam['exam_id'].values[0], 'num_prev_exams'] = len(prev_exams['exam_id'].unique())

            previous_exam_time_delta = np.inf
            previous_exam_shift = np.nan
            year_1_delta = np.nan
            year_2_delta = np.nan
            year_3_delta = np.nan
            year_4_delta = np.nan
            year_5_delta = np.nan
            asym_delta = np.nan

            for cur_eid in prev_exams['exam_id'].unique():
                e_other = prev_exams[prev_exams['exam_id'] == cur_eid]

                this_exam_time_delta = (e_other['years_to_last_followup'].item() - cur_exam['years_to_last_followup'].item()) * 12
                
                year_1_deltas.append(abs(e_other['year_1_risk'].item() - cur_exam['year_1_risk'].item()))
                year_2_deltas.append(abs(e_other['year_2_risk'].item() - cur_exam['year_2_risk'].item()))
                year_3_deltas.append(abs(e_other['year_3_risk'].item() - cur_exam['year_3_risk'].item()))
                year_4_deltas.append(abs(e_other['year_4_risk'].item() - cur_exam['year_4_risk'].item()))
                year_5_deltas.append(abs(e_other['year_5_risk'].item() - cur_exam['year_5_risk'].item()))
                asym_deltas.append(abs(e_other['asymmetries'].item() - cur_exam['asymmetries'].item()))

                if this_exam_time_delta < previous_exam_time_delta:

                    year_1_delta = e_other['year_1_risk'].item() - cur_exam['year_1_risk'].item()
                    year_2_delta = e_other['year_2_risk'].item() - cur_exam['year_2_risk'].item()
                    year_3_delta = e_other['year_3_risk'].item() - cur_exam['year_3_risk'].item()
                    year_4_delta = e_other['year_4_risk'].item() - cur_exam['year_4_risk'].item()
                    year_5_delta = e_other['year_5_risk'].item() - cur_exam['year_5_risk'].item()
                    asym_delta = e_other['asymmetries'].item() - cur_exam['asymmetries'].item()
                    
                    previous_exam_time_delta = this_exam_time_delta
                    
            exam_query = merged_df_filtered_simplified['exam_id'] == cur_exam['exam_id'].values[0]
            merged_df_filtered_simplified.loc[exam_query, 'year_1_risk_delta'] = year_1_delta
            merged_df_filtered_simplified.loc[exam_query, 'year_2_risk_delta'] = year_2_delta
            merged_df_filtered_simplified.loc[exam_query, 'year_3_risk_delta'] = year_3_delta
            merged_df_filtered_simplified.loc[exam_query, 'year_4_risk_delta'] = year_4_delta
            merged_df_filtered_simplified.loc[exam_query, 'year_5_risk_delta'] = year_5_delta
            merged_df_filtered_simplified.loc[exam_query, 'asymmetries'] = asym_delta
            
            merged_df_filtered_simplified.loc[exam_query, 'year_1_risk_delta_med'] = safe_apply(np.median, year_1_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_2_risk_delta_med'] = safe_apply(np.median, year_2_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_3_risk_delta_med'] = safe_apply(np.median, year_3_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_4_risk_delta_med'] = safe_apply(np.median, year_4_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_5_risk_delta_med'] = safe_apply(np.median, year_5_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'asymmetries_med'] = safe_apply(np.median, asym_deltas)
            
            merged_df_filtered_simplified.loc[exam_query, 'year_1_risk_delta_mean'] = safe_apply(np.mean, year_1_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_2_risk_delta_mean'] = safe_apply(np.mean, year_2_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_3_risk_delta_mean'] = safe_apply(np.mean, year_3_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_4_risk_delta_mean'] = safe_apply(np.mean, year_4_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'year_5_risk_delta_mean'] = safe_apply(np.mean, year_5_deltas)
            merged_df_filtered_simplified.loc[exam_query, 'asymmetries_mean'] = safe_apply(np.mean, asym_deltas)
            
            merged_df_filtered_simplified.loc[exam_query, 'prev_exam_time_delta_mon'] = previous_exam_time_delta
           
            print(merged_df_filtered_simplified['centroid_total_shift_from_last'])


    asyms = []
    predictions_pos = []
    predictions_neg = []
    for i, v in enumerate(merged_df_filtered_simplified['asymmetries']):
        if type(v) is np.ndarray:
            asyms.append(float(v[1]))
            predictions_pos.append(float(v[1]))
            predictions_neg.append(float(v[0]))
        else:
            asyms.append(v)
            predictions_pos.append(v)
            predictions_neg.append(1 - v)
                                
    merged_df_filtered_simplified.loc[:, 'asymmetries'] = asyms
    merged_df_filtered_simplified.loc[:, 'prediction_pos'] = predictions_pos
    merged_df_filtered_simplified.loc[:, 'prediction_neg'] = predictions_neg
                

    asym_location_aucs = {}
    year=5
    asym_loc_stats = ('med', 'from_last', 'mean')#'min', 'max', 'mean', 'from_last')

    for stat in asym_loc_stats: 
        aucs_asym = []
        thresholds = []
        included_exams = []
        included_patients = []
        cancer_patients = []
        ci_low = []
        ci_high = []

        col = f'centroid_total_shift_{stat}'

        run_size = len(merged_df_filtered_simplified[col].unique())
        for ind, delta_thresh in tqdm(enumerate(merged_df_filtered_simplified.sort_values(col)[col].unique()[:-1]), total=run_size):

            if col.startswith('from_last'):
                query = (merged_df_filtered_simplified[col] <= delta_thresh) & (merged_df_filtered_simplified['prev_exam_time_delta_mon'] <= 18)
            else:
                query = merged_df_filtered_simplified[col] <= delta_thresh

            subset = merged_df_filtered_simplified[query]
            
            probs_asymmirai, labels = get_arrs_for_auc(subset[f'prediction_pos'],
                                            subset['censor_time'],
                                            subset['any_cancer'],
                                            year)
            try:
                df = pd.DataFrame({
                'AsymMirai': probs_asymmirai,
                'Mirai': probs_asymmirai
                })
                roc = pyroc.ROC(labels, df)
                
                auc = roc.auc[0, 1]
                ci_low.append(roc.ci(0.05)[0, 0])
                ci_high.append(roc.ci(0.05)[1, 0])
            except:
                auc, avg_precision, include = compute_auc_for_asymm(subset[f'asymmetries'].values, 
                                    subset['censor_time'].values, 
                                    subset['any_cancer'].values, year)
                ci_low.append(-1)
                ci_high.append(2)
                
            aucs_asym.append(auc)
            thresholds.append(delta_thresh)
            included_exams.append(subset['exam_id'].nunique())
            included_patients.append(subset['patient_id'].nunique())
            cancer_patients.append(subset[subset['any_cancer']]['patient_id'].nunique())
        
        asym_location_aucs[stat] = {'auc': aucs_asym, 'thresh': thresholds, 
                                    'exam': included_exams, 'patient': included_patients,
                                    'cancer_patients': cancer_patients,
                                    'ci_low': ci_low, 'ci_high': ci_high}
        
    auc_dfs = {}
    auc_dfs.update({f'centroid-asym-loc-{stat}': pd.DataFrame.from_dict(asym_location_aucs[stat]) for stat in asym_loc_stats})
    [(f'{stat}-{field}', len(asym_location_aucs[stat][field])) for stat in asym_loc_stats for field in ('auc', 'thresh', 'exam', 'patient', 'cancer_patients', 'ci_low', 'ci_high')]
    pd.DataFrame(auc_dfs).to_csv('6_28_auc_dfs.csv', index=False)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_in_tuple = [hex_to_tuple(c) for c in colors]
    legend = list(auc_dfs.keys())
    for x_axis in ['patient', 'thresh']:
        for i, metric in enumerate(['centroid-asym-loc-from_last']):
            
            plt.errorbar(auc_dfs[metric][x_axis][3:-1] if 'thresh' not in x_axis else auc_dfs[metric][x_axis][3:-1] * 5/64 * 100, 
                    auc_dfs[metric]['auc'][3:-1],
                    auc_dfs[metric]['ci_high'][3:-1] - auc_dfs[metric]['auc'][3:-1],
                    ecolor=colors_in_tuple[i],
                    elinewidth=1)
            
        matplotlib.rc('xtick', labelsize=19) 
        matplotlib.rc('ytick', labelsize=19) 
        plt.rcParams.update({'axes.labelsize': 40})
        
        if 'patient' in x_axis:
            plt.xlabel('# Patients Included')
            plt.legend(['Consistency with Previous Exam'])
        else:
            plt.xlabel('Maximum Window Shift %')
            plt.xticks([50, 250, 450, 650, 850])
            plt.legend(['Consistency with Previous Exam'])
            plt.axvline(50, linestyle='dashed', c='black')
            
        plt.ylabel("AUC")
        plt.savefig(f'./consistenct_auc_by_{x_axis}.png', dpi=300)
        plt.clf()





if __name__ == "__main__":
    asym_input_file = 'data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW_test.csv'
    asym_preds_file = 'tmp_val_run.csv'
    filtered = pd.read_csv(asym_input_file)
    asym_preds = pd.read_csv(asym_preds_file)
    merged_df_filtered = filtered.merge(asym_preds, on='exam_id', suffixes=['', '_asym'])
    merged_df_filtered_simplified = craete_asum_auc_plot(merged_df_filtered)

    
    # test(merged_df_filtered_simplified)

