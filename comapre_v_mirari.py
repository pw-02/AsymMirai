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
from asymmetry_model.embed_explore import resize_and_normalize, crop
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


def find_missing_predictions(asym_preds, mirai_preds):
    asym_exam_ids = set(asym_preds['exam_id'].unique())
    mirai_exam_ids = set(mirai_preds['exam_id'].unique())
    missing_in_asym = mirai_exam_ids - asym_exam_ids
    missing_in_mirai = asym_exam_ids - mirai_exam_ids
    print("Missing in Asym:", missing_in_asym)
    print("Missing in Mirai:", missing_in_mirai)
    return missing_in_asym, missing_in_mirai


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
        return included, label

    probs_for_eval, golds_for_eval = [], []
    for prob_arr, censor_time, gold in zip(probs, censor_times, golds):
        include, label = include_exam_and_determine_label(censor_time, gold)
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
    merged_df_filtered_copy['prediction_pos']


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
            if not type(v) is float:
                probs_asymmirai[i] = v[1]
        print(len(probs_asymmirai))
        df = pd.DataFrame({
            'AsymMirai': probs_asymmirai,
            'Mirai': probs_asymmirai
        })
        roc = pyroc.ROC(labels,
                        df)
        
        fpr, tpr, thresh = sklearn.metrics.roc_curve(labels, probs_asymmirai)
        matplotlib.rc('xtick', labelsize=19) 
        matplotlib.rc('ytick', labelsize=19) 
        plt.rcParams.update({'axes.labelsize': 40})
        
        plt.plot(fpr, tpr)
        auc = roc.auc[0, 1]
        legend.append("Year {:.0f} AUC: {:.2f} ({:.2f}, {:.2f})".format(
            year+1, auc, roc.ci(0.05)[:, 1][0], roc.ci(0.05)[:, 1][1])
        )
        print(f'AsymMirai year {year + 1} 95% CI: \t', roc.ci(0.05)[:, 0])
        #print(f'Mirai year {year + 1} 95% CI: \t\t', roc.ci(0.05)[:, 1])
    plt.legend(legend, prop={'size': 20})
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


if __name__ == "__main__":

    asym_input_file = 'data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW_test.csv'
    asym_preds_file = 'tmp_val_run.csv'
    filtered = pd.read_csv(asym_input_file)
    asym_preds = pd.read_csv(asym_preds_file)
    merged_df_filtered = filtered.merge(asym_preds, on='exam_id', suffixes=['', '_asym'])

    craete_asum_auc_plot(merged_df_filtered)

    #save merged_df_filtered to csv
    # merged_df_filtered.to_csv('merged_df_filtered.csv', index=False)


    # print(filtered['patient_id'].unique().shape)
    # print(filtered['exam_id'].unique().shape)




# if __name__ == "__main__":

#     asym_input_file = 'data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW_test.csv'
#     mirai_input_file = 'data/mirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW.csv'
#     asym_preds_file = 'asym_validation_predictions1.csv'
#     mirai_preds_file = 'validation_predictions_mirai.csv'

#     # filtered = construct_dataset(asym_input_file, mirai_input_file)
#     filtered = pd.read_csv(asym_input_file)

#     print(filtered['patient_id'].unique().shape)
#     print(filtered['exam_id'].unique().shape)
#     # If we're missing any, run AsymMirai and grab those results
#     asym_preds = pd.read_csv(asym_preds_file)


    
#     missing_asym_exams = filtered[~filtered['exam_id'].isin(asym_preds['exam_id'])]['exam_id'].unique()
#     if missing_asym_exams.shape[0] > 0:

#         #remove missing exams from filtered
#         filtered = filtered[~filtered['exam_id'].isin(missing_asym_exams)]



#         # model = torch.load('snapshots/trained_asymmirai.pt',
#         #                 weights_only=False,
#         #                 map_location = torch.device(f'cpu:0'))
#         # val_df = filtered
#         # print(val_df['exam_id'].unique().shape)
#         # val_df = val_df[val_df['exam_id'].isin(missing_asym_exams)]
#         # print(val_df['exam_id'].unique().shape)
#         # missing_preds = run_validation(model, val_df)
#         # asym_preds = pd.concat([asym_preds, missing_preds])

#     # # 3) Grab Mirai predictions, and figure out which (if any) are missing
#     # mirai_preds = pd.read_csv(mirai_preds_file, header=None)
#     # for i in range(5):
#     #     mirai_preds['year_{}_risk'.format(i+1)] = mirai_preds[4+i]
#     # def get_exam_id(row):
#     #     file_path = row[0]
#     #     return filtered[filtered['file_path'] == file_path]['exam_id'].values[0]
#     # mirai_preds['exam_id'] = mirai_preds.apply(get_exam_id, axis=1)
#     # missing_mirai_exams = filtered[~filtered['exam_id'].isin(mirai_preds['exam_id'])]['exam_id'].unique()
#     # # If we're missing any, run AsymMirai and grab those results
#     # if missing_mirai_exams.shape[0] > 0:
#     #     val_df = filtered
#     #     print(val_df['exam_id'].unique().shape)
#     #     val_df = val_df[val_df['exam_id'].isin(missing_mirai_exams)]
#     #     val_df['split_group'] = 'test'
#     #     val_df.loc[:, 'years_to_last_followup'] = 100
#     #     print(val_df['exam_id'].unique().shape)
        
#     #     val_df[['exam_id','patient_id','laterality','view',
#     #         'file_path','years_to_cancer','years_to_last_followup',
#     #             'split_group']].to_csv('./tmp_val_input_for_mirai_2.csv', index=False)
        
#     merged_df_filtered = filtered.merge(asym_preds, on='exam_id', suffixes=['', '_asym'])
#     merged_df_filtered['exam_id'].unique().shape
#     res = merged_df_filtered.apply(get_label, args=(10, 'censor_time'), axis=1)
#     #save res to file
#     res.to_csv('tmp_censor_time.csv', index=False)
#     merged_df_filtered['censor_time'] = res
#     res = merged_df_filtered.apply(get_label, args=(10, 'any_cancer'), axis=1)
#     merged_df_filtered['any_cancer'] = res
#     merged_df_filtered_simplified = merged_df_filtered.drop_duplicates(['exam_id'])
#     merged_df_filtered_copy = merged_df_filtered.copy()
#     merged_df_filtered_copy.to_csv('tmp_merged_df_filtered_copy.csv', index=False)
#     merged_df_filtered_copy = pd.read_csv('tmp_merged_df_filtered_copy.csv')
#     merged_df_filtered_copy['prediction_pos']

#     legend = []
#     for year in range(5):
#         # probs_mirai, labels = get_arrs_for_auc(merged_df_filtered_simplified[f'year_{year+1}_risk'],
#         #                                 merged_df_filtered_simplified['censor_time'],
#         #                                 merged_df_filtered_simplified['any_cancer'],
#         #                                 year)
#         probs_asymmirai, labels = get_arrs_for_auc(merged_df_filtered_simplified[f'prediction_pos'],
#                                         merged_df_filtered_simplified['censor_time'],
#                                         merged_df_filtered_simplified['any_cancer'],
#                                         year)
#         for i, v in enumerate(probs_asymmirai):
#             if not type(v) is float:
#                 probs_asymmirai[i] = v[1]
#         print(len(probs_asymmirai))
#         df = pd.DataFrame({
#             'AsymMirai': probs_asymmirai,
#             # 'Mirai': probs_mirai
#         })
#         roc = pyroc.ROC(labels,
#                         df)
        
#         fpr, tpr, thresh = sklearn.metrics.roc_curve(labels, probs_asymmirai)
#         matplotlib.rc('xtick', labelsize=19) 
#         matplotlib.rc('ytick', labelsize=19) 
#         plt.rcParams.update({'axes.labelsize': 40})
        
#         plt.plot(fpr, tpr)
#         auc = roc.auc[0, 1]
#         legend.append("Year {:.0f} AUC: {:.2f} ({:.2f}, {:.2f})".format(
#             year+1, auc, roc.ci(0.05)[:, 1][0], roc.ci(0.05)[:, 1][1])
#         )
#         print(f'AsymMirai year {year + 1} 95% CI: \t', roc.ci(0.05)[:, 0])
#         #print(f'Mirai year {year
#         plt.legend(legend, prop={'size': 20})
#         plt.xlabel("False Positive Rate")
#         plt.ylabel("True Positive Rate")
#         plt.show()







