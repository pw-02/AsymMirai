# import pandas as pd
# import torch
# from torch.utils.data import DataLoader
# from mirai_metadataset import MiraiMetadataset
# from embed_explore import resize_and_normalize
# import os

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
from asymmetry_model.mirai_metadatasets3 import MiraiMetadatasetS3
from asymmetry_model.embed_explore import resize_and_normalize

print("Working directory:", os.getcwd())


print("Working directory:", os.getcwd())

use_crop = False
multiple_pairs_per_exam = False
device = 0
torch.cuda.set_device(device)

# model = torch.load('./training_preds/full_model_partial_epoch_15_4_26_ablation_flex_width_5_matrix_learned_dist.pt',
#                     map_location = torch.device(f'cuda:{device}'))

model = torch.load('snapshots/trained_asymmirai.pt',
                   weights_only=False,
                    map_location = torch.device(f'cuda:{device}'))

batch_size = 1

val_file_name = 'data/embed/asymirai_input/EMBED_OpenData_metadata_screening_2D_complete_exams_CLEANED_4VIEW_test.csv'
val_df = pd.read_csv(val_file_name)
val_dataset = MiraiMetadatasetS3(val_df, resizer=resize_and_normalize,
                                s3_bucket='embdedpng',
                                mode='val', align_images=False,
                                multiple_pairs_per_exam=multiple_pairs_per_exam)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=min(0, batch_size))

with torch.no_grad():
    predictions_for_epoch_pos = []
    predictions_for_epoch_neg = []
    eids_for_epoch = []
    y_argmins_mlo_for_epoch = []
    x_argmins_mlo_for_epoch = []
    y_argmins_cc_for_epoch = []
    x_argmins_cc_for_epoch = []
    correct_count = 0
    num_samples = 0

    for index, sample in enumerate(val_dataloader):
        if multiple_pairs_per_exam:
            eid, label, exam_list = sample
            label = label.cuda()
            output, _ = model(None, None, None, None, exam_list=exam_list)

        else:
            eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
            l_cc_img, r_cc_img, l_mlo_img, r_mlo_img = l_cc_img.cuda(), r_cc_img.cuda(), l_mlo_img.cuda(), r_mlo_img.cuda()
            label = label.cuda()

            output, other = model(l_cc_img, r_cc_img, l_mlo_img, r_mlo_img)
        preds = torch.argmax(output, dim=1)

        predictions_for_epoch_neg = predictions_for_epoch_neg + list(output[:, 0].cpu().detach().numpy())
        predictions_for_epoch_pos = predictions_for_epoch_pos + list(output[:, 1].cpu().detach().numpy())
        eids_for_epoch = eids_for_epoch + list(eid)
        y_argmins_cc_for_epoch = y_argmins_cc_for_epoch + list(other[0]['y_argmin'])#.cpu().detach().numpy())
        x_argmins_cc_for_epoch = x_argmins_cc_for_epoch + list(other[0]['x_argmin'])#.cpu().detach().numpy())
        y_argmins_mlo_for_epoch = y_argmins_mlo_for_epoch + list(other[1]['y_argmin'])#.cpu().detach().numpy())
        x_argmins_mlo_for_epoch = x_argmins_mlo_for_epoch + list(other[1]['x_argmin'])#.cpu().detach().numpy())

        correct_count += label[preds == label].shape[0]
        num_samples += label.shape[0]

        cur_preds = pd.DataFrame()
        cur_preds['exam_id'] = eids_for_epoch
        cur_preds['prediction_neg'] = predictions_for_epoch_neg
        cur_preds['prediction_pos'] = predictions_for_epoch_pos
        cur_preds['y_argmin_cc'] = y_argmins_cc_for_epoch
        cur_preds['x_argmin_cc'] = x_argmins_cc_for_epoch
        cur_preds['y_argmin_mlo'] = y_argmins_mlo_for_epoch
        cur_preds['x_argmin_mlo'] = x_argmins_mlo_for_epoch

        if index % 5 == 0:
            cur_preds.to_csv(f'validation_predictions1.csv', index=False)

    cur_preds = pd.DataFrame()
    cur_preds['exam_id'] = eids_for_epoch
    cur_preds['prediction_neg'] = predictions_for_epoch_neg
    cur_preds['prediction_pos'] = predictions_for_epoch_pos
    cur_preds['y_argmin_cc'] = y_argmins_cc_for_epoch
    cur_preds['x_argmin_cc'] = x_argmins_cc_for_epoch
    cur_preds['y_argmin_mlo'] = y_argmins_mlo_for_epoch
    cur_preds['x_argmin_mlo'] = x_argmins_mlo_for_epoch
    cur_preds.to_csv(f'validation_predictions2.csv', index=False)