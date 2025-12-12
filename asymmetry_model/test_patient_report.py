
import pandas as pd
import numpy as np
import re
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import os
import imageio
import aiofiles
import asyncio
import re
import cv2
import logging
import urllib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from IPython.core import display
from ast import literal_eval
from pathlib import Path
import time
import pickle

from random import sample

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

def align_images_given_img(left_data, right_data):
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(left_data,np.amin(left_data)+1e-5,np.amax(left_data),0)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    ret,thresh = cv2.threshold(right_data,np.amin(right_data)+1e-5,np.amax(right_data),0)

    # calculate moments of binary image
    new_M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    new_cX = int(new_M["m10"] / new_M["m00"])
    new_cY = int(new_M["m01"] / new_M["m00"])

    num_rows, num_cols = right_data.shape[:2]   

    translation_matrix = np.float32([ [1,0,cX-new_cX], [0,1,cY-new_cY] ])   
    right_data = cv2.warpAffine(right_data, translation_matrix, (num_cols, num_rows))
    return left_data, right_data


def crop(img):
    nonzero_inds = torch.nonzero(img - torch.min(img))
    top = torch.min(nonzero_inds[:, 0])
    left = torch.min(nonzero_inds[:, 1])
    bottom = torch.max(nonzero_inds[:, 0])
    right = torch.max(nonzero_inds[:, 1])

    return img[top:bottom, left:right]


def resize_and_normalize(img, use_crop=False):
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)
    dummy_batch_dim = False

    # Adding a dummy batch dimension if necessary
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
        dummy_batch_dim = True

    with torch.no_grad():
        img = torch.tensor((img - img_mean)/img_std)
        if use_crop:
            img = crop(img)
        img = img.expand(1, 3, *img.shape)\
                        .type(torch.FloatTensor)
        img_resized = F.upsample(img, size=(target_size[0], target_size[1]), mode='bilinear')
    #img_resized = img

    if dummy_batch_dim:
        return img_resized[0]
    else:
        return img_resized[0]




def run_asym_with_heatmap(
        exam_id,
        original_df,
        model='./asymmetry_model/training_preds/full_model_epoch_21_3_11_corrected_flex.pt',
        device=0,
        target_size=(1664, 2048),
        use_crop=True,
        pooling_size=None):
    
    # # Loading our trained model onto the given device
    # if type(model) is str:
    #     torch.cuda.set_device(device)
    #     model = torch.load(model, map_location = device)
    
    if pooling_size is not None:
        model.latent_h = pooling_size[0]
        model.latent_w = pooling_size[1]
        
    cur_exam = original_df[original_df['exam_id'] == exam_id]
    
    # Loading in each image for this exam
    imgs = []
    for view in ['CC', 'MLO']:
        for side in ['L', 'R']:
            cur_path = cur_exam[(cur_exam['view'] == view) & 
                                (cur_exam['laterality'] == side)]['file_path'].values[-1]
            imgs.append(resize_and_normalize(cv2.imread(cur_path, cv2.IMREAD_UNCHANGED), use_crop=use_crop).unsqueeze(0))

    start = time.perf_counter()

    prediction, other = model(*tuple(imgs))

    
    res = {}
    cc_heatmap = (other[0]['heatmap'] - model.initial_asym_mean) / (2 * model.initial_asym_std)
    cc_heatmap = torch.sigmoid(cc_heatmap)
    res['cc_heatmap'] = cc_heatmap.detach().cpu()
    
    mlo_heatmap = (other[1]['heatmap'] - model.initial_asym_mean) / (2 * model.initial_asym_std)
    mlo_heatmap = torch.sigmoid(mlo_heatmap)
    res['mlo_heatmap'] = mlo_heatmap.detach().cpu()

    res['risk_score']  = prediction[0, 1]
    return res

def boxwise_upsample(array, target_size, pool_size=(5,5), mode="average"):
    """
    Upsample a tensor to target_size using overlapping
    boxes; take the max 
    array is the lower resolution activation map
    target_size is the ultimate target shape
    """
    res = torch.zeros(*target_size)
    div_matrix = torch.zeros(*target_size)
    og_h = array.shape[0]
    og_w = array.shape[1]
    
    b_h = target_size[0] / (og_h)
    b_w = target_size[1] / (og_w)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            top = int(round(i * b_h))
            bot = int(round((i + (og_h / pool_size[0])) * (b_h)))
            left = int(round(j * b_w))
            right = int(round((j + (og_w / pool_size[1])) * (b_w)))
            
            cur_chunk = res[top:bot, left:right]
            
            tmp = np.zeros_like(cur_chunk)
            tmp[:, :] = array[i, j]
            
            if mode == "average":
                # If we are averaging, we're gonna keep track of how many
                # overlapping boxes we average over with div_matrix
                div_matrix[top:bot, left:right] = div_matrix[top:bot, left:right] + 1
                res[top:bot, left:right] = res[top:bot, left:right] + torch.tensor(tmp)
            else:
                maximum = np.maximum(cur_chunk, tmp)

                res[top:bot, left:right] = maximum
    if mode == "average":
        res = res / div_matrix
    return res

def overlay_heatmap_on_image(img, heatmap, pooling_size=(5,5)):
    height, width = img.shape[0], img.shape[1]
    upsampled_heatmap = boxwise_upsample(heatmap, (height, width), pool_size=pooling_size)
    
    final_heatmap = cv2.applyColorMap(np.uint8(255*upsampled_heatmap), cv2.COLORMAP_JET)
    final_heatmap = np.float32(final_heatmap) / 255
    final_heatmap = final_heatmap[...,::-1]
    
    img_rescaled = img - np.amin(img)
    img_rescaled = img_rescaled / np.amax(img_rescaled)
    img_data_rgb = np.repeat(img_rescaled[:, :, np.newaxis], 3, axis=2)
    
    overlayed_img = 0.5 * img_data_rgb + 0.3 * final_heatmap
    return overlayed_img

def highlight_asym(original_data, target_exam, axs=None,
                   align=False, flex=True, overlay_heatmap=False, 
                   model=None, use_crop=False, draw_box=True, pooling_size = (5, 5),
                  roi_info=None, model_input_size=(1664, 2048), latent_size=(52, 64),
                  asymetry_distribution=None):
    latent_h, latent_w = latent_size[0], latent_size[1]
    pool_h = latent_h // pooling_size[0]
    pool_w = latent_w // pooling_size[1]
    
    eid = target_exam

    # Creating the layout for our plot
    if axs is not None:
        ax = axs
        fig = None
    else:
        fig, ax = plt.subplots(2,2, figsize=(8,10))
        [axi.set_axis_off() for axi in ax.ravel()]

    if hasattr(model, "topk_for_heatmap") and model.topk_for_heatmap is not None:
        topk_count = model.topk_for_heatmap
    else:
        topk_count = 1
    # If we are adding heatmaps, compute them
    if model is not None:
        asym_results = run_asym_with_heatmap(target_exam, original_data, target_size=model_input_size, 
                                        model=model, use_crop=use_crop, pooling_size=pooling_size)
    else:
        asym_results = run_asym_with_heatmap(target_exam, original_data, 
                                        target_size=model_input_size, use_crop=use_crop, pooling_size=pooling_size)
    
    # Iterate over the two standard views
    start = time.perf_counter()
    for i, view in enumerate(['MLO', 'CC']):
        for j, side in enumerate(['L', 'R']):
            # Grab the indices we'll use to draw the max box

            ax[i,j].set_title(side + ' ' + view)
            
            if view == 'MLO':
                #max_by_ftr, x_argmin = torch.max(asym_results[f'mlo_heatmap'], dim=-1)
                max_by_ftr, argmax_inds = torch.topk(asym_results[f'mlo_heatmap'].view(-1), topk_count, dim=-1)
                x_argmax = argmax_inds % asym_results[f'mlo_heatmap'].shape[-1]
                y_argmax = argmax_inds // asym_results[f'mlo_heatmap'].shape[-1]
            else:
                max_by_ftr, argmax_inds = torch.topk(asym_results[f'cc_heatmap'].view(-1), topk_count, dim=-1)
                x_argmax = argmax_inds % asym_results[f'cc_heatmap'].shape[-1]
                y_argmax = argmax_inds // asym_results[f'cc_heatmap'].shape[-1]

            asym_score = asym_results['risk_score']
            
            if asymetry_distribution is None:
                ax[i,j].text(x=0, y=-.06, s=f"Asym Score [0-1]: {round(asym_score.item(),3)}", transform=ax[i,j].transAxes)
            else:
                quantile = asymetry_distribution.quantile(asym_score.item())
                ax[i,j].text(x=0, y=-.06, s=f"Asym Quantile: {round(quantile,3)} (Score of {round(asym_score.item(),3)}", transform=ax[i,j].transAxes)
            #max_by_ftr, y_argmin = torch.max(max_by_ftr, dim=-1)
            #for topk_ind in range(topk_count):
            #    y_max_by_ftr, y_argmin = torch.topk(max_by_ftr.view(-1), topk_count, dim=-1)
            
            # Convert these values to fractional locations in [0, 1]
            if view == 'MLO':
                tuple_list = []
                for topk_ind in range(topk_count):
                    y_loc_frac = y_argmax[topk_ind].item() / asym_results[f'mlo_heatmap'].shape[-2]
                    x_loc_frac = x_argmax[topk_ind].item() / asym_results[f'mlo_heatmap'].shape[-1]
                    # logger.debug(f"True max (MLO) ---- {torch.max(asym_results[f'mlo_heatmap'])}", )
                    # logger.debug(f"Max found using indices ---- {asym_results[f'mlo_heatmap'][0, y_argmax[topk_ind].item(), x_argmax[topk_ind].item()]}", )
                    tuple_list.append((x_loc_frac, y_loc_frac))
                    #y_loc_frac = y_argmin / asym_results[f'mlo_heatmap'].shape[-2]
                    #x_loc_frac = x_argmin[0, y_argmin].item() / asym_results[f'mlo_heatmap'].shape[-1]
            else:
                tuple_list = []
                for topk_ind in range(topk_count):
                    y_loc_frac = y_argmax[topk_ind].item() / asym_results[f'cc_heatmap'].shape[-2]
                    x_loc_frac = x_argmax[topk_ind].item() / asym_results[f'cc_heatmap'].shape[-1]
                    # logger.debug(f"True max (CC) ---- {torch.max(asym_results[f'cc_heatmap'])}")
                    # logger.debug(f"Max found using indices ---- {asym_results[f'cc_heatmap'][0, y_argmax[topk_ind].item(), x_argmax[topk_ind].item()]}", )
                    tuple_list.append((x_loc_frac, y_loc_frac))
            
            # Grab the file path for each side for the current view
            img = original_data[(original_data['exam_id'] == eid) 
                                & (original_data['view'] == view)
                                & (original_data['laterality'] == side)]['file_path'].values[-1]

            # Read in our image, crop and align if needed
            img_data = imageio.imread(img)
            if use_crop:
                img_data = crop(torch.tensor(img_data / 1)).numpy()
            if align and side == 'R':
                l_img = original_data[(original_data['exam_id'] == eid) 
                                & (original_data['view'] == view)
                                & (original_data['laterality'] == 'L')]['file_path'].values[-1]
                l_img_data = imageio.imread(l_img)
                if use_crop:
                    l_img_data = crop(torch.tensor(l_img_data / 1)).numpy()
                _, img_data = align_images_given_img(l_img_data, img_data)
        
            # Overlay our heatmap onto the image and display it if desired;
            # otherwise just display the image
            if overlay_heatmap:
                if view == 'MLO':
                    heatmap = asym_results[f'mlo_heatmap'].numpy()[0]
                else:
                    heatmap = asym_results[f'cc_heatmap'].numpy()[0]
                overlayed_img = overlay_heatmap_on_image(img_data, heatmap, pooling_size=pooling_size)
                img_with_heatmap = ax[i, j].imshow(overlayed_img, interpolation='nearest', )
                plt.colorbar(mpl.cm.ScalarMappable(cmap=plt.cm.get_cmap("jet")), ax=ax[i,j], pad=.01)
            else:
                img_data_rescaled = img_data - np.amin(img_data)
                img_data_rescaled = img_data_rescaled / np.amax(img_data_rescaled)
                img_data_rgb = np.repeat(img_data_rescaled[:, :, np.newaxis], 3, axis=2)
                
                ax[i, j].imshow(img_data_rgb, cmap='gray', interpolation='nearest', )
            
            height = img_data.shape[0]
            width = img_data.shape[1]
            
            for topk_ind in range(topk_count):
                x_loc_frac = tuple_list[topk_ind][0]
                y_loc_frac = tuple_list[topk_ind][1]
                weight = model.topk_weights[-topk_ind].item()
                if view == 'MLO':
                    heatmap = asym_results[f'mlo_heatmap'].numpy()[0]
                else:
                    heatmap = asym_results[f'cc_heatmap'].numpy()[0]
                y_loc = int(round(y_loc_frac * height))
                x_loc = int(round(x_loc_frac * width))

                rect_width = int(width / pooling_size[1])
                rect_height = int(height / pooling_size[0])

                # Add the region of asymmetry actually used by our model
                if draw_box and flex:
                    rect = Rectangle((x_loc, y_loc), 
                                            rect_width, rect_height,
                                            linewidth=1, edgecolor=(1,0,0,weight), facecolor='none')
                    # Add the patch to the Axes
                    ax[i, j].add_patch(rect)
                elif draw_box:
                    rect = Rectangle((x_loc * (width / latent_w), y_loc * (height / latent_h)), 
                                            pool_w * (width // latent_w), pool_h * (height // latent_h), 
                                            linewidth=1, edgecolor=(1,0,0,weight), facecolor='none')
                    # Add the patch to the Axes
                    ax[i, j].add_patch(rect)

                if roi_info is not None and roi_info[4] == side and roi_info[5] == view:
                    tl_y, tl_x, br_y, br_x = roi_info[0], roi_info[1], roi_info[2], roi_info[3]
                    rect = Rectangle((tl_x, tl_y), 
                                    br_x - tl_x, br_y - tl_y, 
                                    linewidth=1, edgecolor='g', facecolor='none')
                    # Add the patch to the Axes
                    ax[i, j].add_patch(rect)

    # logger.debug('Heatmap rendered for %s in %s', eid, time.perf_counter()-start)

    if fig is not None:
        fig.show()
        fig.savefig(f'./visualization_{target_exam}.png', dpi=300)
        return f'./visualization_{target_exam}.png'
    

if __name__ == "__main__":

    # Render the figure

    highlight_asym(
        original_data=pd.read_csv('data/embed/asymirai_input/postive_neg_example.csv'),
        target_exam='1960000000000000',
        axs=None,
        model='snapshots/trained_asymmirai.pt',
        use_crop=True,
        pooling_size=(5,5),
        align=True,
        overlay_heatmap=True,
        draw_box=True,
        flex=True,
        model_input_size=(1664, 2048),
        latent_size=(52, 64),
        asymetry_distribution=None
    )

    #  model = torch.load(
    #     'snapshots/trained_asymmirai.pt',
    #     weights_only=False,
    #     map_location=torch.device(f'cuda:{device}')
    # )