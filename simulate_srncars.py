# import stuff
import os
import cv2
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print('Running on', DEVICE)


def get_model():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu',weights_only=True))
    model = model.to(DEVICE).eval()

    return model


def save_1chan_image(np_array, label):
    np_array = np_array - np_array.min()
    np_array = np_array / np_array.max()
    np_array = (np_array * 255).astype(np.uint8)
    image = np.repeat(np_array[:, :, np.newaxis], 3, axis=2) # HxWx3 depth image in numpy
    cv2.imwrite('%s_%s.png'%(fname_header,label), image)


def infer_sonar_img(input_path, model):

    # load raw image
    raw_img = cv2.imread(input_path)
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy

    # create range-angle image
    range_bins = raw_img.shape[0]
    angle_bins = raw_img.shape[1]

    adjusted_depth = depth - depth.min()
    adjusted_depth = adjusted_depth / adjusted_depth.max()
    adjusted_depth = 1 - adjusted_depth

    range_bin_edges = np.linspace(adjusted_depth.min(), adjusted_depth.max(), range_bins + 1)
    angle_bin_edges = np.linspace(0,adjusted_depth.shape[1], angle_bins + 1)

    histograms = []
    for angle in range(angle_bins):
        min_col = int(angle_bin_edges[angle])
        max_col = int(angle_bin_edges[angle + 1])
        image_slice = adjusted_depth[:, min_col:max_col]
        flattened_slice = image_slice.flatten()
        range_hist, _ = np.histogram(flattened_slice, bins=range_bin_edges)
        histograms.append(range_hist.reshape(-1, 1)) # (range_bins, 1)

    range_angle_image = np.concatenate(histograms, axis=1) # (range_bins, angle_bins)

    return range_angle_image


def infer_sonar_on_srn_obj(obj_path, model):
    # make 'sonar' folder
    os.makedir(os.path.join(obj_path, 'sonar'))

    # loop through all images in the rgb folder
    rgb_fnames = os.listdir(os.path.join(obj_path, 'rgb'))
    for fname in rgb_fnames:
        input_path = os.path.join(obj_path, 'rgb', fname)
        
        # calculate the sonar image
        sonar_image = infer_sonar_img(input_path, model)

        # repeat over 3 channels
        sonar_image = np.repeat(sonar_image[:, :, np.newaxis], 3, axis=2)

        # save the sonar image
        cv2.imwrite(os.path.join(obj_path, 'sonar', fname), sonar_image)


def infer_sonar_on_srn_dataset(dataset_path, model):
    # loop through all objects in the dataset
    obj_fnames = os.listdir(dataset_path)
    for fname in obj_fnames:
        obj_path = os.path.join(dataset_path, fname)
        infer_sonar_on_srn_obj(obj_path, model)
    

if __name__ == '__main__':
    model = get_model()
    dataset_paths = [os.path.join('/home/berian/Documents/shapenet', s) for s in ['cars_train', 'cars_val', 'cars_test']]
    for dataset_path in dataset_paths:
        infer_sonar_on_srn_dataset(dataset_path, model)
