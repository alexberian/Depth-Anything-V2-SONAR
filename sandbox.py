import cv2
import torch
import numpy as np
import sys

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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

input_path = 'figures/gape.png'
fname_header = input_path.split('.')[0]

def save_1chan_image(np_array, label):
    np_array = np_array - np_array.min()
    np_array = np_array / np_array.max()
    np_array = (np_array * 255).astype(np.uint8)
    image = np.repeat(np_array[:, :, np.newaxis], 3, axis=2) # HxWx3 depth image in numpy
    cv2.imwrite('%s_%s.png'%(fname_header,label), image)

raw_img = cv2.imread(input_path)
print('raw_img shape: ', raw_img.shape)
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

print('Minimum depth: ', depth.min())
print('Maximum depth: ', depth.max())

# save depth map as image
save_1chan_image(depth, 'depth')

# create range-angle image
range_bins = 100
angle_bins = raw_img.shape[1] // 2

adjusted_depth = depth - depth.min()
adjusted_depth = adjusted_depth / adjusted_depth.max()
adjusted_depth = 1 - adjusted_depth
save_1chan_image(adjusted_depth, 'adjusted_depth')

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

# save range-angle image as image
save_1chan_image(range_angle_image[:-5], 'range-angle')


