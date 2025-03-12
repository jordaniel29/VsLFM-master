import numpy as np
import math
import torch
from torch.utils.data.dataset import Dataset

# Patching assumption: 
# 1. Image dimension is same (height and width) 
class Patch(Dataset):
    def __init__(self, lr_patches, hr_patches):
        self.lr_data = lr_patches
        self.hr_data = hr_patches

    def __getitem__(self, idx):
        image_lr = self.lr_data[idx, :, :, :]
        image_hr = self.hr_data[idx, :, :, :]

        return (image_lr, image_hr)

    def __len__(self):
        return self.lr_data.shape[0]

def patching_training(LR_image, HR_image, patch_size=51, batch_size=2, scale=3):
    stride = patch_size
    lr_patches = LR_image.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    hr_patches = HR_image.unfold(2, patch_size*scale, stride*scale).unfold(3, patch_size*scale, stride*scale)
    lr_patches = lr_patches.permute(0, 2, 3, 1, 4, 5)
    lr_patches = lr_patches.contiguous().view(-1, LR_image.shape[1], patch_size, patch_size)
    hr_patches = hr_patches.permute(0, 2, 3, 1, 4, 5)
    hr_patches = hr_patches.contiguous().view(-1, HR_image.shape[1], patch_size*scale, patch_size*scale)

    patches = Patch(lr_patches, hr_patches)
    patch_loader = torch.utils.data.DataLoader(dataset=patches, batch_size=batch_size, shuffle=False, num_workers=6)

    return patch_loader

"""
    Prerequisite: the result of overlap is of round value and not decimal value
    Equation:
    1. grid_number = image_size // patch_size 
    2. grid_number*patch_size - (grid_number-1)*(overlap) = image_size
"""
def calculate_overlap(image_size, patch_size):
    grid_number = 0
    if image_size % patch_size == 0:
        grid_number = image_size // patch_size
    else:
        grid_number = (image_size // patch_size) + 1

    overlap = (grid_number*patch_size - image_size) / (grid_number-1)
    if (overlap % 1) != 0:
        raise ValueError("Cannot patch the image using this patch number!")

    return int(overlap)

"""
    Crop the LR image into small patches of the specified patch_size and overlap.
""" 
def patching(LR_image, patch_size, overlap):
    H, W = LR_image.shape[1], LR_image.shape[2]
    stride = patch_size-overlap
    patches = []
    

    for h in range(0, H - patch_size + 1, stride):
        for w in range(0, W - patch_size + 1, stride):
            temp_patch = LR_image[:, h:h+patch_size, w:w+patch_size]
            patches.append(temp_patch)

    patches = torch.stack(patches, 0)  # Shape: [num_patches, channels, patch_size, patch_size]
    
    print()
    print("LR:", LR_image.shape)
    print("overlap:", overlap)
    print(patches.shape)
    exit()

    return patches

"""
    Merge the patches back into a full image using the specified overlap for seamless stitching.
    Assumption: 
    1. grid dimension is square (e.g. 3x3 or 6x6, not 3x4)
    2. height & width of the image is same
"""

def depatching(patches, upscaled_overlap):
    Nbx = int(math.sqrt(patches.shape[0]))  # Number of patches along x-axis
    Nby = int(math.sqrt(patches.shape[0]))  # Number of patches along y-axis
    patch_size = patches.shape[2]  # Width/height of each patch
    stride = patch_size - upscaled_overlap if upscaled_overlap > 0 else patch_size  # Adjust stride

    # Handling weight matrix
    if upscaled_overlap > 0:
        a = -0.5
        x = np.arange(1, upscaled_overlap + 1)
        y = 1 / (1 + np.exp(a * (x - 14)))  # Sigmoid function for blending
        z = np.concatenate((y, max(y) * np.ones((patch_size - 2 * upscaled_overlap)), y[::-1])) + 0.001
        z = z.reshape((-1, 1))
        W = (z.T * z)
    else:
        W = np.ones((patch_size, patch_size))  # No weighting needed if no overlap

    W = np.expand_dims(W, 0).repeat(patches.shape[1], axis=0)  # Expand for channels

    # Initialize the result image and weight accumulator
    img_shape = (patches.shape[1], 
                 patch_size * Nbx - upscaled_overlap * (Nbx - 1), 
                 patch_size * Nby - upscaled_overlap * (Nby - 1))
    
    img = np.zeros(img_shape)
    W_f = np.zeros(img_shape)

    # Stitching the patches
    for u in range(Nbx):
        for v in range(Nby):
            x_begin = u * stride
            x_end = x_begin + patch_size
            y_begin = v * stride
            y_end = y_begin + patch_size

            img[:, x_begin:x_end, y_begin:y_end] += W * patches[u * Nbx + v, :, :, :]
            W_f[:, x_begin:x_end, y_begin:y_end] += W

    # Normalize by dividing by the weight accumulator to remove overlap
    img = np.divide(img, W_f, where=W_f > 0)  # Avoid division by zero
    return img