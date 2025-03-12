import os
import tifffile
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Define input folders
folder1 = "Output/VSLFM_ps51/Reconstruction/HR"  
folder2 = "Output/VSLFM_ps51/Reconstruction/SR_paper"  

# Get list of TIFF files
folder1_files = sorted([f for f in os.listdir(folder1) if f.lower().endswith(".tif")])
folder2_files = sorted([f for f in os.listdir(folder2) if f.lower().endswith(".tif")])

# Ensure the file lists match
if folder1_files != folder2_files:
    raise ValueError("Mismatch between Folder 1 and Folder 2 files!")

# Iterate over matching files and compute PSNR and SSIM
psnr_values = []
ssim_values = []

for filename in tqdm(folder1_files):
    folder1_image = tifffile.imread(os.path.join(folder1, filename))
    folder2_image = tifffile.imread(os.path.join(folder2, filename))

    # Ensure both images are of the same shape
    if folder1_image.shape != folder2_image.shape:
        raise ValueError("The images must have the same dimensions.")

    # Compute PSNR & SSIM for each slice
    psnr_values.append(psnr(folder1_image, folder2_image, data_range=folder2_image.max() - folder2_image.min()))
    ssim_values.append(ssim(folder1_image, folder2_image, data_range=folder2_image.max() - folder2_image.min(), multichannel=False))

# Compute overall averages
overall_psnr = np.mean(psnr_values)
overall_ssim = np.mean(ssim_values)

print(f"\nOverall Average PSNR: {overall_psnr:.2f} dB")
print(f"Overall Average SSIM: {overall_ssim:.4f}")