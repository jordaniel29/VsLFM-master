import os
import tifffile
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Define input folders
folder1 = "Output/VSLFM_ps51/HR"  
folder2 = "Output/VSLFM/SR"  

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
    folder1_path = os.path.join(folder1, filename)
    folder2_path = os.path.join(folder2, filename)

    # Load TIFF files
    with tifffile.TiffFile(folder1_path) as folder1_tif, tifffile.TiffFile(folder2_path) as folder2_tif:
        folder1_images = np.array([page.asarray() for page in folder1_tif.pages])  # Shape: (169, 459, 459)
        folder2_images = np.array([page.asarray() for page in folder2_tif.pages])  # Shape: (169, 459, 459)

    # Compute PSNR & SSIM for each slice
    psnr_per_slice = [psnr(folder1_images[i], folder2_images[i], data_range=folder2_images[i].max() - folder2_images[i].min())
                      for i in range(folder1_images.shape[0])]
    
    ssim_per_slice = [ssim(folder1_images[i], folder2_images[i], data_range=folder2_images[i].max() - folder2_images[i].min())
                      for i in range(folder1_images.shape[0])]

    # Compute average for the entire stack
    avg_psnr = np.mean(psnr_per_slice)
    psnr_values.append(avg_psnr)
    avg_ssim = np.mean(ssim_per_slice)
    ssim_values.append(avg_ssim)

# Compute overall averages
overall_psnr = np.mean(psnr_values)
overall_ssim = np.mean(ssim_values)

print(f"\nOverall Average PSNR: {overall_psnr:.2f} dB")
print(f"Overall Average SSIM: {overall_ssim:.4f}")