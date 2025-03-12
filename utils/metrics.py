from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np


def ssim(img_true, img_test):
    '''
    This function input two batches of true images and the fake images. Use skimage.measure.compare_ssim function to compute the mean structural similarity index between two images.
    (Input) img_true: the input should be derived from dataloader, it's in torch.ShortTensor (B,z,x,y). By default, it should be HR images.
    (Input) img_test: the input should be derived from depatching function, it's in torch.float (B,z,x,y). By default, it should be SR images.
    (Output) ssim: an ndarray with length (B,1), which contains the ssim value for each image in the batch.
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()

    img_test = np.clip(img_test, 0, 1.0)

    ssim = []
    for i in range(img_true.shape[0]):
        ssim = np.append(ssim, structural_similarity(img_true[i], img_test[i], data_range=1.0))

    return ssim


def psnr(img_true, img_test):
    img_true = img_true.cpu().numpy()

    return peak_signal_noise_ratio(img_true, img_test, data_range=1.0)