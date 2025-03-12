import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import numpy as np
import logging
import torch
import tifffile as tiff
import pandas as pd

class MitochondriaLoader(Dataset):      # Preprossesing of training data
    def __init__(self, dataset_dir, csv_loc):
        super(MitochondriaLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.csv_data = pd.read_csv(csv_loc)
        self.item_num = len(os.listdir(dataset_dir + 'LR/'))

        # The maxvalue used for normalization. You can adjust it depend on the dataset used.
        self.max_value = 3000 


    def __getitem__(self, index):
        dataset_dir = self.dataset_dir

        img_name = str(self.csv_data.iloc[index, 0])
        HR_file_name = [dataset_dir + 'HR/' + img_name]
        LR_file_name = [dataset_dir + 'LR/' + img_name]

        # Normalization
        HR_data = tiff.imread(HR_file_name)/self.max_value
        LR_data = tiff.imread(LR_file_name)/self.max_value


        return torch.from_numpy(LR_data.copy()).type(torch.FloatTensor), torch.from_numpy(HR_data.copy()).type(torch.FloatTensor), img_name

    def __len__(self):
        return self.item_num



def get_logger(filename, verbosity=1, name=None):  # Log information
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

