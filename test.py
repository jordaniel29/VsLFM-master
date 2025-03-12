import torch
import argparse
from Models.vsnet import VsNet
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from utils.utils import *
import os
import numpy as np
from tqdm import tqdm
from utils.patching import calculate_overlap, patching, depatching
from utils.metrics import psnr

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=13, help="angular resolution")
    parser.add_argument("--K", type=int, default=4, help="number of cascades")
    parser.add_argument("--n_block", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument("--upscale_factor", type=int, default=3, help="upscale factor")
    parser.add_argument("--channels", type=int, default=64, help="channels")
    parser.add_argument('--batch_size', type=int, default=2)

    parser.add_argument('--dataset_csv_loc', type=str, default='./Datasets/dataset.csv')
    parser.add_argument('--dataset_dir', type=str, default='./Datasets/')
    parser.add_argument('--save_dir', type=str, default='./Output/VSLFM/SR/')
    parser.add_argument('--model_path', type=str, default='./Models/pretrained_models/our_model.pth.tar')
    parser.add_argument('--patch_size', type=int, default=51, help='patch size for the LR images')

    return parser.parse_args()



def main(cfg):
    '''
    The input: LR data 'LR.tif' , size of [169,153,153]
    The output: SR data 'SR.tif' , size of [169,459,459]
    Note that, the input data is cropped into 9 small with the patch size of [169,69,69] before being fed into the network,
    and the 9 output SR patches would be spliced into one SR image finally.
    '''

    dir_save_path = cfg.save_dir
    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    # network
    net = VsNet(angRes=cfg.angRes, K=cfg.K, n_block=cfg.n_block, channels=cfg.channels, upscale_factor=cfg.upscale_factor).to(cfg.device)
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.eval()

    # Load the pretrained model for test
    model = torch.load(cfg.model_path)
    net.load_state_dict(model['state_dict'])

    # Load the testing dataset
    dataset = MitochondriaLoader(dataset_dir=cfg.dataset_dir, csv_loc=cfg.dataset_csv_loc)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))
    _, _, test_set = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=6)

    test_psnr = []

    with torch.no_grad():
        for lr_data, hr_data, img_name in tqdm(test_loader):
            lr_data, hr_data = lr_data.to(cfg.device), hr_data.to(cfg.device)

            overlap = calculate_overlap(image_size=lr_data.shape[2], patch_size=cfg.patch_size)
            lr_patches = patching(LR_image=lr_data[0], patch_size=cfg.patch_size, overlap=overlap)

            sr_patches=[]
            for patch_idx in range(lr_patches.shape[0]):
                data = lr_patches[patch_idx,:, :, :]
                data = data.unsqueeze(0)
                data = Variable(data).to(cfg.device)

                with torch.no_grad():
                    out = net(data)
                    out = torch.clamp(out,0,1) # Limit the value to the range of [0,1]

                torch.cuda.empty_cache() # Release GPU memory
                sr_patches.append(out.squeeze())
            
            sr_patches = torch.stack(sr_patches, 0)
            sr_patches = sr_patches.cpu().numpy()
            print(sr_patches.shape)
            sr_data = depatching(patches=sr_patches, upscaled_overlap=overlap*cfg.upscale_factor)
            print(sr_data.shape)
            exit()

            test_psnr.append(psnr(hr_data[0], sr_data))

            max_value = 3000
            sr_data *= max_value
    
            tiff.imwrite(dir_save_path + str(img_name[0]), sr_data.astype(np.uint16))   # Save the output SR image
        
        print("PSNR:", np.mean(test_psnr))


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
