"""Training script for LF-image SR.
Train a model:
        python train.py
See Readme.md for more training details.
"""

import torch
import argparse
from Models.vsnet import VsNet
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
from utils.utils import *
import os
import numpy as np
from tqdm import tqdm
from utils.patching import patching_training, calculate_overlap, patching, depatching
from utils.metrics import psnr

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Create the log path for saving the models and log.txt
log_path = './Models/model/'
os.makedirs(log_path, exist_ok=True)
file_log_name = os.path.join(log_path, 'log.txt')
logger = get_logger(file_log_name)

# Functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angRes", type=int, default=13, help="angular resolution")
    parser.add_argument("--K", type=int, default=4, help="number of cascades")
    parser.add_argument("--n_block", type=int, default=4, help="number of Inter-Blocks")
    parser.add_argument("--upscale_factor", type=int, default=3, help="upscale factor")
    parser.add_argument("--channels", type=int, default=64, help="channels")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--dataset_csv_loc', type=str, default='./Datasets/dataset.csv')
    parser.add_argument('--dataset_dir', type=str, default='./Datasets/')
    parser.add_argument('--model_name', type=str, default='LFSRmodel')
    parser.add_argument('--load_pretrain', type=bool, default=False) # if you want to load the pretrained models, set to True, and set the model_path
    parser.add_argument('--model_path', type=str, default='./Models/pretrained-models/model.pth.tar')
    parser.add_argument('--patch_size', type=int, default=25, help='patch size for the LR images')
    parser.add_argument('--patch_size_val', type=int, default=69, help='patch size for the LR images in validation')

    return parser.parse_args()

def train(train_loader, val_loader, cfg):
    logger.info('Start training!')
    logger.info('batch_size:{:3d}\t learning rate={:.6f}\t  n_steps={:3d}'.format(cfg.batch_size, float(cfg.lr), cfg.n_steps))
    
    # Model creation
    net = VsNet(angRes=cfg.angRes, K=cfg.K, n_block=cfg.n_block, channels=cfg.channels, upscale_factor=cfg.upscale_factor).to(cfg.device)
    net.apply(weights_init_xavier)
    cudnn.benchmark = True

    # Load pre-trained model if available
    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            pth_file_path = [cfg.model_path]
            model = torch.load(pth_file_path[0])  # load the pretrained model
            net.load_state_dict({k.replace('module.', ''): v for k, v in model['state_dict'].items()})
            cfg.start_epoch = model["epoch"]
        else:
            print(f"Model not found at '{cfg.load_model}'")

    net = torch.nn.DataParallel(net, device_ids=[0])
    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    for epoch in range(cfg.n_epochs):
        net.train()
        train_loss = []

        for lr_data, hr_data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}"):
            lr_data, hr_data = lr_data.to(cfg.device), hr_data.to(cfg.device)

            patch_loader=patching_training(LR_image=lr_data, HR_image=hr_data, patch_size=cfg.patch_size, batch_size=cfg.batch_size, scale=cfg.upscale_factor)

            for lr_patches, hr_patches in patch_loader:
                lr_patches=lr_patches.cuda(cfg.device)
                hr_patches=hr_patches.cuda(cfg.device)

                # Forward pass
                sr_patches = net(lr_patches)

                # Backward pass
                loss = criterion_Loss(sr_patches, hr_patches)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

        # Validate after each epoch
        val_loss, val_psnr = validate(val_loader, net, criterion_Loss)
        logger.info(f'Epoch {epoch+1}: Loss={np.mean(train_loss):.6f}, Val Loss={val_loss:.6f}, Val PSNR={val_psnr:.5f}')

        # Save the checkpoint
        if epoch % 10 == 0:
            save_ckpt({'epoch': epoch+1, 'state_dict': net.state_dict()}, log_path, f'{cfg.model_name}_epoch{epoch+1}.pth.tar')
        scheduler.step()

    logger.info('Training complete!')

def validate(val_loader, model):
    model.eval()
    psnr_list, loss_list = [], []

    with torch.no_grad():
        for lr_data, hr_data in val_loader:
            overlap = calculate_overlap(image_size=lr_data.shape[2], patch_size=cfg.patch_size_val)
            lr_patches = patching(LR_image=lr_data[0], patch_size=cfg.patch_size_val, overlap=overlap)

            sr_patches=[]
            for patch_idx in range(lr_patches.shape[0]):
                data = lr_patches[patch_idx,:, :, :]
                data = data.unsqueeze(0)
                data = Variable(data).to(cfg.device)

                with torch.no_grad():
                    out = model(data)
                    out = torch.clamp(out,0,1) # Limit the value to the range of [0,1]

                torch.cuda.empty_cache() # Release GPU memory
                sr_patches.append(out.squeeze())
            
            sr_patches = torch.stack(sr_patches, 0)
            sr_patches = sr_patches.cpu().numpy()   
            sr_data = depatching(patches=sr_patches, upscaled_overlap=overlap*cfg.upscale_factor)

            psnr_list.append(psnr(hr_data[0], sr_data))
    
    return np.mean(loss_list), np.mean(psnr_list)

# Save the trained model checkpoints
def save_ckpt(state, save_path=log_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))

# Weight initial
def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

# Main Function
def main(cfg):
    dataset = MitochondriaLoader(dataset_dir=cfg.dataset_dir, csv_loc=cfg.dataset_csv_loc)

    # Split the dataset into training, validation, and testing
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))
    train_set, val_set, _ = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_seed))

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False, num_workers=6)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6)

    train(train_loader, val_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
