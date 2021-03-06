import argparse

import torch
from torch.utils import data

from utils import Utils
from dataset import DIV2K_train

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--x_path', type=str, default='../data/DIV2K_train_LR_bicubic/X2/')
    parser.add_argument('--y_path', type=str, default='../data/DIV2K_train_HR/')
    parser.add_argument('--scale_factor', type=int, default=2)

    # training settings
    parser.add_argument('--total_step', type=int, default=200000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)

    # config
    config = parser.parse_args()

    # device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # download data
    utils = Utils()
    utils.download('DIV2K_train_HR') # download the HR images
    utils.download('DIV2K_train_LR_bicubic_X2') # download the LR images
    print("[CUSTOM LOG]: data download done")

    # data loader`
    dataset = DIV2K_train(config=config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)