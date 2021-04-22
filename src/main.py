import argparse

import torch
from torch.utils import data
from train import Train
from utils import Utils
from dataset import DIV2K_train

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--x_path', type=str, default='../data/DIV2K_train_LR_bicubic/X2/')
    parser.add_argument('--y_path', type=str, default='../data/DIV2K_train_HR/')
    parser.add_argument('--y_path2', type=str, default='../data/waterloo/')
    parser.add_argument('--y_path3', type=str, default='../data/BSDS300/images/train')
    parser.add_argument('--model_save_path', type=str, default='./models')

    # training settings
    parser.add_argument('--image_size', type=int, default=40)
    parser.add_argument('--total_step', type=int, default=384000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_blocks', type=int, default=11)
    parser.add_argument('--num_channels', type=int, default=18)
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--trained_model', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--testflag', type=int, default=0)
    #misc
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=1000)
    parser.add_argument('--use_tensorboard', type=bool, default=True)

    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--result_path', type=str, default='./results')

    # config
    config = parser.parse_args()

    # device
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(config)
    print("GPU available = ", torch.cuda.is_available())
    print("# GPU = ", torch.cuda.device_count())

    # download data
    utils = Utils()
    # utils.download('DIV2K_train_HR') # download the HR images
    # utils.download('DIV2K_train_LR_bicubic_X2') # download the LR images
    print("[CUSTOM LOG]: data download done")

    # data loader`
    dataset = DIV2K_train(config=config)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    training = Train(data_loader, config)
    training.train()

    print("DONE BOOM BOOM!!")
