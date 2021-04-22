import torch
import torch.nn as nn
import os
from torchvision.utils import save_image, make_grid
from model import SRMD
import numpy as np
import math
import cv2
import sys
from piqa import SSIM
torch.set_default_tensor_type(torch.DoubleTensor)

class Train(object):
    def __init__(self, data_loader, config):
        # Data loader
        self.data_loader = data_loader

        # Model hyper-parameters
        self.num_blocks = config.num_blocks
        self.num_channels = config.num_channels
        self.conv_dim = config.conv_dim
        self.scale_factor = config.scale_factor

        # Training settings
        self.total_step = config.total_step
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.trained_model = config.trained_model
        self.use_tensorboard = config.use_tensorboard

        # Path and step size
        self.log_path = config.log_path
        self.result_path = config.result_path
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Device configuration
        self.device = config.device

        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.trained_model:
            self.load_trained_model()

    def calculate_psnr(self, img1, img2):
    # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def ssim(self, img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


    def calculate_ssim(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return self.ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            print("Image 1:", img1.ndim)
            print("Image 2:", img2.ndim)
            raise ValueError('Wrong input image dimensions.')

    def build_model(self):
        # model and optimizer
        self.model = SRMD(self.num_blocks, self.num_channels, self.conv_dim, self.scale_factor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2])

        self.model.to(self.device)

    def load_trained_model(self):
        self.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.trained_model)))
        print('loaded trained models (step: {})..!'.format(self.trained_model))

    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S['SR'])

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        self.optimizer.zero_grad()

    def detach(self, x):
        return x.data

    def train(self):
        self.model.train()

        # Reconst loss
        reconst_loss = nn.MSELoss()

        # Data iter
        data_iter = iter(self.data_loader)
        iter_per_epoch = len(self.data_loader)

        # Start with trained model
        if self.trained_model:
            start = self.trained_model + 1
        else:
            start = 0

        for step in range(start, self.total_step):
            # Reset data_iter for each epoch
            if (step+1) % iter_per_epoch == 0:
                data_iter = iter(self.data_loader)

            actx, x, y = next(data_iter)
            actx, x, y = actx.to(self.device), x.to(self.device), y.to(self.device)
            y = y.to(torch.float64)

            out = self.model(x)
            loss = reconst_loss(out, y)

            self.reset_grad()

            # For decoder
            loss.backward(retain_graph=True)

            self.optimizer.step()

            # Print out log info
            if (step+1) % self.log_step == 0:
                print("[{}/{}] loss: {:.4f}".format(step+1, self.total_step, loss.item()))

            class SSIMLoss(SSIM):
                def forward(self, x, y):
                    print("SSIM Shape GPU:", x.shape, y.shape)
                    print("SSIM type:", x.dtype, y.dtype)
                    x, y = x.to('cpu').unsqueeze(0).type(torch.FloatTensor), y.to('cpu').unsqueeze(0).type(torch.FloatTensor)
                    print("SSIM type double:", x.dtype, y.dtype)
                    print("SSIM Shape CPU:", x.shape, y.shape)
                    return 1. - super().forward(x, y)

            criterion_ssim = SSIMLoss()
                    
            # Sample images
            if (step+1) % self.sample_step == 0:
                self.model.eval()
                reconst = self.model(x)

                def to_np(x):
                    return x.data.cpu().numpy()

                tmp = nn.Upsample(scale_factor=self.scale_factor)(actx.data[:,:,:])
                pairs = torch.cat((tmp.data[0:2,:], reconst.data[0:2,:], y.data[0:2,:]), dim=3)
                psnrscore1= self.calculate_psnr(to_np(reconst.data[0,:]),to_np(y.data[0,:]))
                print("Shape: ",tmp.data.shape, reconst.data.shape, y.data.shape)
                ssimscore1= criterion_ssim(reconst.data[0,:],y.data[0,:])
                psnrscore2= self.calculate_psnr(to_np(reconst.data[1,:]),to_np(y.data[1,:]))
                ssimscore2= criterion_ssim(reconst.data[1,:],y.data[1,:])
                with open('score.txt', 'a') as f:
                    print('test_{}.jpg PSNR1:{} SSIM1:{} PSNR2:{} SSIM2:{}'.format(step + 1,psnrscore1,ssimscore1,psnrscore2,ssimscore2), file=f)
                f.close()
                pairs = pairs.to('cpu')
                grid = make_grid(pairs, 2)
                from PIL import Image
                tmp = tmp.to('cpu')
                tmp = np.squeeze(grid.numpy().transpose((1, 2, 0)))
                # tmp = torch.from_numpy(tmp)
                tmp = (255 * tmp).astype(np.uint8)
                Image.fromarray(tmp).save('./samples/test_%d.jpg' % (step + 1))

            # Save check points
            if (step+1) % self.model_save_step == 0:
                self.save(os.path.join(self.model_save_path, '{}.pth'.format(self.trained_model)))
    def test(self):
        self.model.eval()
        reconst_loss = nn.MSELoss()
        avg_psnr=0
        avg_ssim=0
        class SSIMLoss(SSIM):
            def forward(self, x, y):
                return 1. - super().forward(x, y)

        criterion_ssim = SSIMLoss()
        data_iter = iter(self.data_loader)
        num_iters = len(self.data_loader)
        for step in range(num_iters):
            actx, x, y = next(data_iter)
            actx, x, y = actx.to(self.device), x.to(self.device), y.to(self.device)
            y = y.to(torch.float64)
            out = self.model(x)
            for i in range(self.batch_size):
                ssim = criterion_ssim(out.data[i,:], y.data[i,:])
                avg_ssim += ssim
            mse = reconst_loss(out, y)
            psnr = 10 * math.log10(1 / mse.item())
            avg_psnr += psnr
        with open('testscore.txt', 'a') as f:
            print(f'PSNR : {avg_psnr/num_iters} SSIM : {avg_ssim/num_iters}',file=f)
        f.close()

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'SR': model}, filename)
