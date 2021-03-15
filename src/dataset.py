import torch
from torch.utils import data
from PIL import Image
# import h5py
import numpy as np
from kernels import Kernels
from torchvision import transforms
import math
import random
# from torchvision.transforms.functional import InterpolationMode

def Scaling(image):
    return np.array(image) / 255.0

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=float(random.randint(0, 76))):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class DIV2K_train(data.Dataset):
    def __init__(self, config=None):

        self.image_paths = []

        for i in range(1,800+1):
            name = '0000' + str(i)
            name = name[-4:]
            # name = "0" * (4-int(math.log10(i))+1) + str(i)
            Y_path = config.y_path + name + '.png'
            self.image_paths.append(Y_path)

        self.scale_factor = config.scale_factor
        self.image_size = config.image_size

        self.kernels = Kernels(self.scale_factor)


    def __getitem__(self, index):
        Y_path = self.image_paths[index]

        Y_image = Image.open(Y_path).convert('RGB') # hr image
        X_image, Y_image = self.transformlr(Y_image)

        return X_image.to(torch.float64), Y_image.to(torch.float64)

    def __len__(self):
        return len(self.image_paths)

    def transformlr(self, Y_image):
        transform = transforms.RandomCrop(self.image_size * self.scale_factor)
        hr_image = transform(Y_image)

        kernel, degradinfo = random.choice(self.kernels.kernels)
        # input (low-resolution image)

        transform = transforms.Compose([
                            transforms.Lambda(lambda x: self.kernels.Blur(x,kernel)),
                            # transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                            transforms.Resize((self.image_size, self.image_size)),
                            transforms.Lambda(lambda x: Scaling(x)),
                            transforms.Lambda(lambda x: self.kernels.ConcatDegraInfo(x,degradinfo)),
                            AddGaussianNoise()
                    ])
        lr_image = transform(hr_image)

        transform = transforms.ToTensor()
        lr_image, hr_image = transform(lr_image), transform(hr_image)
        return lr_image, hr_image
