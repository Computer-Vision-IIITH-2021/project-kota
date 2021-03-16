import torch
from torch.utils import data
from PIL import Image
import numpy as np
from kernels import Kernels
from torchvision import transforms
import math
import random
from torchvision.utils import save_image

def Scaling(image):
    return np.array(image) / 255.0

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=float(random.randint(0, 76))):
        self.std = std
        self.mean = mean

    def __call__(self, vec):
        vec = np.asarray(vec)
        h, w, n = vec.shape
        return vec + np.random.rand(h,w,n) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class DIV2K_train(data.Dataset):
    def __init__(self, config=None):

        self.image_paths = []
        num_images = 800
        for i in range(1, num_images+1):
            name = '0000' + str(i)
            name = name[-4:]
            Y_path = config.y_path + name + '.png'
            self.image_paths.append(Y_path)

        self.scale_factor = config.scale_factor
        self.image_size = config.image_size

        self.kernels = Kernels(self.scale_factor)


    def __getitem__(self, index):
        Y_path = self.image_paths[index]

        Y_image = Image.open(Y_path).convert('RGB') # hr image
        Y_image.save("ogyimage"+str(index)+".jpg")
        X_image, Y_image = self.transformlr(Y_image,index)

        # save_image(X_image, "transximage"+str(index)+".jpg")
        # save_image(Y_image, "transyimage"+str(index)+".jpg")
        # X_image.save("transximage"+str(index)+".jpg")
        # Y_image.save("transyimage"+str(index)+".jpg")

        return X_image.to(torch.float64), Y_image.to(torch.float64)

    def __len__(self):
        return len(self.image_paths)

    def transformlr(self, Y_image,index):
        transform = transforms.RandomCrop(self.image_size * self.scale_factor)
        hr_image = transform(Y_image) #image
        print(hr_image)
        hr_image.save("transyimage"+str(index)+".jpg")

        # print(hr_image,index)
        kernel, degradinfo = random.choice(self.kernels.allkernels)
        # input (low-resolution image)

        transform = transforms.Compose([
                            transforms.Lambda(lambda x: self.kernels.Blur(x,kernel)),
                            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC)
                            # AddGaussianNoise(),
                    ])
        lr_image = np.asarray(transform(hr_image)) #numpy
        print("here",lr_image.shape)
        temp = Image.fromarray(lr_image.astype(np.uint8))
        temp.save("transximage"+str(index)+".jpg")

        transform = transforms.Compose([transforms.Lambda(lambda x: self.kernels.ConcatDegraInfo(x,degradinfo))])
        lr_image = transform(lr_image)



        # print(2)
        # print(lr_image.shape,index)
        # lr_image.save("transximage"+str(index)+".jpg")
        # hr_image.save("transyimage"+str(index)+".jpg")
        # im = Image.fromarray(lr_image)
        # im.save("transximage"+str(index)+".jpg")
        # im = Image.fromarray(hr_image)
        # im.save("transyimage"+str(index)+".jpg")
        transform = transforms.ToTensor()
        lr_image, hr_image = transform(lr_image), transform(hr_image)
        # print(lr_image.shape,index) #tensor
        # print(hr_image.shape,index) #tensor
        return lr_image, hr_image
