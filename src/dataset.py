import torch
from torch.utils import data
from PIL import Image
import h5py
from utils import Kernels
from torchvision import transforms


class DIV2K_train(data.Dataset):
    def __init__(self, config=None):
        # ids are from 0001 to 0800
        # image paths = [[X_root/0001x2.png, Y_root/0001.png], ..., []]
        self.image_paths = []
        for i in range(1,800+1):
            name = '0000' + str(i)
            name = name[:-4]
            X_path = config.x_path + name + 'x' + str(config.scale_factor) + '.png'
            Y_path = config.y_path + name + '.png'
            self.image_paths.append([X_path, Y_path])
        self.scale_factor = config.scale_factor
        self.image_size = config.image_size
        filepath = 'kernels/SRMDNFx' + str(config.scale_factor) + '.mat'
        f = h5py.File(filepath, 'r')
        K, P = extract_KP(f, config.scale_factor)
        self.randkern = Kernels(K, P)


    def __getiterm__(self, index):
        image_paths = self.image_paths[index]
        X_path, Y_path = image_paths[0], image_paths[1]

        X_image = Image.open(X_path).convert('RGB')
        Y_image = Image.open(Y_path).convert('RGB')

        #X_image, Y_image = transformlr(Y_image)

        return X_image.to(torch.float64), Y_image.to(torch.float64)
    
    def __len__(self):
        return len(self.image_paths)

    def transformlr(Y_image):
        transform = transforms.RandomCrop(self.image_size * self.scale_factor)
        hr_image = transform(Y_image)

        # input (low-resolution image)
        transform = transforms.Compose([
                            transforms.Lambda(lambda x: self.randkern.RandomBlur(x)),
                            transforms.Resize((self.image_size, self.image_size)),
                            transforms.Lambda(lambda x: Scaling(x)),
                            transforms.Lambda(lambda x: self.randkern.ConcatDegraInfo(x))
                    ])
        lr_image = transform(hr_image)

        transform = transforms.ToTensor()
        lr_image, hr_image = transform(lr_image), transform(hr_image)
        return lr_image, hr_image

def extract_KP(f, scale_factor = 2):
    P = np.array(f['net/meta/P']).T

    directKernel = None
    if scale_factor != 4:
        directKernel = np.array(f['net/meta/directKernel']).transpose(3, 2, 1, 0)

    AtrpGaussianKernels = np.array(f['net/meta/AtrpGaussianKernel']).transpose(3, 2, 1, 0)

    if directKernel is None:
        K = AtrpGaussianKernels
    else:
        K = np.concatenate((directKernel, AtrpGaussianKernels), axis=-1)

    return K, P

