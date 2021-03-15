import torch
from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import convolve

class Kernels(object):
    def __init__(self, scaleFactor):
        # big-d: the class has other values initialsed, do we not need them?
        self.allkernels = []
        self.scaleFactor = scaleFactor
        
        # sai: Add anisotropic kernels
        for width in range(0.2,self.scaleFactor,0.1):
            kernel = self.isogkern(15,width)
            degradation = self.PCA(kernel)
            self.allkernels.append([kernel,degradation])
            


    def Blur(self, image, kernel):
        return Image.fromarray(self.convolve(image, kernel, mode='nearest'))

    def ConcatDegraInfo(self, image, degradation):
        # big-d: h, w, n not defined
        maps = np.ones((h, w, n))
        for i in range(15):
            maps[:, :, i] = degradation[i] * maps[:, :, i]
        image = np.concatenate((image, maps), axis=-1)
        return image
    
    def PCA(data, k=15):
        X = torch.from_numpy(data)
        X_mean = torch.mean(X, 0)
        X = X - X_mean.expand_as(X)

        v, w = torch.eig(torch.mm(X, torch.t(X)), eigenvectors=True)
        return torch.mm(w[:k, :], X)

    def isogkern(kernlen, std):
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d = gkern2d/np.sum(gkern2d)
        return gkern2d


    def anisogkern(kernlen, std1, std2, angle):
        # big-d: angle NOT used
        gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
        gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d_1, gkern1d_2)
        gkern2d = gkern2d/np.sum(gkern2d)
        return gkern2d
