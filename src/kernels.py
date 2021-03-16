import torch
from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import convolve
from sklearn.decomposition import PCA

class Kernels(object):
    def __init__(self, scaleFactor):
        # big-d: the class has other values initialsed, do we not need them?
        self.allkernels = []
        self.scaleFactor = scaleFactor

        # sai: Add anisotropic kernels
        widths = [x/10 for x in range(2, 10*self.scaleFactor + 1)]
        
        self.allkernels = np.zeros((len(widths),15,15))
        for index, width in enumerate(widths):
            ker = self.isogkern(15,width)
            self.allkernels[index,:,:] = ker

        self.degradation = self.PCA()
        temp = []
        for index in range(len(self.allkernels)):
            temp.append([self.allkernels[index,:,:],self.degradation[index]])
        
        self.allkernels = temp


    def Blur(self, image, kernel):
        return Image.fromarray(convolve(image, np.array([kernel, kernel, kernel]), mode='nearest'))

    def ConcatDegraInfo(self, image, degradation):
        h, w = list(image.shape[0:2])
        n = 15 # taking n=15 PCA components
        maps = np.ones((h, w, n))
        for i in range(15):
            maps[:, :, i] = degradation[i] * maps[:, :, i]
        image = np.concatenate((image, maps), axis=-1)
        return image

    def PCA(self, k=15):

        data = self.allkernels.reshape(-1,225)
        pca = PCA(n_components=k)
        new_data = pca.fit_transform(data)
        return torch.from_numpy(new_data)        


    def isogkern(self, kernlen, std):
        gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        gkern2d = gkern2d/np.sum(gkern2d)
        return gkern2d


    def anisogkern(self, kernlen, std1, std2, angle):
        # big-d: angle NOT used
        gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
        gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d_1, gkern1d_2)
        gkern2d = gkern2d/np.sum(gkern2d)
        return gkern2d
