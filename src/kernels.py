import torch
from PIL import Image
import numpy as np
from scipy import signal
from scipy.ndimage import convolve
from sklearn.decomposition import PCA

class Kernels(object):
    def __init__(self, scaleFactor):
        # big-d: the class has other values initialsed, do we not need them?
        self.kernels = []
        self.scaleFactor = scaleFactor
        
        # sai: Add anisotropic kernels
        widths = [x/10 for x in range(2, 10*self.scaleFactor + 1)]
        for width in widths:
            kernel = self.isogkern(15,width)
            degradation = self.PCA(kernel)
            self.kernels.append([kernel,degradation])
            


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
    
    def PCA(self, data, k=15):
        # old_X = torch.from_numpy(data)
        # old_X = torch.from_numpy(data.reshape(1,-1))
        pca = PCA(n_components=15, svd_solver='full')
        print(data.reshape(-1, 1).shape)
        X = pca.fit_transform(data.reshape(-1, 1))
        # print(type(X))
        return torch.from_numpy(X)

        # X = torch.from_numpy(data.flatten())
        # X_mean = torch.mean(X, 0)
        # X = X - X_mean.expand_as(X)

        # v, w = torch.eig(torch.mm(X, torch.t(X)), eigenvectors=True)
        # return torch.mm(w[:k, :], X)


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
