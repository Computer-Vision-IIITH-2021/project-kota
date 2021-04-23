import cv2
import math
import torch
import random
import numpy as np
from PIL import Image
from scipy import signal
from scipy.ndimage import convolve
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import random


class Kernels(object):
    # def __init__(self, scaleFactor):
    #     # big-d: the class has other values initialsed, do we not need them?
    #     self.allkernels = []
    #     self.scaleFactor = scaleFactor
    #
    #     self.allkernels = np.zeros((10000,15,15))
    #
    #     for count in range(10000):
    #
    #         theta = random.random() * np.pi
    #         l1 = 0.5 + random.random() * ((self.scaleFactor * 2 ) + 1.5)
    #         l2 = 0.5 + random.random() * (l1-0.5)
    #
    #         ker = self.getKernel(theta,l1,l2)
    #         self.allkernels[count,:,:] = ker
    #
    #     self.degradation = self.PCA()
    #     temp = []
    #     for index in range(len(self.allkernels)):
    #         temp.append([self.allkernels[index,:,:],self.degradation[index]])
    #
    #     self.allkernels = temp

    def __init__(self, scaleFactor):
        # big-d: the class has other values initialsed, do we not need them?
        self.allkernels = []
        self.scaleFactor = scaleFactor

        # sai: Add anisotropic kernels
        widths = [x/10 for x in range(2, 10*self.scaleFactor + 1)]

        self.allkernels = np.zeros((len(widths),15,15))
        for index, width in enumerate(widths):
            yeet=random.randint(0, 1)
            if yeet==0:
                ker = self.isogkern(15,width)
            else:
                ker = self.anisogkern(15,width,random.uniform(0.2,width))
            self.allkernels[index,:,:] = ker

        self.degradation = self.PCA()
        temp = []
        for index in range(len(self.allkernels)):
            temp.append([self.allkernels[index,:,:],self.degradation[index]])

        self.allkernels = temp


    def Blur(self, image, kernel):
        image = np.asarray(image)
        dst = cv2.filter2D(image,-1,kernel)
        return Image.fromarray(dst)

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


    def anisogkern(self, kernlen, std1, std2):
        # big-d: angle NOT used
        gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
        gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d_1, gkern1d_2)
        gkern2d = gkern2d/np.sum(gkern2d)
        return gkern2d


    def getKernel(self,theta,l1,l2):

        v = np.dot([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]],[[1],[0]])

        V = [[v[0][0], v[1][0]],[v[1][0], -v[0][0]]]
        D = [[l1, 0],[0, l2]]

        Sigma = np.dot(np.dot(V,D),np.linalg.inv(V))
        rv = multivariate_normal([7,7], Sigma)

        ker = np.zeros((15,15))

        for i in range(15):
            for j in range(15):
                ker[i][j] = rv.pdf([i,j])

        ker /= np.sum(ker)
        return ker
