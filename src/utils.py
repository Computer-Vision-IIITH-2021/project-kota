import os
import shutil

class Utils:

    DATA_BASE_PATH = '../data/'
    URL = {
        'DIV2K_train_LR_bicubic_X2': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip',
        'DIV2K_train_HR': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'
    }

    def download(self, dataset_name, remove_zip=False):

        # make data folder if it doesn't exist
        try:
            os.mkdir(self.DATA_BASE_PATH)
        except OSError as err:
            print(err, "[IGNORE]")

        # wget: save as <dataset_name>.zip in ../data
        ZIP_FILE_PATH = self.DATA_BASE_PATH + dataset_name + ".zip"
        wget_cmd = "wget --continue " + self.URL[dataset_name] + " --output-document " + ZIP_FILE_PATH
        
        os.system(wget_cmd)

        # unpack dataset
        shutil.unpack_archive(ZIP_FILE_PATH, self.DATA_BASE_PATH)

        # remove zip file
        if remove_zip:
            os.system("rm " + ZIP_FILE_PATH)


class Kernels(object):
    def __init__(self, scaleFactor):
        
        self.allkernels = []
        self.scaleFactor = scaleFactor
        
        #TO DO: Add anisotropic kernels
        for width in range(0.2,self.scaleFactor,0.1):
            kernel = self.isogkern(15,width)
            degradation = PCA(kernel)
            self.allkernels.append([kernel,degradation])
            


    def Blur(self, image, kernel):
        return Image.fromarray(convolve(image, kernel, mode='nearest'))

    def ConcatDegraInfo(self, image, degradation):
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
        gkern1d_1 = signal.gaussian(kernlen, std=std1).reshape(kernlen, 1)
        gkern1d_2 = signal.gaussian(kernlen, std=std2).reshape(kernlen, 1)
        gkern2d = np.outer(gkern1d_1, gkern1d_2)
        gkern2d = gkern2d/np.sum(gkern2d)
        return gkern2d
