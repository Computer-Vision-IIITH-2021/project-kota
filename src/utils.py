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
    def __init__(self, kernels, proj_matrix):
        self.kernels = kernels
        self.P = proj_matrix

        # kernels.shape == [H, W, C, N], C: no. of channels / N: no. of kernels
        self.kernels_proj = np.matmul(self.P,
                                      self.kernels.reshape(self.P.shape[-1],
                                                           self.kernels.shape[-1]))

        self.indices = np.array(range(self.kernels.shape[-1]))
        self.randkern = self.RandomKernel(self.kernels, [self.indices])

    def RandomBlur(self, image):
        kern = next(self.randkern)
        return Image.fromarray(convolve(image, kern, mode='nearest'))

    def ConcatDegraInfo(self, image):
        image = np.asarray(image)   # PIL Image to numpy array
        h, w = list(image.shape[0:2])
        proj_kernl = self.kernels_proj[:, self.randkern.index - 1]  
        n = len(proj_kernl)  # dim. of proj_kernl

        maps = np.ones((h, w, n))
        for i in range(n):
            maps[:, :, i] = proj_kernl[i] * maps[:, :, i]
        image = np.concatenate((image, maps), axis=-1)
        return image

    class RandomKernel(object):
        def __init__(self, kernels, indices):
            self.len = kernels.shape[-1]
            self.indices = indices
            np.random.shuffle(self.indices[0])
            self.kernels = kernels[:, :, :, self.indices[0]]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if (self.index == self.len):
                np.random.shuffle(self.indices[0])
                self.kernels = self.kernels[:, :, :, self.indices[0]]
                self.index = 0

            n = self.kernels[:, :, :, self.index]
            self.index += 1
            return n