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

