from os import utime
from utils import Utils

if __name__=='__main__':

    # download data
    utils = Utils()
    utils.download('DIV2K_train_HR') # download the HR images
    utils.download('DIV2K_train_LR_bicubic_X2') # download the LR images
    print("[CUSTOM LOG]: data download done")

    # 