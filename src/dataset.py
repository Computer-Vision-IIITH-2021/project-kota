import torch
from torch.utils import data
from PIL import Image

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

    def __getiterm__(self, index):
        image_paths = self.image_paths[index]
        X_path, Y_path = image_paths[0], image_paths[1]

        X_image = Image.open(X_path).convert('RGB')
        Y_image = Image.open(Y_path).convert('RGB')

        return X_image.to(torch.float64), Y_image.to(torch.float64)
    
    def __len__(self):
        return len(self.image_paths)

