import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from ..utils.util import load_obj


class WrinkleDataset(Dataset):
    def __init__(self, image_size, n_train=100000, isTemporal=False,
                 isEval=False, isSingle=False):
        self.image_size = image_size
        self.ftexfolder = 'dataset/tex'
        self.lightfolder = 'dataset/lighting'
        self.vertsfolder = 'dataset/verts'
        self.gttexfolder = 'dataset/gttex'
        self.namelist = np.loadtxt('dataset/name.txt').astype(int).tolist()
#         print("wrinkle dataset len:")
#         print(len(self.namelist))

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        name = str(self.namelist[idx])

        ftex = cv2.imread(os.path.join(self.ftexfolder, name+'_uv_0.png'))
        light = np.load(os.path.join(self.lightfolder, name+'_gamma.npy'))
        gttex = cv2.imread(os.path.join(self.gttexfolder, name+'_uv.jpg'))
        verts = np.load(os.path.join(self.vertsfolder, name+'_verts.npy'))

        ftex = ftex[70: 370, 106: 406, :]
        ftex224 = cv2.resize(ftex, (224, 224)).transpose(2,0,1)
        ftex256 = cv2.resize(ftex, (256, 256)).transpose(2,0,1)
        gttex = gttex[70: 370, 106: 406, :]
        gttex = cv2.resize(gttex, (256, 256)).transpose(2,0,1)
        light = light.reshape(-1, 3)

        ftex224_tensor = torch.from_numpy(ftex224).type(dtype=torch.float32)  # 224,224,3
        ftex256_tensor = torch.from_numpy(ftex256).type(dtype=torch.float32)  # 224,224,3
        light_tensor = torch.from_numpy(light).type(dtype=torch.float32)  # 224,224,3
        gttex_tensor = torch.from_numpy(gttex).type(dtype=torch.float32)  # 224,224,3
        verts_tensor = torch.from_numpy(verts).type(dtype=torch.float32)  # 224,224,3

        data_dict = {
            'ftex224': ftex224_tensor,
            'ftex256': ftex256_tensor,
            'light': light_tensor,
            'gttex': gttex_tensor,
            'verts': verts_tensor
        }

        return data_dict