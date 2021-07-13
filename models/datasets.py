# Base data of networks
# author: ynie
# date: Feb, 2020
import os
from torch.utils.data import Dataset
import json
from utils.read_and_write import read_json

data_root = "/media/htung/Extreme SSD/fish/RfDNet/"
class ScanNet(Dataset):
    def __init__(self, cfg, mode):
        '''
        initiate SUNRGBD dataset for data loading
        :param cfg: config file
        :param mode: train/val/test mode
        '''
        self.config = cfg.config
        self.dataset_config = cfg.dataset_config
        self.mode = mode
        split_file = os.path.join(cfg.config['data']['split'], 'scannetv2_' + mode + '.json')

        self.split = read_json(split_file)

        for file_id, file in enumerate(self.split):
            self.split[file_id]['scan'] = os.path.join(data_root, file['scan'])
            self.split[file_id]['bbox'] = os.path.join(data_root, file['bbox'])
        self.split = [file for file in self.split if "scene000" in file["scan"]]


    def __len__(self):
        return len(self.split)