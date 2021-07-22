# Base data of networks
# author: ynie
# date: Feb, 2020
import os
from torch.utils.data import Dataset
import json
from utils.read_and_write import read_json
import socket

hostname = socket.gethostname()
if hostname == "aw-m17-R2":
    data_path = f"/media/htung/Extreme SSD/fish/"
elif hostname.endswith("ccncluster") or "physion" in hostname:
    data_path = f"/mnt/fs4/hsiaoyut"
    if hostname.endswith("node19-ccncluster"):
        data_root = "/mnt/fs1/hsiaoyut/DPI-Net/data/"
    #if "physion" in hostname:
    out_root = "/mnt/fs1/hsiaoyut"
else:
    raise ValueError


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

        if self.config['data']['dataset'] == 'scannet':
            self.data_root = os.path.join(data_path, "RfDNet/")
            split_file = os.path.join(cfg.config['data']['split'], 'scannetv2_' + mode + '.json')

            self.split = read_json(split_file)

            for file_id, file in enumerate(self.split):
                self.split[file_id]['scan'] = os.path.join(self.data_root, file['scan'])
                self.split[file_id]['bbox'] = os.path.join(self.data_root, file['bbox'])
            self.split = [file for file in self.split if "scene000" in file["scan"]]
        elif self.config['data']['dataset'] == 'tdw_physics':
            self.data_names = ['positions', 'velocities']
            self.all_trials = []
            self.n_rollout = 0
            self.data_root = os.path.join(data_path, "DPI-Net/")
            if mode == "val":
                mode = "valid"

            self.data_dir = [os.path.join(self.data_root, cfg.config['data']['split'], mode)]

            for ddir in self.data_dir:
                file = open(ddir +  ".txt", "r")
                ddir_root = "/".join(ddir.split("/")[:-1])
                trial_names = [line.strip("\n") for line in file if line != "\n"]
                n_trials = len(trial_names)

                self.all_trials += [os.path.join(ddir_root, trial_name) for trial_name in trial_names]
                self.n_rollout += n_trials

            if mode == "train":
                self.mean_time_step = int(13499/self.n_rollout) + 1
            else:
                self.mean_time_step = 1


    def __len__(self):

        if self.config['data']['dataset'] == 'scannet':
            return len(self.split)
            # each rollout can have different length, sample length in get_item
        elif self.config['data']['dataset'] == 'tdw_physics':
            return self.n_rollout * self.mean_time_step

        return len(self.split)