# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from configs.path_config import SHAPENETCLASSES
from configs.path_config import ScanNet_OBJ_CLASS_IDS as OBJ_CLASS_IDS
import torch

class TdwPhysicsConfig(object):
    def __init__(self):
        self.num_class = len(OBJ_CLASS_IDS)
        self.num_heading_bin = 12
        self.num_size_cluster = len(OBJ_CLASS_IDS)

        #self.type2class = {SHAPENETCLASSES[cls]:index for index, cls in enumerate(OBJ_CLASS_IDS)}
        #self.class2type = {self.type2class[t]: t for t in self.type2class}
        #self.class_ids = OBJ_CLASS_IDS
        self.mean_size_arr = np.array([0.4, 0.4, 0.4])
        self.type_mean_size = {}
        self.data_path = 'datasets/scannet/processed_data'
        #for i in range(self.num_size_cluster):
        #    self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]
        self.with_rotation = False
    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, tmp, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr + residual

    def class2size_cuda(self, residual):
        ''' Inverse function to size2class '''
        mean_size_arr = torch.from_numpy(self.mean_size_arr).to(residual.device).float()
        return torch.expand_dims(mean_size_arr, axis=0) + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb



if __name__ == '__main__':
    cfg = ScannetConfig()