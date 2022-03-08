"""
Copy from BAT   
Update by x.l
The KITTI sampler for pyTorch Dataloader
"""
import torch
import numpy as np
from easydict import EasyDict
from KITTI import KITTI

class TestTrackingSampler(torch.utils.data.Dataset):
    def __init__(self, dataset, config=None, **kwargs):
        if config is None:
            config = EasyDict(kwargs)
        self.dataset = dataset
        self.config = config

    def __len__(self):
        return self.dataset.get_num_tracklets()

    def __getitem__(self, index):
        """
        get item for Dataloader
        :return: frame {"pc": pc, "3d_bbox": bb, 'meta': label info}
        """
        tracklet_annos = self.dataset.tracklet_anno_list[index]
        frame_ids = list(range(len(tracklet_annos)))
        frame = self.dataset.get_frames(index, frame_ids)
        return frame

# This is for debug
if __name__ == '__main__':
    kitti_path = '/Users/avivaliu/Visualize-KITTI-Objects-in-Videos/data/KITTI'
    kitti = KITTI(kitti_path)
    sampler =  TestTrackingSampler(kitti)
    item1 = sampler.__getitem__(1)
    # print(kitti)