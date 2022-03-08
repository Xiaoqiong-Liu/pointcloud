"""
Written by Heng Fan
Update by x.l
The KITTI class for data loading
视频序列包括点云和RGB, 由于test标签不可得，只使用21个train数据集(train 0~18; validation 17～18; test 19～20);
每个label一个tracklet;校正矩阵4个中使用P2
"""
import os
import glob
import numpy as np
# import seaborn as sns
from utility import *
import cv2 as cv
from easydict import EasyDict as edict
import pandas as pd
from pyquaternion import Quaternion
from Box import Box
from PointCloud import PointCloud

class KITTI(object):
    """
    Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,
    Andreas Geiger, Philip Lenz, and Raquel Urtasun,
    CVPR, 2012.
    """
    def __init__(self, dataset_path):
        '''
        :param dataset_path: path to the KITTI dataset
        '''
        super(KITTI, self).__init__()
        self.dataset_path = dataset_path
        self.sequence_list = self._get_sequence_list()
        self.categories = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.colors = custom_colors()
        self.tracklet_anno_list, self.tracklet_len_list = self._get_tracklet_list()
    
    def _get_tracklet_list(self):
        """
        copy from BAT
        :return: the list of tracklet label
        """
        list_of_tracklet_anno = []
        list_of_tracklet_len = []
        for scene in range(2):
            sequence_name = self.sequence_list[scene].name
            label_file = sequence_label_path = '{}/label_2/{}.txt'.format(self.dataset_path, sequence_name)

            df = pd.read_csv(
                label_file,
                sep=' ',
                names=[
                    "frame", "track_id", "type", "truncated", "occluded",
                    "alpha", "bbox_left", "bbox_top", "bbox_right",
                    "bbox_bottom", "height", "width", "length", "x", "y", "z",
                    "rotation_y"
                ])
            if self.categories in ['Car', 'Van', 'Truck',
                                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                                      'Misc']:
                df = df[df["type"] == self.categories]
            elif self.categories == 'All':
                df = df[(df["type"] == 'Car') |
                        (df["type"] == 'Van') |
                        (df["type"] == 'Pedestrian') |
                        (df["type"] == 'Cyclist')]
            else:
                df = df[df["type"] != 'DontCare']
            df.insert(loc=0, column="scene", value=scene)
            for track_id in df.track_id.unique():
                df_tracklet = df[df["track_id"] == track_id]
                df_tracklet = df_tracklet.sort_values(by=['frame'])
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                list_of_tracklet_anno.append(tracklet_anno)
                list_of_tracklet_len.append((len(tracklet_anno)))

        return list_of_tracklet_anno, list_of_tracklet_len


    def _get_sequence_list(self):
        """
        :return: the sequence list
        """

        # used to store the sequence info
        sequence_list = []

        # get all video names
        vid_names = os.listdir('{}/velodyne'.format(self.dataset_path))
        vid_names.sort()
        self.sequence_num = len(vid_names)

        for vid in vid_names:
            # store information of a sequence
            img_list = glob.glob('{}/image_2/{}/*.png'.format(self.dataset_path, vid))
            img_list.sort()
            pcloud_list = glob.glob('{}/velodyne/{}/*.bin'.format(self.dataset_path, vid))
            pcloud_list.sort()
            sequence = edict({
                'name': vid,
                'img_list': img_list,
                'img_size': self.get_sequence_img_size(img_list[0]),
                'pcloud_list': pcloud_list,
                'label_list': self.get_sequence_labels(vid),
                'calib': self.get_sequence_calib(vid)
            })

            sequence_list.append(sequence)
        return sequence_list

    def get_sequence_img_size(self, initial_img_path):
        """
        get the size of image in the sequence
        :return: image size
        """

        img = cv.imread(initial_img_path)  # read image
        img_size = {}
        img_size['height'] = img.shape[0]
        img_size['width'] = img.shape[1]
        return img_size

    def get_sequence_calib(self, sequence_name):
        """
        get the calib parameters
        :param sequence_name: sequence name
        :return: calib dictionary {projection matrix: 6 nparray (3,4), rectification matrix: 1 nparray (3,3)}
        """

        # load data
        sequence_calib_path = '{}/calib/{}.txt'.format(self.dataset_path, sequence_name)
        with open(sequence_calib_path, 'r') as f:
            calib_lines = f.readlines()

        calib = {}
        calib['P0'] = np.array(calib_lines[0].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4) #projective transformation from rectified reference camera frame to camera0
        calib['P1'] = np.array(calib_lines[1].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4) #...................................................................camera1
        calib['P2'] = np.array(calib_lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4) #...................................................................camera2
        calib['P3'] = np.array(calib_lines[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4) #...................................................................camera3
        calib['Rect'] = np.array(calib_lines[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3) #rotation for rectification
        calib['Tr_velo_cam'] = np.array(calib_lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4) #euclidean transformation from lidar to reference camera 
        calib['Tr_imu_velo'] = np.array(calib_lines[6].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4) 

        return calib

    def get_sequence_labels(self, sequence_name):
        """
        get labels for all frames in the sequence
        :param sequence_name: sequence name
        :return: the labels of a sequence
        """
        sequence_label_path = '{}/label_2/{}.txt'.format(self.dataset_path, sequence_name)
        with open(sequence_label_path, 'r') as f:
            labels = f.readlines()

        # parse each line
        # 1 frame number, 2 track id, 3 object type, 4 truncated, 5 occluded (0: full visible, 1: partly occluded, 2: largely occluded),
        # 6 alpha (Observation angle of object, ranging [-pi..pi]), 7-10 2d bbox in RGB image, 11-13 dimension (height, width, length in meters), 14-16 center location (x, y, z in meters),
        # 17 rotation around Y-axis
        frame_id_list = []
        object_list = []
        for line in labels:
            # process each line
            line = line.split()
            frame_id, object_id, object_type, truncat, occ, alpha, l, t, r, b, height, width, length, x, y, z, rotation = line

            # map string to int or float
            frame_id, object_id, truncat, occ = map(int, [frame_id, object_id, truncat, occ])
            alpha, l, t, r, b, height, width, length, x, y, z, rotation = map(float, [alpha, l, t, r, b, height, width, length, x, y, z, rotation])

            if object_type != 'DontCare':
                object = dict()    # store the information of this object
                object['id'] = object_id
                object['object_type'] = object_type
                object['truncat'] = truncat
                object['occ'] = occ
                object['alpha'] = alpha
                object['bbox'] = [l, t, r, b]
                object['dimension'] = [height, width, length]
                object['location'] = [x, y, z]
                object['rotation'] = rotation

                object_list.append(object)
                frame_id_list.append(frame_id)

        # number of frames in this sequence
        frame_num = frame_id + 1

        # collect labels for each single frame
        sequence_label = []     # the labels of all frames in the sequence
        for i in range(frame_num):
            # get all the labels in frame i
            frame_ids = get_all_index_in_list(frame_id_list, i)
            if len(frame_ids) > 0:
                frame_label = object_list[frame_ids[0]:frame_ids[-1]+1]
                sequence_label.append(frame_label)
            else:
                # for some frames, there are no objects
                sequence_label.append([])

        return sequence_label
    
    def get_frames(self, seq_id, frame_ids):
        seq_annos = self.tracklet_anno_list[seq_id]
        frames = [self._get_frame_from_anno(seq_annos[f_id]) for f_id in frame_ids]
        return frames
    
    def _get_frame_from_anno(self, anno):
        scene_id = anno['scene']
        frame_id = anno['frame']
        assert scene_id>=0 and scene_id<len(self.sequence_list), \
        'The id of the scene/sequence should be in the range [0, {}]'.format(str(self.sequence_num-1))
        calib = self.sequence_list[scene_id].calib
        
        dimension = [anno['height'],anno['width'],anno['length']]
        location = [anno['x'],anno['y'],anno['z']]
        rotation = anno['rotation_y']
        velodyne_path = '{}/velodyne/{:04}/{:06}.bin'.format(self.dataset_path, scene_id+1, frame_id)
        box_center_velo = transform_3dbox_to_pointcloud(dimension, location, rotation, returnCenter=True)[0]
        orientation = Quaternion(axis=[0, 1, 0], radians=anno["rotation_y"]) * Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
        bb = Box(box_center_velo, dimension, orientation)
        pc = PointCloud(
                    np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T)
        return {"pc": pc, "3d_bbox": bb, 'meta': anno}

# This is for debug
if __name__ == '__main__':
    kitti_path = '/Users/avivaliu/Visualize-KITTI-Objects-in-Videos/data/KITTI'
    kitti = KITTI(kitti_path)
    print(kitti)


