"""
Created by x.l at 2022/03/03
A test to visualize Kitti 
"""

from dataset.KITTI import KITTI
import argparse
import yaml
from easydict import EasyDict
from dataset.KITTIView import *


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def parse_config():

    parser = argparse.ArgumentParser(description="Visualize KITTI Objects in Videos")

    parser.add_argument('--dataset_path', type=str, default='/Users/avivaliu/Visualize-KITTI-Objects-in-Videos/data/KITTI', help='the path to KITTI, a default dataset is provided')
    parser.add_argument('--sequence_id', type=int, default=0, help='the id of sequence to visualize')
    parser.add_argument('--vis_data_type', type=str, choices=['camera', 'pointcloud', 'bev'], default='camera', \
                        help='show object in camera, pointcloud or birds eye view')
    parser.add_argument('--fov', action='store_true', help='only show front view of pointcloud')
    parser.add_argument('--vis_box', action='store_true', help='show object box or not')
    parser.add_argument('--box_type', type=str, default='3d', choices=['2d', '3d'], help='designed for visualization in camera, show 2d or 3d object box')
    parser.add_argument('--save_img', action='store_true', help='save visualization result or not')
    parser.add_argument('--save_path', type=str, default=None, help='path to save visualization result')
    parser.add_argument('--cfg', type=str, default='/Users/avivaliu/Visualize-KITTI-Objects-in-Videos/config/BAT.yaml', help='the config_file')
    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args

    return EasyDict(config)

#entry poit
if __name__ == '__main__':
    args = parse_config()

    # create KITTI dataset
    kitti = KITTI(args.dataset_path)

    # perform visualization
    if args.vis_data_type == 'camera':
        # visualize object in camera image
        vis_2dbox = False
        vis_3dbox = False
        if args.vis_box:
            if args.box_type == '2d':
                vis_2dbox = True
            if args.box_type == '3d':
                vis_3dbox = True
        kitti.show_sequence_rgb(args.sequence_id, vis_2dbox=vis_2dbox, vis_3dbox=vis_3dbox,
                                save_img=args.save_img, save_path=args.save_path)

    elif args.vis_data_type == 'pointcloud':
        # visualize object in pointcloud
        kitti.show_sequence_pointcloud(args.sequence_id, img_region=args.fov, vis_box=args.vis_box,
                                        save_img=args.save_img, save_path=args.save_path)

    elif args.vis_data_type == 'bev':
        # visualize object in bird's eye view
        kitti.show_sequence_BEV(args.sequence_id, img_region=args.fov, vis_box=args.vis_box,
                                        save_img=args.save_img, save_path=args.save_path)
