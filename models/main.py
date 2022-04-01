import pytorch_lightning as pl
import argparse
from dataset.KITTI import KITTI
from dataset.KITTISampler import KITTISampler
from torch.utils.data import DataLoader
from bat import bat
import yaml
from easydict import EasyDict

kitti_path = '/mnt/Data/KITTI'

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/UNT/xl0217/pointcloud/pretrained_models/bat_kitti_car.ckpt', help='checkpoint location')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--gpu', type=int, nargs='+', default=(0, 1), help='specify gpu devices')
    parser.add_argument('--cfg', type=str, default='/home/UNT/xl0217/pointcloud/config/BAT.yaml', help='the config_file')
    parser.add_argument('--use_fps', default=True, help='specify use farthest point sampling or not')
    parser.add_argument('--test', action='store_true', default=True, help='test mode')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args
    return EasyDict(config)

cfg = parse_config()


"""""""1.implement bat"""""""
# prepare data
kitti = KITTI(kitti_path, split='test', config=cfg)
test_data =  KITTISampler(kitti)  
train_loader = DataLoader(test_data, batch_size=1, num_workers=40, collate_fn=lambda x: x, pin_memory=True)
# get model
net = bat(cfg).load_from_checkpoint(cfg.checkpoint, config=cfg)
trainer = pl.Trainer(gpus=cfg.gpu, accelerator="ddp")
trainer.validate(net, train_loader)