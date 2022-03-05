## Installation
* Python == 3.6 
* Download the repository
```
git clone https://github.com/Xiaoqiong-Liu/pointcloud.git
cd Visualize-KITTI-Objects-in-Videos
```
* Create a new environment
```
conda create -n pointcloud python=3.6 # Known issue: python 3.7,3.8 not work with mayavi on MacOS in my test
conda activate pointcloud
```

* Install required packages 
```
pip install opencv-python
pip install pillow
pip install scipy
conda install yaml
conda install importlib_resources
conda install -c menpo mayavi
```

I use miniconda with Python 3.6 on macOS Big Sur 11.6 for running the code!

## Data Preparation (KITTI)
* Download <a href="http://www.cvlibs.net/datasets/kitti/eval_tracking.php">KITTI tracking data</a>, including `left color images`, `velodyne`, `camera calibration` and `training labels`.
* Unzip all the downloaded files.
* Remove `test` subfolder in each folder, and re-organize each folder as follows
```
KITTI
  --- [label_2]
        --- {0000-0020}.txt
  --- [calib]
        --- {0000-0020}.txt
  --- [image_2]
        --- [0000-0020] folders with .png images
  --- [velodyne]
        --- [0000-0020] folders with .bin files
```
If you don't want to download the dataset, a smaller version in `root_path_to_this_repo/data/KITTI/` is provided in this repository with a simplified seuqnece (sequence `0001`). You can also refer this to prepare the dataset.

