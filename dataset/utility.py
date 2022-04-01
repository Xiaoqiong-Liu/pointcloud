"""
define some utility functions
"""
import numpy as np
from torchmetrics import Metric
import torchmetrics.utilities.data
from shapely.geometry import Polygon
import torch
from shapely.validation import make_valid


def get_all_index_in_list(L, item):
    """
    get all the indexies of the same items in the list
    :param L: list
    :param item: item to be found
    :return: the indexies of all same items in the list
    """

    return [index for (index, value) in enumerate(L) if value == item]


def custom_colors():
    """
    define some colors in BGR, add more if needed
    :return: return a list of colors
    """

    colors = []

    colors.append([0, 255, 255])  # yellow
    colors.append([245, 135, 56])    # light blue
    colors.append([0, 255, 0])       # green
    colors.append([255, 0, 255])     # magenta
    colors.append([240, 32, 160])    # purple
    colors.append([255, 255, 0])     # cyan
    colors.append([0, 0, 255])       # red
    colors.append([0, 215, 255])     # gold
    colors.append([144, 238, 144])   # light green
    colors.append([128, 0, 0])       # navy
    colors.append([0, 0, 128])       # maroon
    colors.append([255, 0, 0])  # blue
    colors.append([128, 128, 0])     # teal
    colors.append([0, 128, 128])     # olive
    colors.append([128, 0, 0])       # navy

    return colors


def transform_3dbox_to_pointcloud(dimension, location, rotation, returnCenter=False):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, length = dimension
    x, y, z = location
    x_corners = [length/2, length/2, -length/2, -length/2,  length/2,  length/2, -length/2, -length/2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width/2, -width/2, -width/2, width/2, width/2, -width/2, -width/2, width/2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    centers_3d = np.vstack([0,-height/2,0])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T
    centers_3d = np.dot(R_matrix, centers_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])
    centers_3d = centers_3d + np.array([x, y, z])

    # from camera coordinate to velodyne coordinate
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])
    centers_3d = centers_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

    if returnCenter:
        return centers_3d
    return corners_3d


def velodyne_to_camera_2(pcloud, calib):

    pcloud_temp = np.hstack((pcloud[:, :3], np.ones((pcloud.shape[0], 1), dtype=np.float32)))  # [N, 4]
    pcloud_C0 = np.dot(pcloud_temp, np.dot(calib['Tr_velo_cam'].T, calib['Rect'].T))  # [N, 3]

    pcloud_C0_temp = np.hstack((pcloud_C0, np.ones((pcloud.shape[0], 1), dtype=np.float32)))
    pcloud_C2 = np.dot(pcloud_C0_temp, calib['P2'].T)  # [N, 3]
    pcloud_C2_depth = pcloud_C2[:, 2]
    pcloud_C2 = (pcloud_C2[:, :2].T / pcloud_C2[:, 2]).T

    return pcloud_C2_depth, pcloud_C2


def remove_cloudpoints_out_of_image(pcloud_C2_depth, pcloud_C2, pcloud, img_size):

    inds = pcloud_C2_depth > 0
    inds = np.logical_and(inds, pcloud_C2[:, 0] > 0)
    inds = np.logical_and(inds, pcloud_C2[:, 0] < img_size['width'])
    inds = np.logical_and(inds, pcloud_C2[:, 1] > 0)
    inds = np.logical_and(inds, pcloud_C2[:, 1] < img_size['height'])

    pcloud_in_img = pcloud[inds]

    return pcloud_in_img


def transform_3dbox_to_image(dimension, location, rotation, calib):
    """
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, length = dimension
    x, y, z = location
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
    y_corners = [0, 0, 0, 0, -height, -height, -height, -height]
    z_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # only show 3D bounding box for objects in front of the camera
    if np.any(corners_3d[:, 2] < 0.1):
        corners_3d_img = None
    else:
        # from camera coordinate to image coordinate
        corners_3d_temp = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
        corners_3d_img = np.matmul(corners_3d_temp, calib['P2'].T)
        corners_3d_img = corners_3d_img[:, :2] / corners_3d_img[:, 2][:, None]

    return corners_3d_img

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def estimateAccuracy(box_a, box_b, dim=3, up_axis=(0, -1, 0)):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        up_axis = np.array(up_axis)
        return np.linalg.norm(
            box_a.center[up_axis != 0] - box_b.center[up_axis != 0], ord=2)


def fromBoxToPoly(box, up_axis=(0, -1, 0)):
    """

    :param box:
    :param up_axis: the up axis must contain only one non-zero component
    :return:
    """
    if up_axis[1] != 0:
        return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))
    elif up_axis[2] != 0:
        return Polygon(tuple(box.bottom_corners().T))


def estimateOverlap(box_a, box_b, dim=2, up_axis=(0, -1, 0)):
    # if box_a == box_b:
    #     return 1.0

    Poly_anno = fromBoxToPoly(box_a, up_axis)
    Poly_subm = fromBoxToPoly(box_b, up_axis)
    try:
        box_inter = Poly_anno.intersection(Poly_subm)
        box_union = Poly_anno.union(Poly_subm)
    except:
        # make a shape valid as per
        Poly_subm = make_valid(Poly_subm)
        box_inter = Poly_anno.intersection(Poly_subm)
        box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area

    else:
        up_axis = np.array(up_axis)
        up_max = min(box_a.center[up_axis != 0], box_b.center[up_axis != 0])
        up_min = max(box_a.center[up_axis != 0] - box_a.wlh[2], box_b.center[up_axis != 0] - box_b.wlh[2])
        inter_vol = box_inter.area * max(0, up_max[0] - up_min[0])
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
        return overlap


class TorchPrecision(Metric):
    """Computes and stores the Precision using torchMetrics"""

    def __init__(self, n=21, max_accuracy=2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_accuracy = max_accuracy
        self.Xaxis = torch.linspace(0, self.max_accuracy, steps=n)
        self.add_state("accuracies", default=[])

    def value(self, accs):
        prec = [
            torch.sum((accs <= thres).float()) / len(accs)
            for thres in self.Xaxis
        ]
        return torch.tensor(prec)

    def update(self, val):
        self.accuracies.append(val)

    def compute(self):
        accs = torchmetrics.utilities.data.dim_zero_cat(self.accuracies)
        if accs.numel() == 0:
            return 0
        return torch.trapz(self.value(accs), x=self.Xaxis * 100 / self.max_accuracy)


class TorchSuccess(Metric):
    """Computes and stores the Success using torchMetrics"""

    def __init__(self, n=21, max_overlap=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_overlap = max_overlap
        self.Xaxis = torch.linspace(0, self.max_overlap, steps=n)
        self.add_state("overlaps", default=[])

    def value(self, overlaps):
        succ = [
            torch.sum((overlaps >= thres).float()) / len(overlaps)
            for thres in self.Xaxis
        ]
        return torch.tensor(succ)

    def compute(self):
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return 0
        return torch.tensor(np.trapz(self.value(overlaps), x=self.Xaxis) * 100 / self.max_overlap)

    def update(self, val):
        self.overlaps.append(val)




