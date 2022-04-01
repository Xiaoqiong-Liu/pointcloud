""" 
pointnet.py
Created by zenn at 2021/5/9 13:41
"""

import torch
import torch.nn.functional as F

import torch.nn as nn
from utils import pytorch_utils as pt_utils
from utils import pointnet2_utils
# from utils.pointnet2_modules import PointnetSAModule

class Pointnet_Backbone(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, use_fps=False, normalize_xyz=False, return_intermediate=False):
        super(Pointnet_Backbone, self).__init__()
        self.return_intermediate = return_intermediate
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.3,
                nsample=32,
                mlp=[0, 64, 64, 128],
                use_xyz=True,
                use_fps=use_fps, normalize_xyz=normalize_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.5,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                use_fps=False, normalize_xyz=normalize_xyz
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                radius=0.7,
                nsample=32,
                mlp=[256, 256, 256, 256],
                use_xyz=True,
                use_fps=False, normalize_xyz=normalize_xyz
            )
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features, l_idxs = [xyz], [features], []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, sample_idxs = self.SA_modules[i](l_xyz[i], l_features[i], numpoints[i], True)
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_idxs.append(sample_idxs)
        if self.return_intermediate:
            return l_xyz[1:], l_features[1:], l_idxs[0]
        return l_xyz[-1], l_features[-1], l_idxs[0]

class _PointnetSAModuleBase(nn.Module):
    def __init__(self, use_fps=False):
        super(_PointnetSAModuleBase, self).__init__()
        self.groupers = None
        self.mlps = None
        self.use_fps = use_fps

    def forward(self, xyz, features, npoint, return_idx=False):
        # modified to return sample idxs
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        self.npoint = npoint
        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.use_fps:
            sample_idxs = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        else:
            sample_idxs = torch.arange(self.npoint).repeat(xyz.size(0), 1).int().cuda()

        new_xyz = (
            pointnet2_utils.gather_operation(xyz_flipped, sample_idxs)
                .transpose(1, 2)
                .contiguous()
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)
        if return_idx:
            return new_xyz, torch.cat(new_features_list, dim=1), sample_idxs
        else:
            return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, radii, nsamples, mlps, bn=True, use_xyz=True, use_fps=False, normalize_xyz=False):
        # type: (PointnetSAModuleMSG, List[float],List[int], List[List[int]], bool, bool,bool) -> None
        super(PointnetSAModuleMSG, self).__init__(use_fps=use_fps)

        assert len(radii) == len(nsamples) == len(mlps)

        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz, normalize_xyz=normalize_xyz))

            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))

class PointnetSAModule(PointnetSAModuleMSG):
    """Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
            self, mlp, radius=None, nsample=None, bn=True, use_xyz=True, use_fps=False, normalize_xyz=False
    ):
        # type: (PointnetSAModule, List[int], float, int, bool, bool, bool,bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            use_fps=use_fps,
            normalize_xyz=normalize_xyz
        )
