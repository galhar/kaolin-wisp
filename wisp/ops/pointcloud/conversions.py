# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from wisp.core import Rays


def create_pointcloud_from_images(rgbs, masks, rays, depths):
    """Given depth images, will create a RGB pointcloud.

    TODO (ttakikawa): Probably make the input a tensor not a list...

    Args:
        rgbs (list of torch.FloatTensor): List of RGB tensors of shape [H, W, 3].
        masks (list of torch.FloatTensor): List of mask tensors of shape [H, W, 1].
        rays (list of wisp.core.Rays): List of rays.origins and rays.dirs of shape [H, W, 3].
        depths (list of torch.FloatTensor): List of depth tensors of shape [H, W, 1].

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        - 3D coordinates of shape [N*H*W, 3]
        - colors of shape [N*H*W, 3]
    """
    cloud_coords = []
    cloud_colors = []

    for i in range(len(rgbs)):
        mask = masks[i].bool()
        h, w = mask.shape[:2]
        mask = mask.reshape(h, w)
        depth = depths[i].reshape(h, w, 1)
        assert(len(mask.shape) == 2 and "Mask shape is not correct... it should be [H,W], check size here")
        coords = rays[i].origins[mask] + rays[i].dirs[mask] * depth[mask]
        colors = rgbs[i][mask]
        cloud_coords.append(coords.reshape(-1, 3))
        cloud_colors.append(colors[...,:3].reshape(-1, 3))

    return torch.cat(cloud_coords, dim=0), torch.cat(cloud_colors, dim=0)

def create_edges_pointcloud_from_rays(rays, masks=None):
    """Given rays, will create a coords pointcloud of a little gap from the edges.

    Args:
        masks (torch.FloatTensor): mask tensors of shape [H, W, 1].
        rays (wisp.core.Rays): rays.origins and rays.dirs of shape [H, W, 3].

    Returns:
        (torch.FloatTensor, torch.FloatTensor):
        - 3D coordinates of shape [N*H*W, 3]
        - colors of shape [N*H*W, 3]
    """
    low_edge_fac = 1.09
    high_edge_fac = 0.999
    low_edge = rays.dist_min * low_edge_fac
    high_edge = rays.dist_max * high_edge_fac
    if not isinstance(low_edge, torch.Tensor):
        low_edge = torch.full((*rays.shape, 1), low_edge)
    if not isinstance(high_edge, torch.Tensor):
        high_edge = torch.full((*rays.shape, 1), high_edge)

    depths = torch.concat([low_edge, high_edge])
    rays = Rays.cat([rays, rays])
    dummy_rgbs = torch.zeros((*rays.shape, 3))
    if masks is None:
        masks = torch.ones((*dummy_rgbs.shape[:-1], 1))

    cloud_coords, _ = create_pointcloud_from_images(dummy_rgbs, masks, rays, depths)

    return cloud_coords
