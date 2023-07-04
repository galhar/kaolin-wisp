import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

EMPTY_POINT = torch.tensor([-1,-1])

def return_heatmap(src_vec, dst_fts):
    """
    Args:
        src_vec: dim C, features of a single point in the src_features image
        dst_fts: NxCxHxW all the target images' features to create heatmaps with
    Returns:

    """
    num_channel = src_vec.size(0)
    cos = nn.CosineSimilarity(dim=1)
    with torch.no_grad():

        src_vec = src_vec.view(1, num_channel, 1, 1)  # 1, C, 1, 1
        cos_map = cos(src_vec, dst_fts)  # N, H, W
        return cos_map

def return_correspondences(src_vec, dst_fts, thresh=0.7, is_norm=True):
    """
    This function will be called over each sampled point during the NeRF's training
    Args:
        src_ft:
        dst_fts:
        point_1:
        thresh:
        is_norm:

    Returns:

    """
    cos_map = return_heatmap(src_vec, dst_fts)
    if is_norm:
        min_vec = torch.min(cos_map, dim=0)
        cos_map = (cos_map - min_vec) / (torch.max(cos_map, dim=0) - min_vec)

    # TODO: compare different threshes manually, and wether it's better to use a threshold over normalized (to get "how unique a point is") or over the absolute value (to get "how strong a match comparing to other matches is it")

    # TODO: check it works... and also the normalization above. Dimension issues might arise. Also np and torch are mixed here
    max_points = np.unravel_index(cos_map.argmax(axis=1), cos_map.shape)
    max_values = cos_map[max_points]

    del cos_map
    gc.collect()
    torch.cuda.empty_cache()

    max_points[max_values < thresh] = EMPTY_POINT
    return max_points

