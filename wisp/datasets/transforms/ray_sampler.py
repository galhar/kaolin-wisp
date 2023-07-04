# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import numpy as np
import torch
from wisp.datasets.batch import MultiviewBatch
from wisp.datasets.dift.utils.utils import return_correspondences


class SampleRays:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples):
        self.num_samples = num_samples

    @torch.cuda.nvtx.range("SampleRays")
    def __call__(self, inputs: MultiviewBatch):
        device = inputs['rays'].origins.device
        ray_idx = torch.randint(0, inputs['rays'].shape[0], [self.num_samples], device=device)

        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()

        # Loop over ray values in this batch
        for channel_name, ray_value in inputs.ray_values().items():
            out[channel_name] = ray_value[ray_idx].contiguous()
        return out


class SampleRaysCalculateCorrespondences:
    """ A dataset transform for sub-sampling a fixed amount of rays. """
    def __init__(self, num_samples):
        self.num_samples = num_samples

    @torch.cuda.nvtx.range("SampleRays")
    def __call__(self, inputs: MultiviewBatch, all_data_features):
        device = inputs['rays'].origins.device
        ray_idx = torch.randint(0, inputs['rays'].shape[0], [self.num_samples], device=device)

        out = {}
        out['rays'] = inputs['rays'][ray_idx].contiguous()

        # Get correspondences ("for loop" instead of torch vectorization due to fear of memory limitation from calculating ~1000 multiplied by HxW):
        N = all_data_features.shape[0]
        corrs = torch.zeros((ray_idx.shape[0], N - 1), dtype=torch.int8)
        # TODO: save the correpondences for each calculated point for both sides, for later usage, in the dataset "cache" for that.
        for i in ray_idx:
            corrs[i] = return_correspondences(out['ft'][i], all_data_features)

        # Loop over ray values in this batch
        for channel_name, ray_value in inputs.ray_values().items():
            out[channel_name] = ray_value[ray_idx].contiguous()

        out['correspondences'] = corrs.contigous()

        return out


class SampleRaysWithSparseChannel:
    """ A dataset transform for sub-sampling a fixed amount of rays.
     Given a sparse channel that requires all of the rays with this
     channel to be sampled, it will sample all of its rays"""
    def __init__(self, num_samples, sparse_channel, force_rgb_random):
        self.num_samples = num_samples
        self.sparse_channel = sparse_channel
        self.force_rgb_random = force_rgb_random

    @torch.cuda.nvtx.range("SampleRays")
    def __call__(self, inputs: MultiviewBatch):
        """
        Sample randomly the rays, when:
        1. sample all of the sparse_channel non-null rays
        2. Make sure that the sparse_channel induced sampled rays
         are less than a half of the desired rays to regularize
         biasing the training towards this sparse channel. (If assertion brakes need to decide how to handle it)
        3. The first rays it outputs are those of the sparse channel (it's in a single batch so it shouldn't matter)
        Args:
            inputs:

        Returns:

        """
        device = inputs['rays'].origins.device
        sparse_channel_value = inputs[self.sparse_channel]
        sprase_input_idx = torch.argwhere(~torch.isnan(sparse_channel_value[:,0]))[:,0]
        sprase_input_idx_n = sprase_input_idx.shape[0]

        ray_idx = torch.randint(0, inputs['rays'].shape[0], [self.num_samples - sprase_input_idx_n], device=device)

        sample_idx = torch.concat([sprase_input_idx, ray_idx])

        out = {}
        out['rays'] = inputs['rays'][sample_idx].contiguous()
        # Loop over ray values in this batch
        for channel_name, ray_value in inputs.ray_values().items():
            out[channel_name] = ray_value[sample_idx].contiguous()
            # To make sure no one uses the rgb from the rays we sampled
            # not in the random sampling but from the sparse input channel,
            # to prevent bias toward this channel in the independent rgb loss
            if channel_name != self.sparse_channel and self.force_rgb_random:
                s_idx_not_randomly_sampled = torch.argwhere(~torch.isin(sprase_input_idx, ray_idx))
                out[channel_name][s_idx_not_randomly_sampled] = torch.nan
        return out