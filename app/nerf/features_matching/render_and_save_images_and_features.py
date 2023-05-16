# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
import argparse
import logging
import sys

import cv2
import numpy as np
import torch
from PIL.Image import Image
from tqdm import tqdm
from wisp.core import RenderBuffer

import wisp
from wisp.app_utils import default_log_setup, args_to_log_format
import wisp.config_parser as config_parser
from wisp.framework import WispState
from wisp.datasets import MultiviewDataset, SampleRays, SampleRaysWithSparseChannel, MultiviewBatch
from wisp.models.grids import BLASGrid, OctreeGrid, CodebookOctreeGrid, TriplanarGrid, HashGrid
from wisp.ops.image import write_png
from wisp.tracers import BaseTracer, PackedRFTracer
from wisp.models.nefs import BaseNeuralField, NeuralRadianceFieldWithFeaturesChannel
from wisp.models.pipeline import Pipeline
from wisp.trainers import BaseTrainer, MultiviewTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.trainers.multiview_trainer import MultiviewWithSparseDepthGtTrainer


def parse_args():
    """Wisp mains define args per app.
    Args are collected by priority: cli args > config yaml > argparse defaults
    For convenience, args are divided into groups.
    """
    parser = argparse.ArgumentParser(description='A script for training simple NeRF variants.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str,
                        help='Path to config file to replace defaults.')
    parser.add_argument('--profile', action='store_true',
                        help='Enable NVTX profiling')

    log_group = parser.add_argument_group('logging')
    log_group.add_argument('--exp-name', type=str,
                           help='Experiment name, unique id for trainers, logs.')
    log_group.add_argument('--log-level', action='store', type=int, default=logging.INFO,
                           help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')
    log_group.add_argument('--perf', action='store_true', default=False,
                           help='Use high-level profiling for the trainer.')

    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--dataset-path', type=str,
                            help='Path to the dataset')
    data_group.add_argument('--dataset-num-workers', type=int, default=-1,
                            help='Number of workers for dataset preprocessing, if it supports multiprocessing. '
                                 '-1 indicates no multiprocessing.')
    data_group.add_argument('--dataloader-num-workers', type=int, default=0,
                            help='Number of workers for dataloader.')
    data_group.add_argument('--bg-color', default='black' if is_interactive() else 'white',
                            choices=['white', 'black'], help='Background color')
    data_group.add_argument('--multiview-dataset-format', default='standard', choices=['standard', 'standard_with_colmap', 'rtmv'],
                            help='Data format for the transforms')
    data_group.add_argument('--num-rays-sampled-per-img', type=int, default='4096',
                            help='Number of rays to sample per image')
    data_group.add_argument('--force-rgb-random', action='store_true', default=False,
                            help='When sampling the rays and depth gt rays, do not use the rgb of the non-randomly'
                                 ' sampled depth gt rays')
    data_group.add_argument('--mip', type=int, default=None,
                            help='MIP level of ground truth image')
    data_group.add_argument('--colmap-results-path', type=str, default=None,
                            help='path to the results of COLMAP over the dataset')

    grid_group = parser.add_argument_group('grid')
    grid_group.add_argument('--grid-type', type=str, default='OctreeGrid',
                            choices=config_parser.list_modules('grid'),
                            help='Type of to use, i.e.:'
                                 '"OctreeGrid", "CodebookOctreeGrid", "TriplanarGrid", "HashGrid".'
                                 'Grids are located in `wisp.models.grids`')
    grid_group.add_argument('--interpolation-type', type=str, default='linear', choices=['linear', 'closest'],
                            help='Interpolation type to use for samples within grids.'
                                 'For a 3D grid structure, linear uses trilinear interpolation of 8 cell nodes,'
                                 'closest uses the nearest neighbor.')
    grid_group.add_argument('--blas-type', type=str, default='octree',  # TODO(operel)
                            choices=['octree',],
                            help='Type of acceleration structure to use for fast raymarch occupancy queries.')
    grid_group.add_argument('--multiscale-type', type=str, default='sum', choices=['sum', 'cat'],
                            help='Aggregation of choice for multi-level grids, for features from different LODs.')
    grid_group.add_argument('--feature-dim', type=int, default=32,
                            help='Dimensionality for features stored within the grid nodes.')
    grid_group.add_argument('--feature-std', type=float, default=0.0,
                            help='Grid initialization: standard deviation used for randomly sampling initial features.')
    grid_group.add_argument('--feature-bias', type=float, default=0.0,
                            help='Grid initialization: bias used for randomly sampling initial features.')
    grid_group.add_argument('--base-lod', type=int, default=2,
                            help='Number of levels in grid, which book-keep occupancy but not features.'
                                 'The total number of levels in a grid is `base_lod + num_lod - 1`')
    grid_group.add_argument('--num-lods', type=int, default=1,
                            help='Number of levels in grid, which store concrete features.')
    grid_group.add_argument('--codebook-bitwidth', type=int, default=8,
                            help='For Codebook and HashGrids only: determines the table size as 2**(bitwidth).')
    grid_group.add_argument('--tree-type', type=str, default='geometric', choices=['geometric', 'quad'],
                            help='For HashGrids only: how the resolution of the grid is determined. '
                                 '"geometric" uses the geometric sequence initialization from InstantNGP,'
                                 'where "quad" uses an octree sampling pattern.')
    grid_group.add_argument('--min-grid-res', type=int, default=16,
                            help='For HashGrids only: min grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--max-grid-res', type=int, default=2048,
                            help='For HashGrids only: max grid resolution, used only in geometric initialization mode')
    grid_group.add_argument('--prune-min-density', type=float, default=(0.01 * 512) / np.sqrt(3),
                            help='For HashGrids only: Minimum density value for pruning')
    grid_group.add_argument('--prune-density-decay', type=float, default=0.6,
                            help='For HashGrids only: The decay applied on the density every pruning')
    grid_group.add_argument('--blas-level', type=float, default=7,
                            help='For HashGrids only: Determines the number of levels in the acceleration structure '
                                 'used to track the occupancy status (bottom level acceleration structure).')

    nef_group = parser.add_argument_group('nef')
    nef_group.add_argument('--pos-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode input coordinates'
                                'or view directions.')
    nef_group.add_argument('--view-embedder', type=str, choices=['none', 'identity', 'positional'],
                           default='positional',
                           help='MLP Decoder of neural field: Positional embedder used to encode view direction')
    nef_group.add_argument('--position-input', type=bool, default=False,
                           help='If True, position coords will be concatenated to the '
                                'features / positional embeddings when fed into the decoder.')
    nef_group.add_argument('--pos-multires', type=int, default=10,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of input coordinates')
    nef_group.add_argument('--view-multires', type=int, default=4,
                           help='MLP Decoder of neural field: Number of frequencies to use for positional encoding'
                                'of view direction')
    nef_group.add_argument('--layer-type', type=str, default='none',
                           choices=['none', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'])
    nef_group.add_argument('--activation-type', type=str, default='relu',
                           choices=['relu', 'sin', 'leaky_relu'])
    nef_group.add_argument('--hidden-dim', type=int, help='MLP Decoder of neural field: width of all hidden layers.')
    nef_group.add_argument('--num-layers', type=int, help='MLP Decoder of neural field: number of hidden layers.')
    nef_group.add_argument('--extra-channels', type=str, choices=['grid_features'],
                           default='grid_features',
                           help='Extra channel of grid_features if desired')

    tracer_group = parser.add_argument_group('tracer')
    tracer_group.add_argument('--raymarch-type', type=str, choices=['ray', 'voxel'], default='ray',
                              help='Marching algorithm to use when generating samples along rays in tracers.'
                                   '`ray` samples fixed amount of randomized `num_steps` along the ray.'
                                   '`voxel` samples `num_steps` samples in each cell the ray intersects.')
    tracer_group.add_argument('--num-steps', type=int, default=1024,
                              help='Number of samples to generate along traced rays. See --raymarch-type for '
                                   'algorithm used to generate the samples.')

    trainer_group = parser.add_argument_group('trainer')
    trainer_group.add_argument('--epochs', type=int, default=250,
                               help='Number of epochs to run the training.')
    trainer_group.add_argument('--batch-size', type=int, default=512,
                               help='Batch size for the training.')
    trainer_group.add_argument('--disable-amp', action='store_true',
                               help='Disabling the mixed precision training.')
    trainer_group.add_argument('--resample', action='store_true',
                               help='Resample the dataset after every epoch.')
    trainer_group.add_argument('--only-last', action='store_true',
                               help='Train only last LOD.')
    trainer_group.add_argument('--resample-every', type=int, default=1,
                               help='Resample every N epochs')
    trainer_group.add_argument('--model-format', type=str, default='full', choices=['full', 'state_dict'],
                               help='Format in which to save models.')
    trainer_group.add_argument('--pretrained', type=str,
                               help='Path to pretrained model weights.')
    trainer_group.add_argument('--save-as-new', action='store_true',
                               help='Save the model at every epoch (no overwrite).')
    trainer_group.add_argument('--save-every', type=int, default=(-1 if is_interactive() else 5),
                               help='Save the model at every N epoch.')
    trainer_group.add_argument('--render-tb-every', type=int, default=(-1 if is_interactive() else 5),
                               help='Render every N epochs')
    trainer_group.add_argument('--log-tb-every', type=int, default=5,  # TODO (operel): move to logging
                               help='Render to tensorboard every N epochs')
    trainer_group.add_argument('--log-dir', type=str, default='_results/logs/runs/',
                               help='Log file directory for checkpoints.')
    trainer_group.add_argument('--prune-every', type=int, default=-1,
                               help='Prune every N epochs')
    trainer_group.add_argument('--grow-every', type=int, default=-1,
                               help='Grow network every X epochs')
    trainer_group.add_argument('--growth-strategy', type=str, default='increase',
                               choices=['onebyone',      # One by one trains one level at a time.
                                        'increase',      # Increase starts from [0] and ends up at [0,...,N]
                                        'shrink',        # Shrink strats from [0,...,N] and ends up at [N]
                                        'finetocoarse',  # Fine to coarse starts from [N] and ends up at [0,...,N]
                                        'onlylast'],     # Only last starts and ends at [N]
                               help='Strategy for coarse-to-fine training')
    trainer_group.add_argument('--valid-only', action='store_true',
                               help='Run validation only (and do not run training).')
    trainer_group.add_argument('--valid-every', type=int, default=-1,
                               help='Frequency of running validation.')
    trainer_group.add_argument('--random-lod', action='store_true',
                               help='Use random lods to train.')
    trainer_group.add_argument('--wandb-project', type=str, default=None,
                               help='Weights & Biases Project')
    trainer_group.add_argument('--wandb-run-name', type=str, default=None,
                               help='Weights & Biases Run Name')
    trainer_group.add_argument('--wandb-entity', type=str, default=None,
                               help='Weights & Biases Entity')
    trainer_group.add_argument('--wandb-viz-nerf-angles', type=int, default=20,
                               help='Number of Angles to visualize a scene on Weights & Biases. '
                                    'Set this to 0 to disable 360 degree visualizations.')
    trainer_group.add_argument('--wandb-viz-nerf-distance', type=int, default=3,
                               help='Distance to visualize Scene from on Weights & Biases')

    optimizer_group = parser.add_argument_group('optimizer')
    optimizer_group.add_argument('--optimizer-type', type=str, default='adam',
                                 choices=config_parser.list_modules('optim'),
                                 help='Optimizer to be used, includes optimizer modules available within `torch.optim` '
                                      'and fused optimizers from `apex`, if apex is installed.')
    optimizer_group.add_argument('--lr', type=float, default=0.001,
                                 help='Base optimizer learning rate.')
    optimizer_group.add_argument('--eps', type=float, default=1e-8,
                                 help='Eps value for numerical stability.')
    optimizer_group.add_argument('--weight-decay', type=float, default=0,
                                 help='Weight decay, applied only to decoder weights.')
    optimizer_group.add_argument('--grid-lr-weight', type=float, default=100.0,
                                 help='Relative learning rate weighting applied only for the grid parameters'
                                      '(e.g. parameters which contain "grid" in their name)')
    optimizer_group.add_argument('--rgb-loss-lambda', type=float, default=1.0,
                                 help='Weight of rgb loss')
    optimizer_group.add_argument('--depth-loss-lambda', type=float, default=0.,
                                 help='Weight of rgb loss')

    # Evaluation renderer (definitions do not affect interactive renderer)
    offline_renderer_group = parser.add_argument_group('renderer')
    offline_renderer_group.add_argument('--render-res', type=int, nargs=2, default=[512, 512],
                                        help='Width/height to render at.')
    offline_renderer_group.add_argument('--render-batch', type=int, default=0,
                                        help='Batch size (in number of rays) for batched rendering.')
    offline_renderer_group.add_argument('--camera-origin', type=float, nargs=3, default=[-2.8, 2.8, -2.8],
                                        help='Camera origin.')
    offline_renderer_group.add_argument('--camera-lookat', type=float, nargs=3, default=[0, 0, 0],
                                        help='Camera look-at/target point.')
    offline_renderer_group.add_argument('--camera-fov', type=float, default=30,
                                        help='Camera field of view (FOV).')
    offline_renderer_group.add_argument('--camera-proj', type=str, choices=['ortho', 'persp'], default='persp',
                                        help='Camera projection.')
    offline_renderer_group.add_argument('--camera-clamp', nargs=2, type=float, default=[0, 10],
                                        help='Camera clipping bounds.')
    offline_renderer_group.add_argument('--channels', nargs=3, type=str, default=['rgb', 'depth','grid_features'],
                                        help='Camera clipping bounds.')

    # Parse CLI args & config files
    args = config_parser.parse_args(parser)

    # Also obtain args as grouped hierarchy, useful for, i.e., logging
    args_dict = config_parser.get_grouped_args(parser, args)
    return args, args_dict


def load_dataset(args) -> MultiviewDataset:
    """ Loads a multiview dataset comprising of pairs of images and calibrated cameras.
    The types of supported datasets are defined by multiview_dataset_format:
    'standard' - refers to the standard NeRF format popularized by Mildenhall et al. 2020,
                 including additions to the metadata format added by Muller et al. 2022.
    'rtmv' - refers to the dataset published by Tremblay et. al 2022,
            "RTMV: A Ray-Traced Multi-View Synthetic Dataset for Novel View Synthesis".
            This dataset includes depth information which allows for performance improving optimizations in some cases.
    """
    transform = SampleRaysWithSparseChannel(num_samples=args.num_rays_sampled_per_img, sparse_channel='gt_depth', force_rgb_random=args.force_rgb_random)
    train_dataset = wisp.datasets.NeRFSyntheticDatasetWithCOLMAP(dataset_path=args.dataset_path,
                                                         colmap_res_path=args.colmap_results_path,
                                                         split='train',
                                                         mip=args.mip,
                                                         bg_color=args.bg_color,
                                                         dataset_num_workers=args.dataset_num_workers,
                                                         transform=transform)
    validation_dataset = None
    return train_dataset, validation_dataset


def load_grid(args, dataset: MultiviewDataset) -> BLASGrid:
    """ Wisp's implementation of NeRF uses feature grids to improve the performance and quality (allowing therefore,
    interactivity).
    This function loads the feature grid to use within the neural pipeline.
    Grid choices are interesting to explore, so we leave the exact backbone type configurable,
    and show how grid instances may be explicitly constructed.
    Grids choices, for example, are: OctreeGrid, TriplanarGrid, HashGrid, CodebookOctreeGrid
    See corresponding grid constructors for each of their arg details.
    """
    grid = None

    # Optimization: For octrees based grids, if dataset contains depth info, initialize only cells known to be occupied
    if args.grid_type == "OctreeGrid":
        if dataset.supports_depth():
            grid = OctreeGrid.from_pointcloud(
                pointcloud=dataset.as_pointcloud(),
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
            )
        else:
            grid = OctreeGrid.make_dense(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
            )
    elif args.grid_type == "CodebookOctreeGrid":
        if dataset.supports_depth:
            grid = CodebookOctreeGrid.from_pointcloud(
                pointcloud=dataset.as_pointcloud(),
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth
            )
        else:
            grid = CodebookOctreeGrid.make_dense(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                interpolation_type=args.interpolation_type,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth
            )
    elif args.grid_type == "TriplanarGrid":
        grid = TriplanarGrid(
            feature_dim=args.feature_dim,
            base_lod=args.base_lod,
            num_lods=args.num_lods,
            interpolation_type=args.interpolation_type,
            multiscale_type=args.multiscale_type,
            feature_std=args.feature_std,
            feature_bias=args.feature_bias,
        )
    elif args.grid_type == "HashGrid":
        # "geometric" - determines the resolution of the grid using geometric sequence initialization from InstantNGP,
        if args.tree_type == "geometric":
            grid = HashGrid.from_geometric(
                feature_dim=args.feature_dim,
                num_lods=args.num_lods,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth,
                min_grid_res=args.min_grid_res,
                max_grid_res=args.max_grid_res,
                blas_level=args.blas_level
            )
        # "quad" - determines the resolution of the grid using an octree sampling pattern.
        elif args.tree_type == "octree":
            grid = HashGrid.from_octree(
                feature_dim=args.feature_dim,
                base_lod=args.base_lod,
                num_lods=args.num_lods,
                multiscale_type=args.multiscale_type,
                feature_std=args.feature_std,
                feature_bias=args.feature_bias,
                codebook_bitwidth=args.codebook_bitwidth,
                blas_level=args.blas_level
            )
    else:
        raise ValueError(f"Unknown grid_type argument: {args.grid_type}")
    return grid


def load_neural_field(args, dataset: MultiviewDataset) -> BaseNeuralField:
    """ Creates a "Neural Field" instance which converts input coordinates to some output signal.
    Here a NeuralRadianceField is created, which maps 3D coordinates (+ 2D view direction) -> RGB + density.
    The NeuralRadianceField uses spatial feature grids internally for faster feature interpolation and raymarching.
    """
    grid = load_grid(args=args, dataset=dataset)
    nef = NeuralRadianceFieldWithFeaturesChannel(
        grid=grid,
        pos_embedder=args.pos_embedder,
        view_embedder=args.view_embedder,
        position_input=args.position_input,
        pos_multires=args.pos_multires,
        view_multires=args.view_multires,
        activation_type=args.activation_type,
        layer_type=args.layer_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        prune_density_decay=args.prune_density_decay,   # Used only for grid types which support pruning
        prune_min_density=args.prune_min_density        # Used only for grid types which support pruning
    )
    return nef


def load_tracer(args) -> BaseTracer:
    """ Wisp "Tracers" are responsible for taking input rays, marching them through the neural field to render
    an output RenderBuffer.
    Wisp's implementation of NeRF uses the PackedRFTracer to trace the neural field:
    - Packed: each ray yields a custom number of samples, which are therefore packed in a flat form within a tensor,
     see: https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.batch.html#packed
    - RF: Radiance Field
    PackedRFTracer is employed within the training loop, and is responsible for making use of the neural field's
    grid to generate samples and decode them to pixel values.
    """
    tracer = PackedRFTracer(
        raymarch_type=args.raymarch_type,   # Chooses the ray-marching algorithm
        num_steps=args.num_steps,           # Number of steps depends on raymarch_type
        bg_color=args.bg_color
    )
    return tracer


def load_neural_pipeline(args, dataset, device) -> Pipeline:
    """ In Wisp, a Pipeline comprises of a neural field + a tracer (the latter is optional in some cases).
    Together, they form the complete pipeline required to render a neural primitive from input rays / coordinates.
    """
    nef = load_neural_field(args=args, dataset=dataset)
    tracer = load_tracer(args=args)
    pipeline = Pipeline(nef=nef, tracer=tracer)
    if args.pretrained:
        if args.model_format == "full":
            pipeline = torch.load(args.pretrained)
        else:
            pipeline.load_state_dict(torch.load(args.pretrained))
    pipeline.to(device)
    return pipeline


def load_trainer(pipeline, train_dataset, validation_dataset, device, scene_state, args, args_dict) -> BaseTrainer:
    """ Loads the NeRF trainer.
    The trainer is responsible for managing the optimization life-cycles and can be operated in 2 modes:
    - Headless, which will run the train() function until all training steps are exhausted.
    - Interactive mode, which uses the gui. In this case, an OptimizationApp uses events to prompt the trainer to
      take training steps, while also taking care to render output to users (see: iterate()).
      In interactive mode, trainers can also share information with the app through the scene_state (WispState object).
    """
    # args.optimizer_type is the name of some optimizer class (from torch.optim or apex),
    # Wisp's config_parser is able to pick this app's args with corresponding names to the optimizer constructor args.
    # The actual construction of the optimizer instance happens within the trainer.
    optimizer_cls = config_parser.get_module(name=args.optimizer_type)
    optimizer_params = config_parser.get_args_for_function(args, optimizer_cls)

    trainer = MultiviewWithSparseDepthGtTrainer(pipeline=pipeline,
                               train_dataset=train_dataset,
                               validation_dataset=validation_dataset,
                               num_epochs=args.epochs,
                               batch_size=args.batch_size,
                               optim_cls=optimizer_cls,
                               lr=args.lr,
                               weight_decay=args.weight_decay,
                               grid_lr_weight=args.grid_lr_weight,
                               optim_params=optimizer_params,
                               log_dir=args.log_dir,
                               device=device,
                               exp_name=args.wandb_run_name if args.wandb_run_name is not None else args.exp_name,
                               info=args_to_log_format(args_dict),
                               extra_args=vars(args),
                               render_tb_every=args.render_tb_every,
                               save_every=args.save_every,
                               scene_state=scene_state,
                               trainer_mode='validate' if args.valid_only else 'train',
                               using_wandb=args.wandb_project is not None,
                               enable_amp=not args.disable_amp)
    return trainer


def load_app(args, scene_state, trainer):
    """ Used only in interactive mode. Creates an interactive app, which employs a renderer which displays
    the latest information from the trainer (see: OptimizationApp).
    The OptimizationApp can be customized or further extend to support even more functionality.
    """
    if not is_interactive():
        logging.info("Running headless. For the app, set $WISP_HEADLESS=0.")
        return None  # Interactive mode is disabled
    else:
        from wisp.renderer.app.optimization_app import OptimizationApp
        scene_state.renderer.device = trainer.device  # Use same device for trainer and app renderer
        app = OptimizationApp(wisp_state=scene_state,
                              trainer_step_func=trainer.iterate,
                              experiment_name="wisp trainer")
        return app


def is_interactive() -> bool:
    """ Returns True if interactive mode with gui is on, False is HEADLESS mode is forced """
    return os.environ.get('WISP_HEADLESS') != '1'

if __name__ == '__main__':
    insert_args_to_cli = [
        # '--dataset-path', '/mnt/more_space/datasets/nerf_llff_data/fern_in_nerf_format_2_views/',
        '--dataset-path', '/home/galharari/datasets/nerf_llff_data/fern_in_nerf_format_5_views/',
        '--config', 'app/nerf/configs/nerf_hash.yaml',
        '--wandb-project', 'wisp_playing',
        '--wandb-run-name', 'reg_prune_with_depthloss',
        '--wandb-viz-nerf-distance', '2',
        '--epochs', '150',
        '--num-rays-sampled-per-img', '8192',
        '--multiview-dataset-format', 'standard_with_colmap',
        # '--colmap-results-path', '/mnt/more_space/datasets/nerf_llff_data/fern',
        '--colmap-results-path', '/home/galharari/datasets/nerf_llff_data/fern',
        '--depth-loss-lambda', '0.2',
        '--force-rgb-random',
        '--valid-every', '30',
        '--prune-every', '20',
        '--exp-name', 'debug_loss',
        '--activation-type', 'leaky_relu',
        '--disable-amp',
        # '--pretrained', '_results/logs/runs/reg_prune_with_depthloss/20230425-194307/model.pth'
        '--pretrained', '_results/logs/runs/5_views_low_depthloss_with_cosine_loss/20230501-010127/model.pth'
    ]
    for arg_to_add in insert_args_to_cli:
        if arg_to_add.split(' ')[0] in sys.argv:
            continue
        sys.argv.append(arg_to_add)
    args, args_dict = parse_args()  # Obtain args by priority: cli args > config yaml > argparse defaults
    default_log_setup(args.log_level)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset, validation_dataset = load_dataset(args=args)
    pipeline_with_features = torch.load('_results/logs/runs/dsnerf_high_cosine_loss/20230515-221556/model.pth')
    scene_state = WispState()   # Joint trainer / app state
    trainer = load_trainer(pipeline=pipeline_with_features,
                           train_dataset=train_dataset, validation_dataset=validation_dataset,
                           device=device, scene_state=scene_state,
                           args=args, args_dict=args_dict)

    trainer.pipeline.eval()

    logging.info("Beginning rendering...")
    img_shape = trainer.train_dataset.img_shape
    logging.info(f"Running rendering on dataset with {len(trainer.train_dataset)} images "
                 f"at resolution {img_shape[0]}x{img_shape[1]}")

    trainer.valid_log_dir = os.path.join(trainer.log_dir, "renders_llff")
    logging.info(f"Saving rendering result to {trainer.valid_log_dir}")


    lods = list(range(trainer.pipeline.nef.grid.num_lods))

    pips_model = None
    dataset = trainer.train_dataset
    lod_idx = lods[-1]
    name = f"lod{lods[-1]}"

    img_count = len(dataset)
    img_shape = dataset.img_shape
    res_dir = 'app/nerf/features_matching/renders_output'

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            # Manually extract the image to avoid the sampleRays transform
            full_batch = MultiviewBatch(
                rays=dataset.data["rays"][idx],
                rgb=dataset.data["rgb"][idx],
                masks=dataset.data["masks"][idx],
                gt_depth=dataset.data["gt_depth"][idx]
            )
            img_name = list(dataset.data['cameras'].keys())[idx]

            rgb_gt = full_batch['rgb'].to('cuda')
            depth_gt = full_batch['gt_depth']
            rays = full_batch['rays'].to('cuda')
            rb = trainer.renderer.render(trainer.pipeline, rays, lod_idx=lod_idx)

            rgb_gt = rgb_gt.reshape(*img_shape, -1)
            depth_gt = depth_gt.reshape(*img_shape, -1)
            rb = rb.reshape(*img_shape, -1)

            out_rb = RenderBuffer(rgb=rb.rgb, depth=rb.depth, alpha=rb.alpha, \
                                  gts=rgb_gt, err=(rgb_gt[..., :3] - rb.rgb[..., :3]) ** 2)

            out_name = f"out_{idx}"
            in_name = f"in_{idx}"
            torch.save(rb.cpu().grid_features, os.path.join(res_dir, out_name + "_features" + ".pt"))
            torch.save(rb.cpu().rgb, os.path.join(res_dir, out_name + "_image" + ".pt"))
            torch.save(rgb_gt.cpu(), os.path.join(res_dir, in_name + "_image" + ".pt"))
            torch.save(depth_gt.cpu(), os.path.join(res_dir, in_name + "_gt_depth" + ".pt"))
            torch.save(rb.cpu().depth, os.path.join(res_dir, out_name + "_depth" + ".pt"))

            write_png(os.path.join(res_dir, in_name + img_name + ".png"), torch.tensor((rgb_gt.cpu() * 255), dtype=torch.uint8))
            write_png(os.path.join(res_dir, out_name + img_name + ".png"), out_rb.cpu().image().byte().rgb)