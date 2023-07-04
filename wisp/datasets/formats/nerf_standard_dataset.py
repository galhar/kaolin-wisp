# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations

import gc
import os
import re
import glob
import warnings

import cv2
import json
from tqdm import tqdm
import logging as log
from typing import Callable, List, Dict, Union
import numpy as np
import torch
from torch.multiprocessing import Pool
from kaolin.render.camera import Camera, blender_coords
from wisp.ops.pointcloud import create_pointcloud_from_images, normalize_pointcloud, create_edges_pointcloud_from_rays

from wisp.core import Rays
from wisp.datasets.load_llff import load_colmap_depth
from wisp.ops.raygen import generate_pinhole_rays, generate_ortho_rays, generate_centered_pixel_coords
from wisp.ops.image import resize_mip, load_rgb
from wisp.datasets.base_datasets import MultiviewDataset
from wisp.datasets.batch import MultiviewBatch, MultiviewBatchWithColmap
from wisp.datasets.dift.models.dift_sd import SDFeaturizer



class NeRFSyntheticDataset(MultiviewDataset):
    """ A dataset for files in the standard NeRF format, including extensions to the format
        supported by Instant Neural Graphics Primitives.
        See: https://github.com/NVlabs/instant-ngp
        NeRF-synthetic scenes include RGBA / RGB information.
    """

    def __init__(self, dataset_path: str, split: str, bg_color: str, mip: int = 0,
                 dataset_num_workers: int = -1, transform: Callable = None, coords_norm_factors=[None, None]):
        """ Loads the NeRF-synthetic data and applies dataset specific transforms required for compatibility with the
        framework.
        The loaded data is cached inside the `data` field.

        Args:
            dataset_path (str): The root directory of the dataset, where images and json files of a single multiview
                scene reside.
            split (str): The dataset split to use, corresponding to the transform file to load, when splits are
                available. In case of a single transform file, it will be considered as a single split of
                'train' by default.
                Options: 'train', 'val', 'test'.
            bg_color (str): The background color to use for when alpha=0.
                Options: 'black', 'white'.
            mip (int): If provided, will rescale images by 2**mip. Useful when large images are loaded.
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        super().__init__(dataset_path=dataset_path, dataset_num_workers=dataset_num_workers,
                         transform=transform, split=split)
        self.coords_center, self.coords_scale = coords_norm_factors
        self.mip = mip
        self.bg_color = bg_color

        self.coords = self.data = None
        self._transform_file = self._validate_and_find_transform()
        self.data = self.load()

        self._img_shape = self.data["rgb"].shape[1:3]
        self.flatten_tensors()

    def create_split(self, split: str, transform: Callable = None) -> NeRFSyntheticDataset:
        """ Creates a dataset with the same parameters and a different split.
        This is a convenient way of creating validation and test datasets, while making sure they're compatible
        with the train dataset.

        All settings except for split and transform will be copied from the current dataset.

        Args:
            split (str): The dataset split to use, corresponding to the transform file to load.
                Options: 'train', 'val', 'test'.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        return NeRFSyntheticDataset(
            dataset_path=self.dataset_path,
            split=split,
            bg_color=self.bg_color,
            mip=self.mip,
            dataset_num_workers=self.dataset_num_workers,
            transform=transform,
            coords_norm_factors=[self.coords_center, self.coords_scale]
        )

    def __getitem__(self, idx) -> MultiviewBatch:
        """Retrieve a batch of rays and their corresponding values.
        Rays are precomputed from the dataset's cameras, and are cached within the dataset.
        By default, rays are assumed to have corresponding rgb values, sampled from the dataset's images.

        Returns:
            (MultiviewBatch): A batch of rays and their rgb values. The fields can be accessed as a dictionary:
                "rays" - a wisp.core.Rays pack of ray origins and directions, pre-generated from the dataset camera.
                "rgb" - a torch.Tensor of rgb color which corresponds the gt image's pixel each ray intersects.
                "masks" - a torch.BoolTensor specifying if the ray hits a dense area or not.
                 This is estimated from the alpha channel of the gt image, where mask=True if alpha > 0.5.
        """
        out = MultiviewBatch(
            rays=self.data["rays"][idx],
            rgb=self.data["rgb"][idx],
            masks=self.data["masks"][idx]
        )

        if self.transform is not None:
            out = self.transform(out)

        return out

    @classmethod
    def is_root_of_dataset(cls, root: str, files_list: List[str]) -> bool:
        """ Each dataset may implement a simple set of rules to distinguish it from other datasets.
        Rules should be unique for this dataset type, such that given a general root path, Wisp will know
        to associate it with this dataset class.

        Datasets which don't implement this function should be created explicitly.

        Args:
                root (str): A path to the root directory of the dataset.
                files_list (List[str]): List of files within the dataset root, without their prefix path.
        Returns:
                True if the root folder points to content loadable by this dataset.
        """
        # NeRF-synthetic data is distinguished by the transform jsons and an additional image folder.
        try:
            regex = re.compile(r"transform.+\.json")
            transform_files = list(filter(regex.match, files_list))
            if 'transforms.json' in transform_files and 'images' in files_list:
                # Single transform file
                images_path = os.path.join(root, 'images')
                return os.path.isdir(images_path)
            elif 'transforms_train.json' in transform_files and 'train' in files_list:
                # Three transform files, with splits. Check only for train split
                train_path = os.path.join(root, 'train')
                return os.path.isdir(train_path)
            else:
                return False
        except ValueError:
            return False

    def _validate_and_find_transform(self) -> str:
        """
        Validates the file structure and returns the filename for the dataset's split / transform.
        There are two pairs of standard file structures this dataset can parse:

        ```
        /path/to/dataset/transform.json
        /path/to/dataset/images/____.png
        ```

        or

        ```
        /path/to/dataset/transform_{split}.json
        /path/to/dataset/{split}/_____.png
        ```

        In the former case, the single transform file is assumed to be loaded as a train set,
        for the latter split is assumed to be any of: 'train', 'val', 'test'.
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"NeRF dataset path does not exist: {self.dataset_path}")

        transforms = sorted(glob.glob(os.path.join(self.dataset_path, "*.json")))
        if len(transforms) == 0:
            raise RuntimeError(f"NeRF dataset folder has no transform *.json files with camera data: {self.dataset_path}")
        elif len(transforms) > 3 or len(transforms) == 2:
            raise RuntimeError(f"NeRF dataset folder has an unsupported number of splits, "
                               f"there should be ['test', 'train', 'val'], but found: {transforms}.")
        transform_dict = {}
        if len(transforms) == 1:
            transform_dict['train'] = transforms[0]
        elif len(transforms) == 3:
            fnames = [os.path.basename(transform) for transform in transforms]

            # Create dictionary of split to file path, probably there is simpler way of doing this
            for _split in ['test', 'train', 'val']:
                for i, fname in enumerate(fnames):
                    if _split in fname:
                        transform_dict[_split] = transforms[i]

        if self.split not in transform_dict:
            log.warning(
                f"WARNING: Split type ['{self.split}'] does not exist in the dataset. Falling back to train data.")
            self.split = 'train'
        return transform_dict[self.split]

    @staticmethod
    def _load_single_entry(frame, root, mip=None):
        """ Loads a single image: takes a frame from the JSON to load image and associated poses from json.
        This is a helper function which also supports multiprocessing for the standard dataset.

        Args:
            root (str): The root of the dataset.
            frame (dict): The frame object from the transform.json. The frame contains the metadata.
            mip (int): Optional, If set, rescales the image by 2**mip.

        Returns:
            (dict): Dictionary of the image and pose.
        """
        fpath = os.path.join(root, frame['file_path'].replace("\\", "/"))

        basename = os.path.basename(os.path.splitext(fpath)[0])
        if os.path.splitext(fpath)[1] == "":
            # Assume PNG file if no extension exists... the NeRF synthetic data follows this convention.
            fpath += '.png'

        # For some reason instant-ngp allows missing images that exist in the transform but not in the data.
        # Handle this... also handles the above case well too.
        if os.path.exists(fpath):
            img = load_rgb(fpath)
            if mip is not None:
                img = resize_mip(img, mip, interpolation=cv2.INTER_AREA)
            return dict(basename=basename,
                        img=torch.FloatTensor(img), pose=torch.FloatTensor(np.array(frame['transform_matrix'])))
        else:
            # log.info(f"File name {fpath} doesn't exist. Ignoring.")
            return None

    def load_singleprocess(self):
        """Standard parsing function for loading nerf-synthetic files on the main process.
        This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

        Returns:
            (dict of torch.FloatTensors): Channels of information from NeRF:
                - 'rays': a list of ray packs, each entry corresponds to a single camera view
                - 'rgb', 'masks': a list of torch.Tensors, each entry corresponds to a single gt image
                - 'cameras': a list of Camera objects, one camera per view
        """
        with open(self._transform_file, 'r') as f:
            metadata = json.load(f)

        imgs = []
        poses = []
        basenames = []

        for frame in tqdm(metadata['frames'], desc='loading data'):
            _data = self._load_single_entry(frame, self.dataset_path, mip=self.mip)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])

        return self._collect_data_entries(metadata=metadata, basenames=basenames, imgs=imgs, poses=poses)

    @staticmethod
    def _parallel_load_standard_imgs(args):
        """ Internal function used by the multiprocessing loader: allocates a single entry task for a worker.
        """
        torch.set_num_threads(1)
        result = NeRFSyntheticDataset._load_single_entry(args['frame'], args['root'], mip=args['mip'])
        if result is None:
            return dict(basename=None, img=None, pose=None)
        else:
            return dict(basename=result['basename'], img=result['img'], pose=result['pose'])

    def load_multiprocess(self):
        """Standard parsing function for loading nerf-synthetic files with multiple workers.
        This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

        Returns:
            (dict of torch.FloatTensors): Channels of information from NeRF:
                - 'rays': a list of ray packs, each entry corresponds to a single camera view
                - 'rgb', 'masks': a list of torch.Tensors, each entry corresponds to a single gt image
                - 'cameras': a list of Camera objects, one camera per view
        """
        with open(self._transform_file, 'r') as f:
            metadata = json.load(f)

        imgs = []
        poses = []
        basenames = []

        p = Pool(self.dataset_num_workers)
        try:
            mp_entries = [dict(frame=frame, root=self.dataset_path, mip=self.mip)
                          for frame in metadata['frames']]
            iterator = p.imap(NeRFSyntheticDataset._parallel_load_standard_imgs, mp_entries)

            for _ in tqdm(range(len(metadata['frames']))):
                result = next(iterator)
                basename = result['basename']
                img = result['img']
                pose = result['pose']
                if basename is not None:
                    basenames.append(basename)
                if img is not None:
                    imgs.append(img)
                if pose is not None:
                    poses.append(pose)
        finally:
            p.close()
            p.join()

        return self._collect_data_entries(metadata=metadata, basenames=basenames, imgs=imgs, poses=poses)

    def _collect_data_entries(self, metadata, basenames, imgs, poses, **kargs) -> Dict[str, Union[torch.Tensor, Rays, Camera]]:
        """ Internal function for aggregating the pre-loaded multi-views.
        This function will:
            1. Read the metadata & compute the intrinsic parameters of the camera view, (such as fov and focal length
                i.e., in the case of a pinhole camera).
            2. Apply various scaling and offsets transformations to the extrinsics,
                as specified by the metadata by parameters
               such as 'scale', 'offset' and 'aabb_scale'
            3. Create kaolin Camera objects out of the computed extrinsics and intrinsics.
            4. Invoke ray generation on each camera view.
            5. Stack the images pixel values and rays as per-view information entries.
        """
        imgs = torch.stack(imgs)
        poses = torch.stack(poses)

        # TODO(ttakikawa): Assumes all images are same shape and focal. Maybe breaks in general...
        h, w = imgs[0].shape[:2]

        if 'x_fov' in metadata:
            # Degrees
            x_fov = metadata['x_fov']
            fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
            if 'y_fov' in metadata:
                y_fov = metadata['y_fov']
                fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
            else:
                fy = fx
        elif 'fl_x' in metadata and False:
            fx = float(metadata['fl_x']) / float(2**self.mip)
            if 'fl_y' in metadata:
                fy = float(metadata['fl_y']) / float(2**self.mip)
            else:
                fy = fx
        elif 'camera_angle_x' in metadata:
            # Radians
            camera_angle_x = metadata['camera_angle_x']
            fx = (0.5 * w) / np.tan(0.5 * float(camera_angle_x))

            if 'camera_angle_y' in metadata:
                camera_angle_y = metadata['camera_angle_y']
                fy = (0.5 * h) / np.tan(0.5 * float(camera_angle_y))
            else:
                fy = fx

        else:
            fx = 0.0
            fy = 0.0

        if 'fix_premult' in metadata:
            log.info("WARNING: The dataset expects premultiplied alpha correction, "
                     "but the current implementation does not handle this.")

        if 'k1' in metadata:
            log.info \
                ("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

        if 'rolling_shutter' in metadata:
            log.info("WARNING: The dataset expects rolling shutter correction,"
                     "but the current implementation does not handle this.")

        # The principal point in wisp are always a displacement in pixels from the center of the image.
        x0 = 0.0
        y0 = 0.0
        # The standard dataset generally stores the absolute location on the image to specify the principal point.
        # Thus, we need to scale and translate them such that they are offsets from the center.
        if 'cx' in metadata:
            x0 = (float(metadata['cx']) / (2**self.mip)) - (w//2)
        if 'cy' in metadata:
            y0 = (float(metadata['cy']) / (2**self.mip)) - (h//2)

        offset = metadata['offset'] if 'offset' in metadata else [0 ,0 ,0]
        scale = metadata['scale'] if 'scale' in metadata else 1.0
        aabb_scale = metadata['aabb_scale'] if 'aabb_scale' in metadata else 1.25

        # TODO(ttakikawa): Actually scale the AABB instead? Maybe
        poses[..., :3, 3] /= aabb_scale
        poses[..., :3, 3] *= scale
        poses[..., :3, 3] += torch.FloatTensor(offset)

        # nerf-synthetic uses a default far value of 6.0
        default_far = 6.0

        rays = []

        cameras = dict()
        for i in range(imgs.shape[0]):
            view_matrix = torch.zeros_like(poses[i])
            view_matrix[:3, :3] = poses[i][:3, :3].T
            view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
            view_matrix[3, 3] = 1.0
            camera = Camera.from_args(view_matrix=view_matrix,
                                      focal_x=fx,
                                      focal_y=fy,
                                      width=w,
                                      height=h,
                                      far=default_far,
                                      near=0.0,
                                      x0=x0,
                                      y0=y0,
                                      dtype=torch.float64)
            camera.change_coordinate_system(blender_coords())
            cameras[basenames[i]] = camera
            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                      camera.width, camera.height, device='cuda')
            rays.append \
                (generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to
                    ('cpu'))

        rays = Rays.stack(rays).to(dtype=torch.float)
        rays, coords_center, coords_scale = self._normalize(cameras, rays, self.coords_center, self.coords_scale)
        self.coords_center = coords_center
        self.coords_scale = coords_scale

        rgbs = imgs[... ,:3]
        alpha = imgs[... ,3:4]
        if alpha.numel() == 0:
            masks = torch.ones_like(rgbs[... ,0:1]).bool()
        else:
            masks = (alpha > 0.5).bool()

            if self.bg_color == 'black':
                rgbs[... ,:3] -= ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)
            else:
                rgbs[... ,:3] *= alpha
                rgbs[... ,:3] += ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)

        data_dict = {"rgb": rgbs, "masks": masks, "rays": rays, "cameras": cameras}
        for key, value in kargs:
            data_dict[key] = value
        return data_dict

    @staticmethod
    def _normalize(cameras: Dict[str, Camera], rays: List[Rays], coords_center, coords_scale):
        """ Normalizes the content of all views to fit within an axis aligned bounding box of [-1, 1]:
        1. The pointcloud of a little gap from the edges is created.
        2. The pointcloud is normalized within the AABB of [-1, 1].
        3. The depth information, generated rays and cameras are rescaled according to normalization factors:
            coords_center, coords_scale.

        Returns:
            - (wisp.core.Rays) rays: the rescaled rays
            - (torch.Tensor) coords_center: Value used to centeralize the point cloud around 0, 0, 0.
            - (torch.Tensor) coords_scale: Value used to scale the point cloud within [-1, 1].
        """
        # edge_coords = create_edges_pointcloud_from_rays(rays)
        # normalized_coords, coords_center, coords_scale = normalize_pointcloud(edge_coords, return_scale=True)
        #
        if coords_center is None or coords_scale is None:
            coords = create_edges_pointcloud_from_rays(rays)
            normalized_coords, coords_center, coords_scale = normalize_pointcloud(coords, return_scale=True)

        rays.origins = (rays.origins - coords_center) * coords_scale
        # The following is only correct when using a single dist_min and dist_max for all rays
        rays.dist_min *= coords_scale.item()
        rays.dist_max *= coords_scale.item()

        for cam_id, cam in cameras.items():
            cam.translate(-coords_center.to(cam.dtype))
            cam.t = cam.t * coords_scale.to(cam.dtype)

        return rays, coords_center, coords_scale

    def flatten_tensors(self) -> None:
        """ Flattens the cached data tensors to (NUM_VIEWS, NUM_RAYS, *).
        """
        num_imgs = len(self)
        self.data["rgb"] = self.data["rgb"].reshape(num_imgs, -1, 3)
        self.data["rays"] = self.data["rays"].reshape(num_imgs, -1, 3)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(num_imgs, -1, 1)

    @property
    def img_shape(self) -> torch.Size:
        """ Returns the shape of the rescaled dataset images (cached values are flattened) """
        return self._img_shape

    @property
    def cameras(self) -> List[Camera]:
        """ Returns the list of camera views used to generate rays for this dataset. """
        return self.data["cameras"]

    @property
    def num_images(self) -> int:
        """ Returns the number of views this dataset stores. """
        return self.data["rgb"].shape[0]




class NeRFSyntheticDatasetWithCOLMAP(NeRFSyntheticDataset):
    """ A dataset for files in the standard NeRF format, including extensions to the format
        supported by Instant Neural Graphics Primitives.
        See: https://github.com/NVlabs/instant-ngp
        NeRF-synthetic scenes include RGBA / RGB information.
    """

    def __init__(self, dataset_path: str, colmap_res_path: str, split: str, bg_color: str, mip: int = 0,
                 dataset_num_workers: int = -1, transform: Callable = None, coords_norm_factors=[None, None]):
        """ Loads the NeRF-synthetic data and applies dataset specific transforms required for compatibility with the
        framework.
        The loaded data is cached inside the `data` field.

        Args:
            dataset_path (str): The root directory of the dataset, where images and json files of a single multiview
                scene reside.
            split (str): The dataset split to use, corresponding to the transform file to load, when splits are
                available. In case of a single transform file, it will be considered as a single split of
                'train' by default.
                Options: 'train', 'val', 'test'.
            bg_color (str): The background color to use for when alpha=0.
                Options: 'black', 'white'.
            mip (int): If provided, will rescale images by 2**mip. Useful when large images are loaded.
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        self.colmap_res_path = colmap_res_path
        self.coords_center, self.coords_scale = coords_norm_factors
        super().__init__(dataset_path=dataset_path, dataset_num_workers=dataset_num_workers,
                         transform=transform, split=split, bg_color=bg_color, mip=mip)


    def create_split(self, split: str, transform: Callable = None) -> NeRFSyntheticDataset:
        """ Creates a dataset with the same parameters and a different split.
        This is a convenient way of creating validation and test datasets, while making sure they're compatible
        with the train dataset.

        All settings except for split and transform will be copied from the current dataset.

        Args:
            split (str): The dataset split to use, corresponding to the transform file to load.
                Options: 'train', 'val', 'test'.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        return NeRFSyntheticDatasetWithCOLMAP(
            dataset_path=self.dataset_path,
            colmap_res_path=self.colmap_res_path,
            split=split,
            bg_color=self.bg_color,
            mip=self.mip,
            dataset_num_workers=self.dataset_num_workers,
            transform=transform,
            coords_norm_factors=[self.coords_center, self.coords_scale]
        )

    def __getitem__(self, idx) -> MultiviewBatch:
        """Retrieve a batch of rays and their corresponding values.
        Rays are precomputed from the dataset's cameras, and are cached within the dataset.
        By default, rays are assumed to have corresponding rgb values, sampled from the dataset's images.

        Returns:
            (MultiviewBatch): A batch of rays and their rgb values. The fields can be accessed as a dictionary:
                "rays" - a wisp.core.Rays pack of ray origins and directions, pre-generated from the dataset camera.
                "rgb" - a torch.Tensor of rgb color which corresponds the gt image's pixel each ray intersects.
                "masks" - a torch.BoolTensor specifying if the ray hits a dense area or not.
                 This is estimated from the alpha channel of the gt image, where mask=True if alpha > 0.5.
        """
        out = MultiviewBatchWithColmap(
            rays=self.data["rays"][idx],
            rgb=self.data["rgb"][idx],
            masks=self.data["masks"][idx],
            gt_depth=self.data["gt_depth"][idx]
        )

        if self.transform is not None:
            out = self.transform(out)

        return out

    def load_singleprocess(self):
        """Standard parsing function for loading nerf-synthetic files on the main process.
        This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

        Returns:
            (dict of torch.FloatTensors): Channels of information from NeRF:
                - 'rays': a list of ray packs, each entry corresponds to a single camera view
                - 'rgb', 'masks': a list of torch.Tensors, each entry corresponds to a single gt image
                - 'cameras': a list of Camera objects, one camera per view
        """
        with open(self._transform_file, 'r') as f:
            metadata = json.load(f)

        # Load COLMAP
        colmap_depth, col_near, col_far = load_colmap_depth(self.colmap_res_path, metadata['colmap_bin_model_path'], factor=self.mip**2)

        # Load the images

        imgs = []
        poses = []
        basenames = []
        colmap_depth_gt = []

        for frame in tqdm(metadata['frames'], desc='loading data'):
            _data = self._load_single_entry(frame, self.dataset_path, mip=self.mip)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])
                colmap_depth_gt.append(colmap_depth.get(_data["basename"], None))

        return self._collect_data_entries_with_colmap(metadata=metadata, basenames=basenames, imgs=imgs, poses=poses, colmap_depth_gt=colmap_depth_gt, col_near=col_near, col_far=col_far)

    def _collect_data_entries_with_colmap(self, metadata, basenames, imgs, poses, colmap_depth_gt, col_near, col_far) -> Dict[str, Union[torch.Tensor, Rays, Camera]]:
        """ Internal function for aggregating the pre-loaded multi-views.
        This function will:
            1. Read the metadata & compute the intrinsic parameters of the camera view, (such as fov and focal length
                i.e., in the case of a pinhole camera).
            2. Apply various scaling and offsets transformations to the extrinsics,
                as specified by the metadata by parameters
               such as 'scale', 'offset' and 'aabb_scale'
            3. Create kaolin Camera objects out of the computed extrinsics and intrinsics.
            4. Invoke ray generation on each camera view.
            5. Stack the images pixel values and rays as per-view information entries.
        """
        imgs = torch.stack(imgs)
        poses = torch.stack(poses)

        # TODO(ttakikawa): Assumes all images are same shape and focal. Maybe breaks in general...
        h, w = imgs[0].shape[:2]

        if 'x_fov' in metadata:
            # Degrees
            x_fov = metadata['x_fov']
            fx = (0.5 * w) / np.tan(0.5 * float(x_fov) * (np.pi / 180.0))
            if 'y_fov' in metadata:
                y_fov = metadata['y_fov']
                fy = (0.5 * h) / np.tan(0.5 * float(y_fov) * (np.pi / 180.0))
            else:
                fy = fx
        elif 'fl_x' in metadata and False:
            fx = float(metadata['fl_x']) / float(2**self.mip)
            if 'fl_y' in metadata:
                fy = float(metadata['fl_y']) / float(2**self.mip)
            else:
                fy = fx
        elif 'camera_angle_x' in metadata:
            # Radians
            camera_angle_x = metadata['camera_angle_x']
            fx = (0.5 * w) / np.tan(0.5 * float(camera_angle_x))

            if 'camera_angle_y' in metadata:
                camera_angle_y = metadata['camera_angle_y']
                fy = (0.5 * h) / np.tan(0.5 * float(camera_angle_y))
            else:
                fy = fx

        else:
            fx = 0.0
            fy = 0.0

        if 'fix_premult' in metadata:
            log.info("WARNING: The dataset expects premultiplied alpha correction, "
                     "but the current implementation does not handle this.")

        if 'k1' in metadata:
            log.info \
                ("WARNING: The dataset expects distortion correction, but the current implementation does not handle this.")

        if 'rolling_shutter' in metadata:
            log.info("WARNING: The dataset expects rolling shutter correction,"
                     "but the current implementation does not handle this.")

        # The principal point in wisp are always a displacement in pixels from the center of the image.
        x0 = 0.0
        y0 = 0.0
        # The standard dataset generally stores the absolute location on the image to specify the principal point.
        # Thus, we need to scale and translate them such that they are offsets from the center.
        if 'cx' in metadata:
            x0 = (float(metadata['cx']) / (2**self.mip)) - (w//2)
        if 'cy' in metadata:
            y0 = (float(metadata['cy']) / (2**self.mip)) - (h//2)

        offset = metadata['offset'] if 'offset' in metadata else [0 ,0 ,0]
        scale = metadata['scale'] if 'scale' in metadata else 1.0
        aabb_scale = metadata['aabb_scale'] if 'aabb_scale' in metadata else 1.25

        # TODO(ttakikawa): Actually scale the AABB instead? Maybe
        poses[..., :3, 3] /= aabb_scale
        poses[..., :3, 3] *= scale
        poses[..., :3, 3] += torch.FloatTensor(offset)

        # nerf-synthetic uses a default far value of 6.0
        default_far = 6.0

        rays = []

        cameras = dict()
        for i in range(imgs.shape[0]):
            view_matrix = torch.zeros_like(poses[i])
            view_matrix[:3, :3] = poses[i][:3, :3].T
            view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], poses[i][:3, -1])
            view_matrix[3, 3] = 1.0
            camera = Camera.from_args(view_matrix=view_matrix,
                                      focal_x=fx,
                                      focal_y=fy,
                                      width=w,
                                      height=h,
                                      far=col_far,#default_far,
                                      near=col_near,#0.0,
                                      x0=x0,
                                      y0=y0,
                                      dtype=torch.float64)
            camera.change_coordinate_system(blender_coords())
            cameras[basenames[i]] = camera
            ray_grid = generate_centered_pixel_coords(camera.width, camera.height,
                                                      camera.width, camera.height, device="cuda")
            rays.append \
                (generate_pinhole_rays(camera.to(ray_grid[0].device), ray_grid).reshape(camera.height, camera.width, 3).to
                    ('cpu'))

        rays = Rays.stack(rays).to(dtype=torch.float)

        rgbs = imgs[... ,:3]
        alpha = imgs[... ,3:4]
        if alpha.numel() == 0:
            masks = torch.ones_like(rgbs[... ,0:1]).bool()
        else:
            masks = (alpha > 0.5).bool()

            if self.bg_color == 'black':
                rgbs[... ,:3] -= ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)
            else:
                rgbs[... ,:3] *= alpha
                rgbs[... ,:3] += ( 1 -alpha)
                rgbs = np.clip(rgbs, 0.0, 1.0)

        colmap_depth_gt_sparse = torch.full((*rgbs.shape[:-1], 3), torch.nan,dtype=torch.float32)

        for i in range(imgs.shape[0]):
            _coord = colmap_depth_gt[i]['coord']
            _depth = colmap_depth_gt[i]['depth']
            _weight = colmap_depth_gt[i]['error']
            _point3d_idx = colmap_depth_gt[i]['point3d_idx']

            # big_points_idx = np.argwhere(_depth > 2.6)
            # gt_depth_and_error = np.stack([_depth[big_points_idx].flatten(), _weight[big_points_idx].flatten(), _point3d_idx[big_points_idx].flatten()], axis=1)
            # _coord = _coord[big_points_idx].reshape((big_points_idx.shape[0],2))
            gt_depth_and_error = np.stack([_depth, _weight, _point3d_idx], axis=1)


            colmap_depth_gt_sparse[
                (
                    torch.full((_coord.shape[0],), i),
                    torch.tensor(_coord[:, 1]),
                    torch.tensor(_coord[:, 0])
                )] = torch.tensor(
                gt_depth_and_error, dtype=torch.float32)

        # Take all of the supervision points, and make sure they fit in the feature grid
        depths, rays, coords_center, coords_scale = self._normalize(colmap_depth_gt_sparse[...,0], cameras, rays, self.coords_center, self.coords_scale)
        self.coords_center = coords_center
        self.coords_scale = coords_scale
        colmap_depth_gt_sparse[..., 0] = depths

        # Check how many from the rays in each batch comes from the depth supervision
        # If warning is raised, think what to do with that we have so little rgb supervision...
        max_depth_n_in_batch = (~torch.isnan(colmap_depth_gt_sparse)).sum(dim=[1, 2]).max().item()
        rays_n_in_batch = torch.inf if self.transform is None else self.transform.num_samples

        if not max_depth_n_in_batch <= rays_n_in_batch / 3:
            warnings.warn("Your depth supervision contains more than 1/3 from each batch. From the %d batch rays %d are depth rays" % (rays_n_in_batch, max_depth_n_in_batch))

        # colmap_range = col_far - col_near
        # colmap_depth_gt_sparse[:, 0] = (colmap_depth_gt_sparse[:, 0] - col_near) * (default_far / colmap_range)

        return {"rgb": rgbs, "masks": masks, "rays": rays, "cameras": cameras, "gt_depth": colmap_depth_gt_sparse.to(rgbs.device)}

    @staticmethod
    def _normalize(depths: torch.Tensor, cameras: Dict[str, Camera], rays: List[Rays], coords_center, coords_scale):
        """ Normalizes the content of all views to fit within an axis aligned bounding box of [-1, 1]:
        1. The pointcloud of a little gap from the edges is created.
        2. The pointcloud is normalized within the AABB of [-1, 1].
        3. The depth information, generated rays and cameras are rescaled according to normalization factors:
            coords_center, coords_scale.

        Returns:
            - (torch.Tensor) depths: the rescaled depth values of each ray
            - (wisp.core.Rays) rays: the rescaled rays
            - (torch.Tensor) coords_center: Value used to centeralize the point cloud around 0, 0, 0.
            - (torch.Tensor) coords_scale: Value used to scale the point cloud within [-1, 1].
        """
        # edge_coords = create_edges_pointcloud_from_rays(rays)
        # normalized_coords, coords_center, coords_scale = normalize_pointcloud(edge_coords, return_scale=True)
        #
        if coords_center is None or coords_scale is None:
            coords = create_edges_pointcloud_from_rays(rays)
            normalized_coords, coords_center, coords_scale = normalize_pointcloud(coords, return_scale=True)

        depths = depths * coords_scale
        rays.origins = (rays.origins - coords_center) * coords_scale
        # The following is only correct when using a single dist_min and dist_max for all rays
        rays.dist_min *= coords_scale.item()
        rays.dist_max *= coords_scale.item()

        for cam_id, cam in cameras.items():
            cam.translate(-coords_center.to(cam.dtype))
            cam.t = cam.t * coords_scale.to(cam.dtype)

        return depths, rays, coords_center, coords_scale
    def flatten_tensors(self) -> None:
        """ Flattens the cached data tensors to (NUM_VIEWS, NUM_RAYS, *).
        """
        num_imgs = len(self)
        self.data["rgb"] = self.data["rgb"].reshape(num_imgs, -1, 3)
        self.data["rays"] = self.data["rays"].reshape(num_imgs, -1, 3)
        self.data["gt_depth"] = self.data["gt_depth"].reshape(num_imgs, -1, 3)
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(num_imgs, -1, 1)




class NeRFSyntheticDatasetWithDIFT(NeRFSyntheticDataset):

    def __init__(self, dataset_path: str, split: str, bg_color: str, scene_name: str, dift_img_size: int = 256,
                 dift_ensemble: int = 4, mip: int = 0, dataset_num_workers: int = -1, transform: Callable = None,
                 coords_norm_factors=[None, None]):
        """ Loads the NeRF-synthetic data and applies dataset specific transforms required for compatibility with the
        framework.
        The loaded data is cached inside the `data` field.

        Args:
            dataset_path (str): The root directory of the dataset, where images and json files of a single multiview
                scene reside.
            split (str): The dataset split to use, corresponding to the transform file to load, when splits are
                available. In case of a single transform file, it will be considered as a single split of
                'train' by default.
                Options: 'train', 'val', 'test'.
            bg_color (str): The background color to use for when alpha=0.
                Options: 'black', 'white'.
            mip (int): If provided, will rescale images by 2**mip. Useful when large images are loaded.
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                If dataset_num_workers < 1, processing will take place on the main process.
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        self.scene_name = scene_name
        self.dift_img_size = dift_img_size
        self.dift_ensemble = dift_ensemble
        super().__init__(dataset_path=dataset_path, dataset_num_workers=dataset_num_workers,
                         transform=transform, split=split, bg_color=bg_color, mip=mip, coords_norm_factors=coords_norm_factors)

    def __getitem__(self, idx) -> MultiviewBatch:
        """Retrieve a batch of rays and their corresponding values.
        Rays are precomputed from the dataset's cameras, and are cached within the dataset.
        By default, rays are assumed to have corresponding rgb values, sampled from the dataset's images.

        Returns:
            (MultiviewBatch): A batch of rays and their rgb values. The fields can be accessed as a dictionary:
                "rays" - a wisp.core.Rays pack of ray origins and directions, pre-generated from the dataset camera.
                "rgb" - a torch.Tensor of rgb color which corresponds the gt image's pixel each ray intersects.
                "masks" - a torch.BoolTensor specifying if the ray hits a dense area or not.
                 This is estimated from the alpha channel of the gt image, where mask=True if alpha > 0.5.
        """
        out = MultiviewBatchWithColmap(
            rays=self.data["rays"][idx],
            rgb=self.data["rgb"][idx],
            masks=self.data["masks"][idx]
        )

        if self.transform is not None:
            out = self.transform(out, self.data['ft'])

        return out

    def load_singleprocess(self):
        """Standard parsing function for loading nerf-synthetic files on the main process.
        This follows the conventions defined in https://github.com/NVlabs/instant-ngp.

        Returns:
            (dict of torch.FloatTensors): Channels of information from NeRF:
                - 'rays': a list of ray packs, each entry corresponds to a single camera view
                - 'rgb', 'masks': a list of torch.Tensors, each entry corresponds to a single gt image
                - 'cameras': a list of Camera objects, one camera per view
        """
        with open(self._transform_file, 'r') as f:
            metadata = json.load(f)

        # Load the images

        imgs = []
        poses = []
        basenames = []

        for frame in tqdm(metadata['frames'], desc='loading data'):
            _data = self._load_single_entry(frame, self.dataset_path, mip=self.mip)
            if _data is not None:
                basenames.append(_data["basename"])
                imgs.append(_data["img"])
                poses.append(_data["pose"])


        dift_feats = self.calc_dift_features(imgs) # Ideally here I would also calculate correspondences for each point with the other points, later sampling from here. But due to memory limitations it is done later in the sampling process

        return self._collect_data_entries(metadata=metadata, basenames=basenames, imgs=imgs, poses=poses, ft=dift_feats)


    def calc_dift_features(self, imgs):
        dift = SDFeaturizer(device='cuda:1')
        prompt = f'a photo of a {self.scene_name}'

        orig_img_size = imgs[0].shape
        ft = []
        for img in imgs:
            resized = cv2.resize(img, dsize=(self.dift_img_size, self.dift_img_size), interpolation=cv2.INTER_CUBIC)
            # TODO make sure this normalizes to [-1,1]
            resized = (resized - 0.5) * 2
            ft.append(dift.forward(resized,
                                   prompt=prompt,
                                   ensemble_size=self.dift_ensemble))
            gc.collect()
            torch.cuda.empty_cache()

        ft = torch.cat(ft, dim=0)
        gc.collect()
        torch.cuda.empty_cache()
        return ft


    def flatten_tensors(self) -> None:
        """ Flattens the cached data tensors to (NUM_VIEWS, NUM_RAYS, *).
        """
        num_imgs = len(self)
        self.data["rgb"] = self.data["rgb"].reshape(num_imgs, -1, 3)
        self.data["rays"] = self.data["rays"].reshape(num_imgs, -1, 3)
        self.data["ft"] = self.data["ft"].reshape(num_imgs, -1, self.data["ft"].shape[-1])
        if "masks" in self.data:
            self.data["masks"] = self.data["masks"].reshape(num_imgs, -1, 1)


