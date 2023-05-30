import numpy as np
import os, imageio
import torch as torch
from pathlib import Path

from wisp.datasets.colmapUtils.pose_utils import gen_poses
from wisp.ops.image import load_rgb, resize_mip

from wisp.datasets.colmapUtils.read_write_model import *
from wisp.datasets.colmapUtils.read_write_dense import *
import json


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, bin_model_dir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = gen_poses(basedir, bin_model_dir, factors=[factor])
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    poses, bds, imgs = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # print('poses_bound.npy:\n', poses[:,:,0])

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)  # [-u, r, -t] -> [r, u, -t]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    print("bds:", bds[0])

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    # print('before recenter:\n', poses[0])

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def load_colmap_depth(basedir, bin_model_dir, factor=8, bd_factor=.75):
    abs_model_dir = os.path.join(basedir, bin_model_dir)
    images = read_images_binary(os.path.join(abs_model_dir, 'images.bin'))
    points = read_points3d_binary(os.path.join(abs_model_dir, 'points3D.bin'))

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses = get_poses(images)
    _, bds_raw = _load_data(basedir, bin_model_dir, factor=factor, load_imgs=False) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_dict = dict()
    for id_im in range(1, len(images) + 1):
        cur_image = images[id_im]
        depth_list = []
        coord_list = []
        point_3d_idx = []
        weight_list = []
        for i in range(len(cur_image.xys)):
            point2D = cur_image.xys[i]
            id_3D = cur_image.point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @ (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)
            depth_list.append(depth)
            coord_list.append(np.array(point2D / factor, dtype=int))
            point_3d_idx.append(id_3D)
            weight_list.append(weight)
        if len(depth_list) > 0:
            cur_img_name = cur_image.name.split('.')[0]
            print(id_im, cur_img_name, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_dict[cur_img_name] = {
                "depth": np.array(depth_list),
                "coord": np.array(coord_list),
                "point3d_idx": np.array(point_3d_idx),
                "error": np.array(weight_list)}
        else:
            print(id_im, len(depth_list))
    # json.dump(data_dict, open(data_file, "w"))
    # np.save(data_file, data_dict)
    return data_dict, near, far


def load_2_images_matchpoints(basedir, factor=8, bd_factor=.75, id_img_1=1, id_img_2=2):
    data_file = Path(basedir) / 'colmap_depth.npy'

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')


    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses = get_poses(images)
    poses_, bds_raw, images_ = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)


    data_list = []
    img_1_obj = images[id_img_1]
    img_2_obj = images[id_img_2]
    for i in range(len(img_1_obj.xys)):
        point2D_1 = img_1_obj.xys[i]
        id_3D = img_1_obj.point3D_ids[i]
        if id_3D == -1:
            continue

        point3D_obj = points[id_3D]
        if id_img_2 not in point3D_obj.image_ids:
            continue

        img_2_id_in_point3D = np.argwhere(point3D_obj.image_ids == id_img_2)[0, 0]
        point_id_in_img_obj = point3D_obj.point2D_idxs[img_2_id_in_point3D]
        point2D_2 = img_2_obj.xys[point_id_in_img_obj]

        point3D = points[id_3D].xyz
        # Now we have a 3d point that appear both in img1 and img2
        depth_1 = (poses[id_img_1 - 1, :3, 2].T @ (point3D - poses[id_img_1 - 1, :3, 3])) * sc
        depth_2 = (poses[id_img_2 - 1, :3, 2].T @ (point3D - poses[id_img_1 - 1, :3, 3])) * sc
        if (
                (depth_1 < bds_raw[id_img_1 - 1, 0] * sc or depth_1 > bds_raw[id_img_1 - 1, 1] * sc)
                or
                (depth_2 < bds_raw[id_img_2 - 1, 0] * sc or depth_2 > bds_raw[id_img_2 - 1, 1] * sc)
        ):
            # Not sure why, I guess here it's a noise point of COLMAP they filter out since it's out of bounds
            continue
        err = points[id_3D].error
        weight = 2 * np.exp(-(err / Err_mean) ** 2)
        data_list.append(
            {
                'img_1': img_1_obj, 'img_2': img_2_obj,
                'd_1': depth_1, 'd_2': depth_2,
                'coord_1': point2D_1 / factor, 'coord_2': point2D_2 / factor,
                'weight': weight
            })
    return data_list


def load_sensor_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    depthfiles = [Path(basedir) / 'depth' / f for f in sorted(os.listdir(Path(basedir) / 'depth')) if
                  f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depths = [imageio.imread(f) for f in depthfiles]
    depths = np.stack(depths, 0)

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @ (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)
            depth_list.append(depth)
            coord_list.append(point2D / factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append(
                {"depth": np.array(depth_list), "coord": np.array(coord_list), "weight": np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list


def load_colmap_llff(basedir):
    basedir = Path(basedir)

    train_imgs = np.load(basedir / 'train_images.npy')
    test_imgs = np.load(basedir / 'test_images.npy')
    train_poses = np.load(basedir / 'train_poses.npy')
    test_poses = np.load(basedir / 'test_poses.npy')
    video_poses = np.load(basedir / 'video_poses.npy')
    depth_data = np.load(basedir / 'train_depths.npy', allow_pickle=True)
    bds = np.load(basedir / 'bds.npy')

    return train_imgs, test_imgs, train_poses, test_poses, video_poses, depth_data, bds


if __name__ == '__main__':
    import cv2
    # data_dict, near, far = load_colmap_depth('/home/galharari/datasets/nerf_llff_data/fern')
    bash_path = '/mnt/more_space/datasets/nerf_llff_data/fern'
    data_dict, near, far = load_colmap_depth(bash_path)

    depth_gt_sparse = torch.full((10, 1000, 1000, 2), torch.nan)
    _coord_0 = data_dict['IMG_4034']['coord']
    _depth_0 = data_dict['IMG_4034']['depth']
    _weight_0 = data_dict['IMG_4034']['error']
    _point3d_idx_0 = data_dict['IMG_4034']['error']

    _coord_1 = data_dict['IMG_4027']['coord']
    _depth_1 = data_dict['IMG_4027']['depth']
    _weight_1 = data_dict['IMG_4027']['error']
    _point3d_idx_1 = data_dict['IMG_4027']['error']

    mip = 2
    img_0 = load_rgb(bash_path + '/images/IMG_4034.JPG')
    if mip is not None:
        img_0 = resize_mip(img_0, mip, interpolation=cv2.INTER_AREA)

    img_1 = load_rgb(bash_path + '/images/IMG_4043.JPG')
    if mip is not None:
        img_1 = resize_mip(img_1, mip, interpolation=cv2.INTER_AREA)

    # Coord[0] is on img[1] and coord[1] is on img[0]
    point3d_idx_0_non_nan = _point3d_idx_0[~np.isnan(_point3d_idx_0)]
    point3d_idx_1_non_nan = _point3d_idx_1[~np.isnan(_point3d_idx_1)]
    point3d_matches_idx = point3d_idx_0_non_nan[np.isin(point3d_idx_0_non_nan, point3d_idx_1_non_nan)]

    point3d_i = point3d_matches_idx[0]
    coord_0 = np.argwhere(_point3d_idx_0 == point3d_i)
    coord_1 = np.argwhere(_point3d_idx_1 == point3d_i)
    all_coord_0 = np.argwhere(np.isin(_point3d_idx_0, point3d_matches_idx))
    all_coord_1 = np.argwhere(np.isin(_point3d_idx_1, point3d_matches_idx))

    # datalist = load_2_images_matchpoints('/mnt/more_space/datasets/nerf_llff_data/fern/', id_img_1=2, id_img_2=15)
    # print(datalist)

    import cv2, torch

    rgb = torch.load('_results/logs/runs/rgb.pt')
    gt_depth = torch.load('_results/logs/runs/gt_depth.pt')
    rgb_0 = rgb[0,...].numpy()
    rgb_1 = rgb[1,...].numpy()

    gt_depth_0 = gt_depth[0, ...].numpy()
    gt_depth_1 = gt_depth[1, ...].numpy()

    depth_0, error_0, point3d_idx_0 = gt_depth_0[..., 0], gt_depth_0[..., 1], gt_depth_0[..., 2]
    depth_1, error_1, point3d_idx_1 = gt_depth_1[..., 0], gt_depth_1[..., 1], gt_depth_1[..., 2]

    point3d_idx_0_non_nan = point3d_idx_0[~np.isnan(point3d_idx_0)]
    point3d_idx_1_non_nan = point3d_idx_1[~np.isnan(point3d_idx_1)]

    point3d_matches_idx = point3d_idx_0_non_nan[np.isin(point3d_idx_0_non_nan, point3d_idx_1_non_nan)]

    point3d_i = point3d_matches_idx[0]
    coord_0 = np.argwhere(point3d_idx_0 == point3d_i)
    coord_1 = np.argwhere(point3d_idx_1 == point3d_i)
    all_coord_0 = np.argwhere(np.isin(point3d_idx_0, point3d_matches_idx))
    all_coord_1 = np.argwhere(np.isin(point3d_idx_1, point3d_matches_idx))

    # img3 = cv2.drawMatches(rgb_0,
    #                        [cv2.KeyPoint(x=int(x[0]), y=int(x[1]), size=31) for x in all_coord_0],
    #                        rgb_1,
    #                        [cv2.KeyPoint(x=int(x[0]), y=int(x[1]), size=31) for x in all_coord_1],
    #                        [i for i in range(10)], None, flags=2)
    for point3d_i in point3d_matches_idx[0: 30]:
        coord_0 = np.where(point3d_idx_0 == point3d_i)
        coord_1 = np.where(point3d_idx_1 == point3d_i)
        # Opencv draws circle with reverse coordinates
        # cv2.circle(rgb_0, (coord_0[1][0], coord_0[0][0]), radius=5, color=(0, 0, 255),thickness=-1)
        # cv2.circle(rgb_1, (coord_1[1][0], coord_1[0][0]), radius=5, color=(0, 0, 255),thickness=-1)
        cv2.circle(rgb_0, (950, 750), radius=5, color=(0, 0, 255),thickness=-1)
        cv2.circle(rgb_1, (750, 950), radius=5, color=(0, 0, 255),thickness=-1)
        # rgb_0[coord_0] = (0,0,255)
        # rgb_1[coord_1] = (0,0,255)

    cv2.imshow('img_0', rgb_0)
    cv2.waitKey(30)
    cv2.imshow('img_1', rgb_1)
    cv2.waitKey(30)
    # cv2.imshow('img_3', img3)
    cv2.waitKey(0)



