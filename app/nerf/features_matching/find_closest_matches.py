import cv2
import numpy as np
import torch
# Check over 92 and 93 those are images where I know there are points we see on both images
image_1_features_path = '/home/gal/Gits/kaolin-wisp/app/nerf/features_matching/renders_output/out_0_features.pt'
image_1_path = '/home/gal/Gits/kaolin-wisp/app/nerf/features_matching/renders_output/in_0_image.pt'

image_2_features_path = '/home/gal/Gits/kaolin-wisp/app/nerf/features_matching/renders_output/out_1_features.pt'
image_2_path = '/home/gal/Gits/kaolin-wisp/app/nerf/features_matching/renders_output/in_1_image.pt'

# In format of tev's (h, w) (in tev it is written (w,h) you can know by moving the cursor from left to right over an image)
points_to_check = [(30, 900)]#, (35, 129), (54, 160), (162, 47)]
colors = [np.array([0,0,255]), np.array([255,0,0]), np.array([0,255,0]), np.array([255,255,0])]

img_1_tensor = torch.load(image_1_path)
img_2_tensor = torch.load(image_2_path)
features_1 = torch.load(image_1_features_path).numpy()
features_2 = torch.load(image_2_features_path).numpy()

import torchvision.transforms as T
from PIL import Image, ImageDraw
from numpy import dot
from numpy.linalg import norm


# def cos_sim(a, b):
#     return dot(a, b) / (norm(a) * norm(b))
cos_sim = torch.nn.CosineSimilarity()

to_pil_image = T.ToPILImage()
# hwc to chw
img_1 = to_pil_image(img_1_tensor.permute(2, 0, 1))
img_2 = to_pil_image(img_2_tensor.permute(2, 0, 1))
img_2_drawed = img_2.copy()

orig_draw_r = 1
found_draw_r = 2
k = 5
img_1_drw = ImageDraw.Draw(img_1, 'RGBA')
img_2_drw = ImageDraw.Draw(img_2_drawed, 'RGBA')
for i, point in enumerate(points_to_check):
    # Draw on source image to see it
    print(point)
    # Draw ellipse given a bounding box of (x0,y0,x1,y1), x in width, y in height, x1>x0 y1>y0
    img_1_drw.ellipse(xy=(point[1] - orig_draw_r, point[0] - orig_draw_r, \
                          point[1] + orig_draw_r, point[0] + orig_draw_r), fill=tuple(colors[i]))

    # Fine nearest Neighbors
    point_features = features_1[point[0], point[1]]
    # distances = ((features_2 - point_features) ** 2).sum(axis=-1)
    distances = cos_sim(features_2, point_features)
    nearest_idxs = np.dstack(np.unravel_index(np.argsort(distances, axis=None), distances.shape))[0, :k, ...]

    print(point)
    for n_p in nearest_idxs:
        print(n_p, distances[n_p[0], n_p[1]])

    nearest_p = nearest_idxs[0]
    # Draw ellipse given a bounding box of (x0,y0,x1,y1), x in width, y in height, x1>x0 y1>y0
    img_2_drw.ellipse(xy=(nearest_p[1] - found_draw_r, nearest_p[0] - found_draw_r, \
                          nearest_p[1] + found_draw_r, nearest_p[0] + found_draw_r), fill=tuple(colors[i]))
    for near_p in nearest_idxs[1:]:
        img_2_drw.ellipse(xy=(near_p[1] - found_draw_r, near_p[0] - found_draw_r, \
                              near_p[1] + found_draw_r, near_p[0] + found_draw_r),
                          fill=(colors[i][0], colors[i][1], colors[i][2], 100))

del img_1_drw
del img_2_drw

cv2.imshow('img1', (img_1))
cv2.imshow('img2', (img_2))
cv2.imshow('img2_drawed', (img_2_drawed))
