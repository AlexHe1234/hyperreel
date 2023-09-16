#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import json, csv
import os
from torch.utils.data import Dataset

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import imageio

import numpy as np
import torch
import glob

from PIL import Image
from scipy.spatial.transform import Rotation

from utils.pose_utils import (
    average_poses,
    correct_poses_bounds,
    create_rotating_spiral_poses,
    create_spiral_poses,
    interpolate_poses,
)

from utils.ray_utils import (
    get_ndc_rays_fx_fy,
    get_pixels_for_image,
    get_ray_directions_K,
    get_rays,
    sample_images_at_xy,
)

from .base import Base5DDataset, Base6DDataset
from .if_nerf_utils import *


class NHRDataset(Base6DDataset):
    def __init__(self, cfg, split='train', **kwargs):
        self.use_reference = False
        self.correct_poses = False
        self.use_ndc = False

        self.num_frames = cfg.dataset.num_frames if 'num_frames' in cfg.dataset else 100
        self.start_frame = cfg.dataset.start_frame if 'start_frame' in cfg.dataset else 0
        self.keyframe_step = cfg.dataset.keyframe_step if "keyframe_step" in cfg.dataset else 1
        self.num_keyframes = cfg.dataset.num_keyframes if "num_keyframes" in cfg.dataset else self.num_frames // self.keyframe_step

        self.load_full_step = cfg.dataset.load_full_step if "load_full_step" in cfg.dataset else 1
        self.subsample_keyframe_step = cfg.dataset.subsample_keyframe_step if "subsample_keyframe_step" in cfg.dataset else 1
        self.subsample_keyframe_frac = cfg.dataset.subsample_keyframe_frac if "subsample_keyframe_frac" in cfg.dataset else 1.0
        self.subsample_frac = cfg.dataset.subsample_frac if "subsample_frac" in cfg.dataset else 1.0

        self.keyframe_offset = 0
        self.frame_offset = 0
        
        super().__init__(cfg, split, **kwargs)
        
    def get_transform(self):
        ann_file = self.root_dir
        if 'basketball' in ann_file:
            transform = np.array([[ 0.90630779, -0.07338689,  0.41619774,  2.],
                [ 0.42261826,  0.1573787 , -0.89253894, -5.13],
                [ 0.        ,  0.98480775,  0.17364818,  2.85],
                [ 0.        ,  0.        ,  0.        ,  1.]])
        elif 'sport_1' in ann_file:
            transform = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.17364818, -0.98480775, -3.88852744],
                [ 0.        ,  0.98480775,  0.17364818,  2.16014765],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])
        elif 'sport_2' in ann_file:
            transform = np.array([[ 0.90630779, -0.07338689,  0.41619774,  2.],
                [ 0.42261826,  0.1573787 , -0.89253894, -5.13],
                [ 0.        ,  0.98480775,  0.17364818,  2.85],
                [ 0.        ,  0.        ,  0.        ,  1.]])
        elif 'sport_3' in ann_file:
            transform = np.array([[ 0.98480775, -0.04494346,  0.16773126, -0.75454014],
                [-0.17364818, -0.254887  ,  0.95125124, -4.9550868 ],
                [ 0.        , -0.96592583, -0.25881905,  1.9544816 ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])
        else:
            raise NotImplementedError('This NHR Dataset is not supported')
        
        return transform
    
    def read_meta(self):        
        ann_file = os.path.join(self.root_dir, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        cams = annots['cams']
        ims = annots['ims']
        
        img_paths = []
        for i in range(self.num_frames):
            img_paths += ims[i]['ims']
        self.img_paths = img_paths  # frame * num_cams
        
        self.images_per_frame = len(ims[0]['ims'])
        
        assert self.images_per_frame == 56
        
        self.total_images_per_frame = len(ims[0]['ims'])
        
        self.bounds = np.load(os.path.join(self.root_dir, 'bounds.npy'))
        
        self.Ks = cams['K']  # num_cam, 3, 3
        Rs = np.array(cams['R'])
        Ts = np.array(cams['T']) / 1000.
        num_cams = len(Rs)
        
        assert num_cams == self.images_per_frame
        
        for i in range(num_cams):
            R = Rs[i]
            T = Ts[i]
            RT = np.concatenate([R, T], axis=1)
            RT = np.concatenate([RT, [[0, 0, 0, 1]]], axis=0)
            transform = self.get_transform()
            RT = np.dot(RT, np.linalg.inv(transform))
            R, T = RT[:3, :3], RT[:3, 3:]
            Rs[i] = R
            Ts[i] = T
            
        self.Rs = Rs
        self.Ts = Ts
        
        self.near = self.bounds.min() * 0.95
        self.far = self.bounds.max() * 1.05
        self.bounds = np.array([self.near, self.far])
        self.depth_range = np.array([self.near, self.far])
        
        R_p = Rs.transpose((0, 2, 1))
        T_p = -R_p @ Ts
        print('T_p shape: ', T_p.shape)
        self.poses = np.empty((num_cams, 3, 4))
        self.poses[:, :, :3] = R_p.reshape(num_cams, 3, 3)
        self.poses[:, :, 3] = T_p.reshape(num_cams, 3)
        
        self.HW = np.empty((num_cams, 2), dtype=np.uint)
        for i in range(num_cams):
            img_path = os.path.join(self.root_dir, ims[0]['ims'][i])
            img = cv2.imread(img_path)
            self.HW[i] = img.shape[:2]

        self.poses = np.stack([self.poses for i in range(self.num_frames)]).reshape(-1, 3, 4)  # frame * num_cams, 3, 4
        self.times = np.tile(np.linspace(0, 1, self.num_frames)[..., None], (1, self.images_per_frame))
        self.times = self.times.reshape(-1)  # frame * num_cams
        self.camera_ids = np.tile(np.linspace(0, self.images_per_frame - 1, self.images_per_frame, dtype=np.uint)[None, :], (self.num_frames, 1))
        self.camera_ids = self.camera_ids.reshape(-1)  # frames * num_cams
        # self.camera_ids = self.camera_ids.astype(np.uint)
        assert self.poses.shape[0] == self.times.shape[0] and self.times.shape[0] == self.camera_ids.shape[0]
        
        # pose, time, camera_id indices for train and val split
        val_indices = []

        for idx in self.val_set:
            val_indices += [frame * self.images_per_frame + idx for frame in range(self.num_frames)]

        train_indices = [
            i for i in range(len(self.poses)) if i not in val_indices
        ]
        
        if self.split == "val":
            self.img_paths = [self.img_paths[i] for i in val_indices]

            self.poses = self.poses[val_indices]
            self.times = self.times[val_indices]
            self.camera_ids = self.camera_ids[val_indices]
        elif self.split == "train":
            self.img_paths = [self.img_paths[i] for i in train_indices]

            self.poses = self.poses[train_indices]
            self.times = self.times[train_indices]
            self.camera_ids = self.camera_ids[train_indices]
        else:
            raise NotImplementedError
        
        self.num_images = len(self.poses)
    
    def subsample(self, coords, rgb, frame, cam_idx, fac=1.0):
        return coords, rgb
        # if frame % self.load_full_step == 0:
        #     return coords, rgb
        
        # if frame % self.subsample_keyframe_step == 0:
        #     subsample_every = int(np.round(1.0 / (self.subsample_keyframe_frac * fac)))
        #     offset = self.keyframe_offset
        #     self.keyframe_offset += 1
        # else:
        #     subsample_every = int(np.round(1.0 / (self.subsample_frac * fac)))
        #     offset = self.frame_offset
        #     self.frame_offset += 1
            
        # pixels = get_pixels_for_image(
        #     self.HW[cam_idx][0], self.HW[cam_idx][1]
        # ).reshape(-1, 2).long()
        # mask = ((pixels[..., 0] + pixels[..., 1] + offset) % subsample_every) == 0.0

        # return coords[mask].view(-1, coords.shape[-1]), rgb[mask].view(-1, rgb.shape[-1])
    
    def prepare_train_data(self):
        pass
        ## Collect training data
        # self.all_coords = []
        # self.all_rgb = []
        # num_pixels = 0
        # # last_rgb_full = None
        
        # for idx in range(len(self.img_paths)):
            
        #     # img = self.get_rgb(idx)
            
        #     cur_coords, cur_rgb = self.get_coords(idx, None)
            
        #     # img = self.transform(img)
        #     # cur_rgb = img.view(3, -1).permute(1, 0)  # rays, 3
            
        #     # if self.split == 'train':
        #     # mask_ = cv2.imread(os.path.join(self.root_dir, self.img_paths[idx]).replace('images', 'mask'), cv2.IMREAD_GRAYSCALE) > 150
        #     # mask_ = mask_.flatten()
        
        #     # # cur_coords = cur_coords[mask_]
        #     # cur_rgb[mask_] *= 0
            
        #     cur_frame = int(np.round(self.times[idx] * (self.num_frames - 1)))
            
        #     # cur_coords, cur_rgb = self.subsample(cur_coords, cur_rgb, cur_frame, self.camera_ids[idx])
            
        #     # Coords
        #     self.all_coords += [cur_coords]

        #     # Color
        #     self.all_rgb += [cur_rgb]

        #     # Number of pixels
        #     num_pixels += cur_rgb.shape[0]

        #     print("Full res images loaded:", num_pixels / (self.img_wh[0] * self.img_wh[1]))

        # self.all_coords = torch.cat(self.all_coords, 0)  # all_rays, 3+3+1+1
        # self.all_rgb = torch.cat(self.all_rgb, 0)  # all_rays, 3
        
        # assert len(self.all_coords) == len(self.all_rgb)
        
        # self.update_all_data()
    
    def update_all_data(self):
        # self.all_weights = self.get_weights()

        # ## All inputs
        # self.all_inputs = torch.cat(
        #     [
        #         self.all_coords,
        #         self.all_rgb,
        #         self.all_weights,
        #     ],
        #     -1,
        # )
        pass

    def format_batch(self, batch):
        # print(batch['inputs'].shape)
        batch["coords"] = batch["inputs"][..., :8]
        batch["rgb"] = batch["inputs"][
            ..., 8:11
        ]
        batch["weight"] = batch["inputs"][..., 11:]
        del batch["inputs"]

        return batch
    
    # here idx is index of an image
    def get_coords(self, idx, img):

        # img path
        # pose
        # time
        # camera id
        cam_idx = self.camera_ids[idx]
        # k = self.Ks[cam_idx]
        HW = self.HW[cam_idx]
        # c2w = torch.FloatTensor(self.poses[idx])
        time = self.times[idx]
        cur_frame = int(np.round(time * (self.num_frames - 1)))
        
        # TODO: copy mlpmaps code
        # 1. open point cloud
        # 2. get wbounds
        # 3. get ray_o and ray_d
        pc_path = os.path.join(self.root_dir, 'vertices', f'{cur_frame}.npy')
        wpts = np.load(pc_path).astype(np.float32)
        transform = self.get_transform()
        wpts = np.dot(wpts, transform[:3, :3].T) + transform[:3, 3]
        wbounds = get_bounds(wpts)
        K = self.Ks[cam_idx]
        R = self.Rs[cam_idx]
        T = self.Ts[cam_idx]
        msk_path = os.path.join(self.root_dir, 'mask', f'{cam_idx:04d}/{cur_frame:06d}.jpg')
        msk = imageio.imread(msk_path)[..., 0].astype(np.int32)
        msk = (msk > 100).astype(np.uint8)
        # print(f'image id: {idx}, camera id: {cam_idx}, frame: {cur_frame}')
        if img is None:
            img_path = os.path.join(self.root_dir, 'images', f'{cam_idx:04d}/{cur_frame:06d}.jpg')
            img_path2 = os.path.join(self.root_dir, self.img_paths[idx])
            assert img_path == img_path2
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        
        rgb, rays_o, rays_d = sample_ray_h36m(img, msk, K, R, T, wbounds, self.split)
    
        rays_o /= 2.
        
        rgb = torch.tensor(rgb, dtype=torch.float)
        rays_o = torch.tensor(rays_o, dtype=torch.float)
        rays_d = torch.tensor(rays_d, dtype=torch.float)
    
        # print("Loading time:", cur_frame)
        rays = torch.cat([rays_o, rays_d], dim=-1)  # ? rays, 6
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * cam_idx], dim=-1)  # ? rays, 7
        rays = torch.cat([rays, torch.ones_like(rays[..., :1]) * time], dim=-1)  # ? rays, 8
        
        return rays, rgb
    

    def get_rgb(self, idx):
        img = cv2.imread(os.path.join(self.root_dir, self.img_paths[idx]))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # if img.shape[0] != self._img_wh[0] or img.shape[1] != self._img_wh[1]:
        #     img = img[:self._img_wh[0], :self._img_wh[1]]

        # if self.img_wh[0] != self._img_wh[0] or self.img_wh[1] != self._img_wh[1]:
        #     img = cv2.resize(img, self.img_wh, cv2.INTER_AREA)
        img = self.transform(img)
        img = img.view(3, -1).permute(1, 0)  # rays, 3

        return img
    
    def __getitem__(self, idx):

        if self.split == 'train':
            # print(f'getting {idx}')
            rays, rgb = self.get_coords(idx, None)
            
            weights = torch.ones(
                *rgb[..., 0:1].shape, device='cpu'
            )

            ## All inputs
            inputs = torch.cat(
                [
                    rays,
                    rgb,
                    weights,
                ],
                -1,
            )
            
            batch = {
                "inputs": inputs,
            }
        elif self.split == 'val':
            coords, rgb = self.get_coords(idx, None)
            batch = {
                "coords": coords,
                "rgb": rgb,
                "idx": idx,
            }

            batch["weight"] = torch.ones_like(batch["coords"][..., -1:])
        else:
            raise NotImplementedError

        if self.split == 'val':
            cam_idx = self.camera_ids[idx]
            HW = self.HW[cam_idx]
            # W, H, batch = self.crop_batch(batch)
            batch["W"] = HW[1].astype(np.int32)
            batch["H"] = HW[0].astype(np.int32)

        return batch

    def __len__(self):
        if self.split == 'train':
                return len(self.img_paths)
        elif self.split == 'val':
            return min(self.val_num, len(self.poses))
        elif self.split == 'render':
            if self.render_max_frames > 0:
                return  min(self.render_max_frames, len(self.poses))
            else:
                return len(self.poses)
        else:
            return len(self.poses)
