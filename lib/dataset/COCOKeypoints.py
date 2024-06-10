# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import pycocotools
from .COCODataset import CocoDataset
from .target_generators import HeatmapGenerator


logger = logging.getLogger(__name__)


class CocoKeypoints(CocoDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 joints_generator,
                 transforms=None):
        super().__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT)
        
        print("Number of joints:", cfg.DATASET.NUM_JOINTS)

        if cfg.DATASET.WITH_CENTER:
            assert cfg.DATASET.NUM_JOINTS == 17, 'Number of joint with center for COCO is 17'
        else:
            print("Number of joints:", cfg.DATASET.NUM_JOINTS)

            assert cfg.DATASET.NUM_JOINTS == 17, 'Number of joint for COCO is 17'

        self.num_scales = self._init_check(heatmap_generator, joints_generator)

        self.num_joints = cfg.DATASET.NUM_JOINTS
        #assert self.num_joints == 5, 'Number of joints for COCO should be 5'
        self.with_center = cfg.DATASET.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.scale_aware_sigma = cfg.DATASET.SCALE_AWARE_SIGMA
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.int_sigma = cfg.DATASET.INT_SIGMA

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.joints_generator = joints_generator

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)

        mask = self.get_mask(anno, idx)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        # TODO(bowen): to generate scale-aware sigma, modify `get_joints` to associate a sigma to each joint
        joints = self.get_joints(anno)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]
        target_list = list()

        if self.transforms:
            img, mask_list, joints_list = self.transforms(
                img, mask_list, joints_list
            )

        for scale_id in range(self.num_scales):
            target_t = self.heatmap_generator[scale_id](joints_list[scale_id])
            joints_t = self.joints_generator[scale_id](joints_list[scale_id])

            target_list.append(target_t.astype(np.float32))
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            joints_list[scale_id] = joints_t.astype(np.int32)

        return img, target_list, mask_list, joints_list

    def get_joints(self, anno):
        num_people = len(anno)
         
        if self.scale_aware_sigma:
            joints = np.zeros((num_people, self.num_joints, 4))
        else:
            joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            keypoints = np.array(obj['keypoints']).reshape(-1, 3)
            num_keypoints = min(self.num_joints_without_center, keypoints.shape[0])
            joints[i, :num_keypoints, :3] = keypoints[:num_keypoints]
            
            if self.with_center:
                joints_sum = np.sum(joints[i, :num_keypoints, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :num_keypoints, 2])[0])
                if num_vis_joints > 0:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                    joints[i, -1, 2] = 1
                    
            if self.scale_aware_sigma:
                box = obj['bbox']
                size = max(box[2], box[3])
                sigma = size / self.base_size * self.base_sigma
                if self.int_sigma:
                    sigma = int(np.round(sigma + 0.5))
                assert sigma > 0, sigma
                joints[i, :, 3] = sigma

        return joints


    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif 'num_keypoints' not in obj or obj['num_keypoints'] == 0:
                if isinstance(obj['segmentation'], list) and len(obj['segmentation']) > 0:
                 try:
                    rles = pycocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'], img_info['width'])
                    for rle in rles:
                        m += pycocotools.mask.decode(rle)
                 except Exception as e:
                    print(f"Error processing segmentation for annotation {obj['id']}: {e}")  
                else:
                    print(f"Invalid segmentation data for annotation {obj['id']}")
        
        return m < 0.5
    
    def _init_check(self, heatmap_generator, joints_generator):
        assert isinstance(heatmap_generator, (list, tuple)), 'heatmap_generator should be a list or tuple'
        assert isinstance(joints_generator, (list, tuple)), 'joints_generator should be a list or tuple'
        assert len(heatmap_generator) == len(joints_generator), \
            'heatmap_generator and joints_generator should have same length,'\
            'got {} vs {}.'.format(
                len(heatmap_generator), len(joints_generator)
            )
        return len(heatmap_generator)