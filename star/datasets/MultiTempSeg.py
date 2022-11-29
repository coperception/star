import os
from multiprocessing import Manager

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from coperception.datasets.V2XSimSeg import V2XSimSeg


class Transform:
    def __init__(self, split):
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize((256, 256))
        self.split = split

    def __call__(self, img, label):
        img = self.totensor(img.copy())
        label = self.totensor(label.copy())

        if self.split != "train":
            return img.permute(1, 2, 0).float(), label.squeeze(0).int()

        crop = transforms.RandomResizedCrop(256)
        params = crop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        img = TF.crop(img, *params)
        label = TF.crop(label, *params)

        if np.random.random() > 0.5:
            img = TF.hflip(img)
            label = TF.hflip(label)

        if np.random.random() > 0.5:
            img = TF.vflip(img)
            label = TF.vflip(label)

        img = self.resize(img)
        label = cv2.resize(
            label.squeeze(0).numpy(), dsize=(256, 256), interpolation=cv2.INTER_NEAREST
        )  # Resize provided by pytorch will have some random noise
        # return img.permute(1, 2, 0).float(), label.squeeze(0).int()
        return img.permute(1, 2, 0).float(), label


class MultiTempV2XSimSeg(V2XSimSeg):
    def __init__(
        self,
        dataset_roots=None,
        config=None,
        split=None,
        cache_size=1000,
        val=False,
        com=False,
        bound=None,
        kd_flag=False,
        no_cross_road=False,
        time_stamp = 2,
    ):
        """
        Inherited from the V2XSimSeg dataset, now support multi timestamp loading
        output padded_voxel_points_next are for other timestamp
        it is an empty list if timestamp = 1
        """
        super().__init__(dataset_roots, config, split, cache_size, val, com, bound, kd_flag, no_cross_road)
        self.time_stamp = config.time_stamp if hasattr(config, 'time_stamp') else time_stamp
        print("use time stamp", self.time_stamp)

    def get_seginfo_from_single_agent(self, agent_id, idx):
        empty_flag = False
        if idx in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][idx]
        else:
            seq_file = self.seq_files[agent_id][idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                if self.time_stamp > 1:
                    padded_voxel_next = torch.zeros((self.time_stamp-1, 256, 256, 13)).bool()
                else:
                    padded_voxel_next = torch.zeros(0).bool()
                if self.com:
                    return (
                        torch.zeros((256, 256, 13)).bool(),
                        padded_voxel_next,
                        torch.zeros((256, 256, 13)).bool(),
                        torch.zeros((256, 256)).int(),
                        torch.zeros((self.num_agent, 4, 4)),
                        0,
                        0,
                    )
                else:
                    return (
                        torch.zeros((256, 256, 13)).bool(),
                        padded_voxel_next,
                        torch.zeros((256, 256, 13)).bool(),
                        torch.zeros((256, 256)).int(),
                    )
            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][idx] = gt_dict

        if not empty_flag:
            bev_seg = gt_dict["bev_seg"].astype(np.int32)

            padded_voxel_points = list()

            # if self.bound == 'lowerbound':
            for i in range(self.num_past_pcs):
                indices = gt_dict["voxel_indices_" + str(i)]
                curr_voxels = np.zeros(self.dims, dtype=bool)
                curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

                curr_voxels = np.rot90(curr_voxels, 3)
                # curr_voxels = np.rot90(np.fliplr(curr_voxels), 3)
                bev_seg = np.rot90(bev_seg, 1)  # to align with voxel

                padded_voxel_points.append(curr_voxels)
            padded_voxel_points = np.stack(padded_voxel_points, 0)
            padded_voxel_points = np.squeeze(padded_voxel_points, 0)

            padded_voxel_points_teacher = list()
            # if self.bound == 'upperbound' or self.kd_flag:
            if self.no_cross_road:
                indices_teacher = gt_dict["voxel_indices_teacher_no_cross_road"]
            else:
                indices_teacher = gt_dict["voxel_indices_teacher"]

            curr_voxels_teacher = np.zeros(self.dims, dtype=bool)
            curr_voxels_teacher[
                indices_teacher[:, 0], indices_teacher[:, 1], indices_teacher[:, 2]
            ] = 1
            curr_voxels_teacher = np.rot90(curr_voxels_teacher, 3)
            padded_voxel_points_teacher.append(curr_voxels_teacher)
            padded_voxel_points_teacher = np.stack(padded_voxel_points_teacher, 0)
            padded_voxel_points_teacher = np.squeeze(padded_voxel_points_teacher, 0)

            # temporal
            padded_voxel_points_next_list = []
            end_frame_idx = - self.time_stamp + 1 + idx
            current_scene = self.seq_scenes[agent_id][idx]
            for nextframe_idx in range(idx-1, end_frame_idx-1, -1):
                while ( nextframe_idx < 0 or self.seq_scenes[agent_id][nextframe_idx] != current_scene):
                    nextframe_idx = nextframe_idx + 1 
                next_seq_file = self.seq_files[agent_id][nextframe_idx]
                next_gt_data_handle = np.load(next_seq_file, allow_pickle=True)
                if next_gt_data_handle == 0:
                    padded_voxel_points_next_list.append(padded_voxel_points[0].copy())
                else:
                    next_gt_dict = next_gt_data_handle.item()
                    indices_next = next_gt_dict['voxel_indices_0'] # FIXME: is it always 0.
                    next_voxels = np.zeros(self.dims, dtype=bool)
                    next_voxels[indices_next[:, 0], indices_next[:, 1], indices_next[:, 2]] = 1
                    next_voxels = np.rot90(next_voxels, 3)
                    padded_voxel_points_next_list.append(next_voxels)
            if len(padded_voxel_points_next_list)>0:
                padded_voxel_points_next_list = np.stack(padded_voxel_points_next_list, 0).astype(np.int32)
            else:
                padded_voxel_points_next_list = np.asarray(padded_voxel_points_next_list)

            if self.com:
                if self.no_cross_road:
                    trans_matrices = gt_dict["trans_matrices_no_cross_road"]
                else:
                    trans_matrices = gt_dict["trans_matrices"]

                target_agent_id = gt_dict["target_agent_id"]
                num_sensor = gt_dict["num_sensor"]

                return (
                    torch.from_numpy(padded_voxel_points),
                    torch.from_numpy(padded_voxel_points_next_list),
                    torch.from_numpy(padded_voxel_points_teacher),
                    torch.from_numpy(bev_seg.copy()),
                    torch.from_numpy(trans_matrices.copy()),
                    target_agent_id,
                    num_sensor,
                )
            else:
                return (
                    torch.from_numpy(padded_voxel_points),
                    torch.from_numpy(padded_voxel_points_next_list),
                    torch.from_numpy(padded_voxel_points_teacher),
                    torch.from_numpy(bev_seg.copy()),
                )
