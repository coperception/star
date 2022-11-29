import os
import math
from multiprocessing import Manager

import numpy as np
from coperception.utils.obj_util import *
from coperception.datasets.V2XSimDet import V2XSimDet


class MultiTempV2XSimDet(V2XSimDet):
    def __init__(
        self,
        dataset_roots=None,
        config=None,
        config_global=None,
        split=None,
        cache_size=10000,
        val=False,
        bound=None,
        kd_flag=False,
        no_cross_road=False,
        time_stamp = 2,
    ):
        """
        Inherited from the V2XSimDet dataset, now support multi timestamp training
        output padded_voxel_points_next are for other timestamp
        it is an empty list if timestamp = 1
        """
        super().__init__(dataset_roots, config, config_global, split, cache_size, val, bound, kd_flag, no_cross_road)
        self.time_stamp = config.time_stamp if hasattr(config, 'time_stamp') else time_stamp
        print("use time stamp", self.time_stamp)

    def pick_single_agent(self, agent_id, idx):
        empty_flag = False
        if idx in self.cache[agent_id]:
            gt_dict = self.cache[agent_id][idx]
        else:
            seq_file = self.seq_files[agent_id][idx]
            gt_data_handle = np.load(seq_file, allow_pickle=True)
            if gt_data_handle == 0:
                empty_flag = True
                padded_voxel_points = []
                padded_voxel_points_next_list = []
                padded_voxel_points_teacher = []
                label_one_hot = np.zeros_like(self.label_one_hot_meta)
                reg_target = np.zeros_like(self.reg_target_meta)
                anchors_map = np.zeros_like(self.anchors_map_meta)
                vis_maps = np.zeros_like(self.vis_maps_meta)
                reg_loss_mask = np.zeros_like(self.reg_loss_mask_meta)

                if self.bound == "lowerbound" or self.bound == 'both':
                    padded_voxel_points = np.zeros_like(self.padded_voxel_points_meta)
                    # TODO: multi time stamp
                    if self.time_stamp>1:
                        for i in range(self.time_stamp-1):
                            padded_voxel_points_next_list.append(np.zeros_like(self.padded_voxel_points_meta))
                        padded_voxel_points_next_list = np.concatenate(padded_voxel_points_next_list, 0).astype(np.float32)
                    else:
                        padded_voxel_points_next_list = np.asarray(padded_voxel_points_next_list)
                    # print("padded other time stamp shape", padded_voxel_points_next_list.shape)

                if self.kd_flag or self.bound == "upperbound" or self.bound == 'both':
                    padded_voxel_points_teacher = np.zeros_like(
                        self.padded_voxel_points_meta
                    )

                if self.val:
                    return (
                        padded_voxel_points,
                        padded_voxel_points_next_list, # added
                        padded_voxel_points_teacher,
                        label_one_hot,
                        reg_target,
                        reg_loss_mask,
                        anchors_map,
                        vis_maps,
                        [{"gt_box": []}],
                        [seq_file],
                        0,
                        0,
                        np.zeros((self.num_agent, 4, 4)),
                    )
                else:
                    return (
                        padded_voxel_points,
                        padded_voxel_points_next_list, # added
                        padded_voxel_points_teacher,
                        label_one_hot,
                        reg_target,
                        reg_loss_mask,
                        anchors_map,
                        vis_maps,
                        0,
                        0,
                        np.zeros((self.num_agent, 4, 4)),
                    )
            else:
                gt_dict = gt_data_handle.item()
                if len(self.cache[agent_id]) < self.cache_size:
                    self.cache[agent_id][idx] = gt_dict

        if not empty_flag:
            allocation_mask = gt_dict["allocation_mask"].astype(bool)
            reg_loss_mask = gt_dict["reg_loss_mask"].astype(bool)
            gt_max_iou = gt_dict["gt_max_iou"]

            # load regression target
            reg_target_sparse = gt_dict["reg_target_sparse"]
            # need to be modified Yiqi , only use reg_target and allocation_map
            reg_target = np.zeros(self.reg_target_shape).astype(reg_target_sparse.dtype)

            reg_target[allocation_mask] = reg_target_sparse
            reg_target[np.bitwise_not(reg_loss_mask)] = 0
            label_sparse = gt_dict["label_sparse"]

            one_hot_label_sparse = self.get_one_hot(label_sparse, self.category_num)
            label_one_hot = np.zeros(self.label_one_hot_shape)
            label_one_hot[:, :, :, 0] = 1
            label_one_hot[allocation_mask] = one_hot_label_sparse

            if self.only_det:
                reg_target = reg_target[:, :, :, :1]
                reg_loss_mask = reg_loss_mask[:, :, :, :1]

            # only center for pred
            elif self.config.pred_type in ["motion", "center"]:
                reg_loss_mask = np.expand_dims(reg_loss_mask, axis=-1)
                reg_loss_mask = np.repeat(reg_loss_mask, self.box_code_size, axis=-1)
                reg_loss_mask[:, :, :, 1:, 2:] = False

            # Prepare padded_voxel_points
            padded_voxel_points = []
            if self.bound == "lowerbound" or self.bound == "both":
                for i in range(self.num_past_pcs):
                    indices = gt_dict["voxel_indices_" + str(i)]
                    curr_voxels = np.zeros(self.dims, dtype=bool)
                    curr_voxels[indices[:, 0], indices[:, 1], indices[:, 2]] = 1
                    curr_voxels = np.rot90(curr_voxels, 3)
                    padded_voxel_points.append(curr_voxels)
                padded_voxel_points = np.stack(padded_voxel_points, 0).astype(
                    np.float32
                )
                padded_voxel_points = padded_voxel_points.astype(np.float32)

            anchors_map = self.anchors_map

            if self.config.use_vis:
                vis_maps = np.zeros(
                    (
                        self.num_past_pcs,
                        self.config.map_dims[-1],
                        self.config.map_dims[0],
                        self.config.map_dims[1],
                    )
                )
                vis_free_indices = gt_dict["vis_free_indices"]
                vis_occupy_indices = gt_dict["vis_occupy_indices"]
                vis_maps[
                    vis_occupy_indices[0, :],
                    vis_occupy_indices[1, :],
                    vis_occupy_indices[2, :],
                    vis_occupy_indices[3, :],
                ] = math.log(0.7 / (1 - 0.7))
                vis_maps[
                    vis_free_indices[0, :],
                    vis_free_indices[1, :],
                    vis_free_indices[2, :],
                    vis_free_indices[3, :],
                ] = math.log(0.4 / (1 - 0.4))
                vis_maps = np.swapaxes(vis_maps, 2, 3)
                vis_maps = np.transpose(vis_maps, (0, 2, 3, 1))
                for v_id in range(vis_maps.shape[0]):
                    vis_maps[v_id] = np.rot90(vis_maps[v_id], 3)
                vis_maps = vis_maps[-1]
            else:
                vis_maps = np.zeros(0)

            if self.no_cross_road:
                trans_matrices = gt_dict["trans_matrices_no_cross_road"]
            else:
                trans_matrices = gt_dict["trans_matrices"]

            label_one_hot = label_one_hot.astype(np.float32)
            reg_target = reg_target.astype(np.float32)
            anchors_map = anchors_map.astype(np.float32)
            vis_maps = vis_maps.astype(np.float32)

            target_agent_id = gt_dict["target_agent_id"]
            num_sensor = gt_dict["num_sensor"]

            # Prepare padded_voxel_points_teacher
            padded_voxel_points_teacher = []
            if "voxel_indices_teacher" in gt_dict and (
                self.kd_flag or self.bound == "upperbound" or self.bound == "both"
            ):
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
                padded_voxel_points_teacher = np.stack(
                    padded_voxel_points_teacher, 0
                ).astype(np.float32)
                padded_voxel_points_teacher = padded_voxel_points_teacher.astype(
                    np.float32
                )

            # Temporal, added by Juexiao--------------------------------
            # main modification here: -----------------
            # use next time stamp image
            # if no next time stamp, use the same voxel (a plausible pseudo next)
            # -----------------------------------------
            # TODO: future -> past
            padded_voxel_points_next_list = [] # a list holding every padded_voxel_points_next for all the multi timestamp
            # end_frame_idx = self.time_stamp - 1 + idx
            # current_scene = self.seq_scenes[agent_id][idx]
            # for nextframe_idx in range(idx+1, end_frame_idx+1):
            #     while ((nextframe_idx) > (len(self.seq_scenes[agent_id])-1) or self.seq_scenes[agent_id][nextframe_idx] != current_scene):
            #         # padded_voxel_points_next = padded_voxel_points
            #         nextframe_idx = nextframe_idx - 1 # still use last frame as the next frame
            # use past features
            end_frame_idx = - self.time_stamp + 1 + idx
            current_scene = self.seq_scenes[agent_id][idx]
            for nextframe_idx in range(idx-1, end_frame_idx-1, -1):
                while ( nextframe_idx < 0 or self.seq_scenes[agent_id][nextframe_idx] != current_scene):
                    # padded_voxel_points_next = padded_voxel_points
                    nextframe_idx = nextframe_idx + 1 # still use last frame as the next frame

                next_seq_file = self.seq_files[agent_id][nextframe_idx]
                next_gt_data_handle = np.load(next_seq_file, allow_pickle=True)
                if next_gt_data_handle == 0:
                    padded_voxel_points_next_list.append(padded_voxel_points[0].copy())
                    # next_trans_matrices = trans_matrices
                else:
                    next_gt_dict = next_gt_data_handle.item()
                    indices_next = next_gt_dict['voxel_indices_0'] # FIXME: is it always 0.
                    # next_trans_matrices = next_gt_dict['trans_matrices']
                    next_voxels = np.zeros(self.dims, dtype=bool)
                    next_voxels[indices_next[:, 0], indices_next[:, 1], indices_next[:, 2]] = 1
                    next_voxels = np.rot90(next_voxels, 3)
                    padded_voxel_points_next_list.append(next_voxels)
                    # padded_voxel_points_next = np.stack(padded_voxel_points_next, 0).astype(np.float32)
                    # padded_voxel_points_next = padded_voxel_points_next.astype(np.float32)
                    # print(nextframe_idx, "voxels shape", padded_voxel_points_next.shape)
                # padded_voxel_points_next_list.append(padded_voxel_points_next)
            if len(padded_voxel_points_next_list)>0:
                padded_voxel_points_next_list = np.stack(padded_voxel_points_next_list, 0).astype(np.float32)
            else:
                padded_voxel_points_next_list = np.asarray(padded_voxel_points_next_list)
            # print("all other timestamp voxels shape", padded_voxel_points_next_list.shape)
            ## ----------------------------------------

            if self.val:
                return (
                    padded_voxel_points,
                    padded_voxel_points_next_list,
                    padded_voxel_points_teacher,
                    label_one_hot,
                    reg_target,
                    reg_loss_mask,
                    anchors_map,
                    vis_maps,
                    [{"gt_box": gt_max_iou}],
                    [seq_file],
                    target_agent_id,
                    num_sensor,
                    trans_matrices,
                )

            else:
                return (
                    padded_voxel_points,
                    padded_voxel_points_next_list,
                    padded_voxel_points_teacher,
                    label_one_hot,
                    reg_target,
                    reg_loss_mask,
                    anchors_map,
                    vis_maps,
                    target_agent_id,
                    num_sensor,
                    trans_matrices,
                )

    
