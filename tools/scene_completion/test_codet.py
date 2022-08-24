import argparse
import os
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import torch.optim as optim
from torch.utils.data import DataLoader

from coperception.datasets import V2XSimDet, MultiTempV2XSimDet
from coperception.configs import Config, ConfigGlobal
from coperception.utils.CoDetModule import *
from coperception.utils.loss import *
from coperception.utils.mean_ap import eval_map
from coperception.models.det import *
from coperception.utils.detection_util import late_fusion
from coperception.utils.data_util import apply_pose_noise

import glob

from coperception.models.transformers import multiagent_mae
from coperception.models.generatives import VQVAENet
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


@torch.no_grad()
def main(args):
    config = Config("train", binary=True, only_det=True)
    config_global = ConfigGlobal("train", binary=True, only_det=True)

    need_log = args.log
    num_workers = args.nworker
    apply_late_fusion = args.apply_late_fusion
    pose_noise = args.pose_noise
    compress_level = args.compress_level
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    config.inference = args.inference
    if args.bound == "upperbound":
        flag = "upperbound"
    else:
        if args.com == "when2com":
            flag = "when2com"
            if args.inference == "argmax_test":
                flag = "who2com"
            if args.warp_flag:
                flag = flag + "_warp"
        elif args.com in {"v2v", "disco", "sum", "mean", "max", "cat", "agent", "ind_mae", "joint_mae", "late", "vqvae"}:
            flag = args.com
        else:
            flag = "lowerbound"
            if args.box_com:
                flag += "_box_com"

    print("flag", flag)
    config.flag = flag
    config.split = "test"

    num_agent = args.num_agent
    # agent0 is the cross road
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)
    validation_dataset = MultiTempV2XSimDet(
        dataset_roots=[f"{args.data}/agent{i}" for i in agent_idx_range],
        config=config,
        config_global=config_global,
        split="val",
        val=True,
        bound="both",
        kd_flag=args.kd_flag,
        no_cross_road=args.no_cross_road,
        time_stamp= args.time_stamp
    )
    validation_data_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Validation dataset size:", len(validation_dataset))

    if args.no_cross_road:
        num_agent -= 1

    if args.com == "sum":
        model_completion = SumFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "mean":
        model_completion = MeanFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "max":
        model_completion = MaxFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "cat":
        model_completion = CatFusion(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "v2v":
        model_completion = V2VNet(
            config,
            gnn_iter_times=args.gnn_iter_times,
            layer=args.layer,
            layer_channel=256,
            num_agent=num_agent,
        )
    elif args.com == "joint_mae" or  args.com == "ind_mae":
        # Juexiao added for mae
        model_completion = multiagent_mae.__dict__[args.mae_model](norm_pix_loss=args.norm_pix_loss, time_stamp=args.time_stamp, mask_method=args.mask, encode_partial=args.encode_partial)
        # also include individual reconstruction: reconstruct then aggregate
    elif args.com == "late":
        model_completion = FaFNet(
            config,
            layer=args.layer,
            kd_flag=args.kd_flag,
            num_agent=num_agent,
            train_completion=True,
        )
    elif args.com == "vqvae":
        model_completion = VQVAENet(
            num_hiddens = 256,
            num_residual_hiddens = 32,
            num_residual_layers = 2,
            num_embeddings = 512,
            embedding_dim = 64,
            commitment_cost = 0.25,
            decay = 0., # for VQ
        )
    else:
        raise NotImplementedError("Fusion type: {args.com} not implemented")

    
    model = FaFNet(
        config, layer=args.layer, kd_flag=args.kd_flag, num_agent=num_agent, train_completion=False,
    )
    
    # model_completion = nn.DataParallel(model_completion)
    model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = {
        "cls": SoftmaxFocalClassificationLoss(),
        "loc": WeightedSmoothL1LocalizationLoss(),
    }

    if args.com == "ind_mae" or args.com == "joint_mae":
        completion_param_groups = optim_factory.add_weight_decay(model_completion, 0.05)
        optimizer_completion = optim.Adam(completion_param_groups, lr=args.lr, betas=(0.9, 0.95))
    else:
        optimizer_completion = optim.Adam(model_completion.parameters(), lr=args.lr)

    # detection model
    fafmodule = FaFModule(model, model, config, optimizer, criterion, args.kd_flag)
    # completion model
    faf_module_completion = FaFModule(
        model_completion,
        model_completion,
        config,
        optimizer_completion,
        criterion,
        args.kd_flag,
    )
    checkpoint_completion = torch.load(args.resume_completion, map_location='cpu')
    completion_start_epoch = checkpoint_completion["epoch"] + 1
    faf_module_completion.resume_from_cpu(checkpoint=checkpoint_completion, device=device, trainable=False)
    print("completion model loaded from {} at epoch {}".format(args.resume_completion, completion_start_epoch-1))

    # model_save_path = args.resume[: args.resume.rfind("/")]
    model_save_path = os.path.join(args.logpath, "agnostic-det")

    if args.inference == "argmax_test":
        model_save_path = model_save_path.replace("when2com", "who2com")

    os.makedirs(model_save_path, exist_ok=True)
    log_file_name = os.path.join(model_save_path, "log_det.txt")
    saver = open(log_file_name, "a")
    saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
    saver.flush()

    # Logging the details for this experiment
    saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
    saver.write(args.__repr__() + "\n\n")
    saver.flush()

    checkpoint = torch.load(
        args.resume, map_location="cpu"
    )  # We have low GPU utilization for testing
    start_epoch = checkpoint["epoch"] + 1
    fafmodule.model.load_state_dict(checkpoint["model_state_dict"])
    fafmodule.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    fafmodule.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    print("Load detection model from {}, at epoch {}".format(args.resume, start_epoch - 1))

    #  ===== eval =====
    fafmodule.model.eval()
    faf_module_completion.model.eval()
    # eval_save_path = check_folder(os.path.join(model_save_path, flag))
    eval_save_path = check_folder(model_save_path)
    print("save evalutaion results at", eval_save_path)
    save_fig_path = [
        check_folder(os.path.join(eval_save_path, f"vis{i}")) for i in agent_idx_range
    ]
    tracking_path = [
        check_folder(os.path.join(eval_save_path, f"tracking{i}"))
        for i in agent_idx_range
    ]

    # for local and global mAP evaluation
    det_results_local = [[] for i in agent_idx_range]
    annotations_local = [[] for i in agent_idx_range]

    tracking_file = [set()] * num_agent
    for cnt, sample in enumerate(validation_data_loader):
        t = time.time()
        (
            padded_voxel_point_list,
            padded_voxel_point_next_list,
            padded_voxel_points_teacher_list,
            label_one_hot_list,
            reg_target_list,
            reg_loss_mask_list,
            anchors_map_list,
            vis_maps_list,
            gt_max_iou,
            filenames,
            target_agent_id_list,
            num_agent_list,
            trans_matrices_list,
        ) = zip(*sample)

        print(filenames)

        filename0 = filenames[0]
        trans_matrices = torch.stack(tuple(trans_matrices_list), 1)
        target_agent_ids = torch.stack(tuple(target_agent_id_list), 1)
        num_all_agents = torch.stack(tuple(num_agent_list), 1)

        # add pose noise
        if pose_noise > 0:
            apply_pose_noise(pose_noise, trans_matrices)

        if args.no_cross_road:
            num_all_agents -= 1
        # print("bev", type(padded_voxel_point_list), len(padded_voxel_point_list), len(padded_voxel_point_list[0]))
        # print(padded_voxel_point_list)
        if flag == "upperbound":
            padded_voxel_points = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
        else:
            padded_voxel_points = torch.cat(tuple(padded_voxel_point_list), 0)

        label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
        reg_target = torch.cat(tuple(reg_target_list), 0)
        reg_loss_mask = torch.cat(tuple(reg_loss_mask_list), 0)
        anchors_map = torch.cat(tuple(anchors_map_list), 0)
        vis_maps = torch.cat(tuple(vis_maps_list), 0)

        padded_voxel_points_teacher = torch.cat(
                tuple(padded_voxel_points_teacher_list), 0
        )

        data = {
            "bev_seq": padded_voxel_points.to(device),
            "bev_seq_teacher": padded_voxel_points_teacher.to(device),
            "labels": label_one_hot.to(device),
            "reg_targets": reg_target.to(device),
            "anchors": anchors_map.to(device),
            "vis_maps": vis_maps.to(device),
            "reg_loss_mask": reg_loss_mask.to(device).type(dtype=torch.bool),
            "target_agent_ids": target_agent_ids.to(device),
            "num_agent": num_all_agents.to(device),
            "trans_matrices": trans_matrices.to(device),
        }

        if flag != "upperbound":
            print("predict detection over reconstructed.")
            if args.com == "ind_mae" or args.com == "joint_mae":
                # multi timestamp input bev supported in the data
                data["bev_seq_next"] = torch.cat(tuple(padded_voxel_point_next_list), 0).to(device)
                # loss, reconstruction, _ = faf_module.step_mae_completion(data, batch_size, args.mask_ratio, trainable=False)
                loss, completed_point_cloud, _ = faf_module_completion.infer_mae_completion(data, batch_size=1, mask_ratio=args.mask_ratio)
            elif args.com == "vqvae":
                _, completed_point_cloud, _ = faf_module_completion.step_vae_completion(data, batch_size=1, trainable=False)
            else:
                _, completed_point_cloud = faf_module_completion.step_completion(
                    data, batch_size=1, trainable=False
                )

            completed_point_cloud = completed_point_cloud.unsqueeze(1)
            completed_point_cloud = completed_point_cloud.permute(0, 1, 3, 4, 2)

            # substitute with the completed data
            data["bev_seq"] = completed_point_cloud
            # print("check lowerbound")
            if args.visualization:
                # visualize completion and ground truth (teacher)
                reconstructed = data["bev_seq"].squeeze(1).permute(0,3,1,2)
                recon_img = torch.max(reconstructed, dim=1, keepdim=True)[0]
                target = data['bev_seq_teacher'].squeeze(1).permute(0,3,1,2)
                target_img = torch.max(target, dim=1, keepdim=True)[0]
                filename = str(filename0[0][0])
                eval_start_idx = 0 if args.no_cross_road else 1
                # local qualitative evaluation
                for kid in range(eval_start_idx, num_agent):
                    recon_name = "recon-epc{}-agent{}.png".format(completion_start_epoch-1, kid)
                    target_name = "target-epc{}-agent{}.png".format(completion_start_epoch-1, kid)
                    cut = filename[filename.rfind("agent") + 7 :]
                    seq_name = cut[: cut.rfind("_")]
                    idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
                    seq_save = os.path.join(save_fig_path[kid], seq_name)
                    check_folder(seq_save)
                    recon_save = str(idx) + recon_name
                    target_save = str(idx) + target_name
                    plt.imshow(target_img[kid,:,:,:].permute(1,2,0).squeeze(-1).cpu().numpy(), alpha=1.0, zorder=12, cmap="Purples")
                    plt.axis('off')
                    print("save", os.path.join(seq_save, target_save))
                    plt.savefig(os.path.join(seq_save, target_save))
                    plt.imshow(recon_img[kid,:,:,:].permute(1,2,0).squeeze(-1).cpu().numpy(), alpha=1.0, zorder=12, cmap="Purples")
                    plt.axis('off')
                    plt.savefig(os.path.join(seq_save, recon_save))
                    # imshow(target_img[kid,:,:,:].permute(1,2,0).squeeze(-1).numpy(), alpha=1.0, zorder=12)
                    # imshow(recon_img[kid,:,:,:].permute(1,2,0).squeeze(-1).numpy(), alpha=1.0, zorder=12)


        if flag == "lowerbound_box_com":
            loss, cls_loss, loc_loss, result = fafmodule.predict_all_with_box_com(
                data, data["trans_matrices"]
            )
        elif flag == "disco":
            (
                loss,
                cls_loss,
                loc_loss,
                result,
                save_agent_weight_list,
            ) = fafmodule.predict_all(data, 1, num_agent=num_agent)
        else:
            loss, cls_loss, loc_loss, result = fafmodule.predict_all(
                data, 1, num_agent=num_agent
            )

        box_color_map = ["red", "yellow", "blue", "purple", "black", "orange"]

        # If has RSU, do not count RSU's output into evaluation
        eval_start_idx = 0 if args.no_cross_road else 1
        
        # local qualitative evaluation
        for k in range(eval_start_idx, num_agent):
            box_colors = None
            if apply_late_fusion == 1 and len(result[k]) != 0:
                pred_restore = result[k][0][0][0]["pred"]
                score_restore = result[k][0][0][0]["score"]
                selected_idx_restore = result[k][0][0][0]["selected_idx"]

            data_agents = {
                # "bev_seq": torch.unsqueeze(padded_voxel_points[k, :, :, :, :], 1), #NOTE: what is this for? only visualization, not involved in mAP calc
                "bev_seq": torch.unsqueeze(padded_voxel_points_teacher[k, :, :, :, :], 1),
                "reg_targets": torch.unsqueeze(reg_target[k, :, :, :, :, :], 0),
                "anchors": torch.unsqueeze(anchors_map[k, :, :, :, :], 0),
            }
            temp = gt_max_iou[k]

            if len(temp[0]["gt_box"]) == 0:
                data_agents["gt_max_iou"] = []
            else:
                data_agents["gt_max_iou"] = temp[0]["gt_box"][0, :, :]

            # late fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                box_colors = late_fusion(
                    k, num_agent, result, trans_matrices, box_color_map
                )

            result_temp = result[k]

            temp = {
                "bev_seq": data_agents["bev_seq"][0, -1].cpu().numpy(), 
                "result": [] if len(result_temp) == 0 else result_temp[0][0],
                "reg_targets": data_agents["reg_targets"].cpu().numpy()[0],
                "anchors_map": data_agents["anchors"].cpu().numpy()[0],
                "gt_max_iou": data_agents["gt_max_iou"],
            }
            det_results_local[k], annotations_local[k] = cal_local_mAP(
                config, temp, det_results_local[k], annotations_local[k]
            )

            filename = str(filename0[0][0])
            cut = filename[filename.rfind("agent") + 7 :]
            seq_name = cut[: cut.rfind("_")]
            idx = cut[cut.rfind("_") + 1 : cut.rfind("/")]
            seq_save = os.path.join(save_fig_path[k], seq_name)
            check_folder(seq_save)
            idx_save = str(idx) + ".png"
            temp_ = deepcopy(temp)
            if args.visualization:
                visualization(
                    config,
                    temp,
                    box_colors,
                    box_color_map,
                    apply_late_fusion,
                    os.path.join(seq_save, idx_save),
                )

            # # plot the cell-wise edge
            # if flag == "disco" and k < len(save_agent_weight_list):
            #     one_agent_edge = save_agent_weight_list[k]
            #     for kk in range(len(one_agent_edge)):
            #         idx_edge_save = (
            #             str(idx) + "_edge_" + str(kk) + "_to_" + str(k) + ".png"
            #         )
            #         savename_edge = os.path.join(seq_save, idx_edge_save)
            #         sns.set()
            #         plt.savefig(savename_edge, dpi=500)
            #         plt.close(0)

            # == tracking ==
            if args.tracking:
                scene, frame = filename.split("/")[-2].split("_")
                det_file = os.path.join(tracking_path[k], f"det_{scene}.txt")
                if scene not in tracking_file[k]:
                    det_file = open(det_file, "w")
                    tracking_file[k].add(scene)
                else:
                    det_file = open(det_file, "a")
                det_corners = get_det_corners(config, temp_)
                for ic, c in enumerate(det_corners):
                    det_file.write(
                        ",".join(
                            [
                                str(
                                    int(frame) + 1
                                ),  # frame idx is 1-based for tracking
                                "-1",
                                "{:.2f}".format(c[0]),
                                "{:.2f}".format(c[1]),
                                "{:.2f}".format(c[2]),
                                "{:.2f}".format(c[3]),
                                str(result_temp[0][0][0]["score"][ic]),
                                "-1",
                                "-1",
                                "-1",
                            ]
                        )
                        + "\n"
                    )
                    det_file.flush()

                det_file.close()

            # restore data before late-fusion
            if apply_late_fusion == 1 and len(result[k]) != 0:
                result[k][0][0][0]["pred"] = pred_restore
                result[k][0][0][0]["score"] = score_restore
                result[k][0][0][0]["selected_idx"] = selected_idx_restore

        print("Validation scene {}, at frame {}".format(seq_name, idx))
        print("Takes {} s\n".format(str(time.time() - t)))

    logger_root = args.logpath if args.logpath != "" else "logs"
    logger_root = os.path.join(
        logger_root, f"{flag}_eval", "no_cross" if args.no_cross_road else "with_cross"
    )
    os.makedirs(logger_root, exist_ok=True)
    log_file_path = os.path.join(logger_root, "log_test.txt")
    log_file = open(log_file_path, "w")

    def print_and_write_log(log_str):
        print(log_str)
        log_file.write(log_str + "\n")

    mean_ap_local = []
    # local mAP evaluation
    det_results_all_local = []
    annotations_all_local = []
    for k in range(eval_start_idx, num_agent):
        if type(det_results_local[k]) != list or len(det_results_local[k]) == 0:
            continue

        print_and_write_log("Local mAP@0.5 from agent {}".format(k))
        mean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.5,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)
        print_and_write_log("Local mAP@0.7 from agent {}".format(k))

        ean_ap, _ = eval_map(
            det_results_local[k],
            annotations_local[k],
            scale_ranges=None,
            iou_thr=0.7,
            dataset=None,
            logger=None,
        )
        mean_ap_local.append(mean_ap)

        det_results_all_local += det_results_local[k]
        annotations_all_local += annotations_local[k]

    # average local mAP evaluation
    print_and_write_log("Average Local mAP@0.5")

    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.5,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    print_and_write_log("Average Local mAP@0.7")

    mean_ap_local_average, _ = eval_map(
        det_results_all_local,
        annotations_all_local,
        scale_ranges=None,
        iou_thr=0.7,
        dataset=None,
        logger=None,
    )
    mean_ap_local.append(mean_ap_local_average)

    print_and_write_log(
        "Quantitative evaluation results of model from {}, at epoch {}".format(
            args.resume, start_epoch - 1
        )
    )
    print_and_write_log(
        "Quantitative evaluation results of completion model from {}".format(
            args.resume_completion,
        )
    )

    for k in range(eval_start_idx, num_agent):
        print_and_write_log(
            "agent{} mAP@0.5 is {} and mAP@0.7 is {}".format(
                k, mean_ap_local[k * 2], mean_ap_local[(k * 2) + 1]
            )
        )

    print_and_write_log(
        "average local mAP@0.5 is {} and average local mAP@0.7 is {}".format(
            mean_ap_local[-2], mean_ap_local[-1]
        )
    )

    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default=None,
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--batch", default=4, type=int, help="The number of scene")
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=1, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="The path to the saved det model that is loaded to test",
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument(
        "--layer",
        default=3,
        type=int,
        help="Communicate which layer in the single layer com mode",
    )
    parser.add_argument(
        "--warp_flag", action="store_true", help="Whether to use pose info for When2com"
    )
    parser.add_argument(
        "--kd_flag",
        default=0,
        type=int,
        help="Whether to enable distillation (only DiscNet is 1 )",
    )
    parser.add_argument("--kd_weight", default=100000, type=int, help="KD loss weight")
    parser.add_argument(
        "--gnn_iter_times",
        default=3,
        type=int,
        help="Number of message passing for V2VNet",
    )
    parser.add_argument(
        "--visualization", type=int, default=0, help="Visualize validation result"
    )
    parser.add_argument(
        "--com", default="", type=str, help="disco/when2com/v2v/sum/mean/max/cat/agent"
    )
    parser.add_argument(
        "--bound",
        type=str,
        help="The input setting: lowerbound -> single-view or upperbound -> multi-view",
    )
    parser.add_argument("--inference", type=str)
    parser.add_argument("--tracking", action="store_true")
    parser.add_argument("--box_com", action="store_true")
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    # scene_batch => batch size in each scene
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--apply_late_fusion",
        default=0,
        type=int,
        help="1: apply late fusion. 0: no late fusion",
    )
    parser.add_argument(
        "--compress_level",
        default=0,
        type=int,
        help="Compress the communication layer channels by 2**x times in encoder",
    )
    parser.add_argument(
        "--pose_noise",
        default=0,
        type=float,
        help="draw noise from normal distribution with given mean (in meters), apply to transformation matrix.",
    )
    parser.add_argument(
        "--only_v2i",
        default=0,
        type=int,
        help="1: only v2i, 0: v2v and v2i",
    )

    ## ----- args for completion model load ----
    parser.add_argument(
        "--resume_completion",
        default="",
        type=str,
        help="The path to the saved completion model that is loaded to resume training",
    )

    ## ----- mae args added by Juexiao -----
    parser.add_argument(
        "--mask_ratio", default=0.50, type=float, help="The input mask ratio"
    )
    parser.add_argument(
        "--mask", default="random", type=str, choices=["random", "complement"], help="Used in MAE training"
    )
    parser.add_argument(
        "--time_stamp", default=2, type=int, help="The total number of time stamp to use"
    )
    parser.add_argument(
        "--mae_model", default="fusion_bev_mae_vit_base_patch8", type=str, 
        help="The mae model to use"
    )
    parser.add_argument(
        "--norm_pix_loss", default=False, type=bool, help="Whether normalize target pixel value for loss"
    )
    ## ----------------------
    parser.add_argument(
        "--encode_partial", action="store_true", help="encode partial information before masking"
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    print(args)
    main(args)