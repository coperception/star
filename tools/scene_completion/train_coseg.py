# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from coperception.datasets import V2XSimSeg, MultiTempV2XSimSeg
from coperception.configs import Config, ConfigGlobal
from coperception.utils.SegModule import *
from coperception.utils.loss import *
from coperception.models.seg import UNet

from coperception.utils.AverageMeter import AverageMeter
from coperception.utils.data_util import apply_pose_noise
import glob
import os

from coperception.models.transformers import multiagent_mae
from coperception.models.det import *
import timm
assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def main(config, args):
    config.nepoch = args.nepoch
    num_epochs = args.nepoch
    need_log = args.log
    batch_size = args.batch
    num_workers = args.nworker
    compress_level = args.compress_level
    start_epoch = 1
    pose_noise = args.pose_noise
    only_v2i = args.only_v2i

    # Specify gpu device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)

    auto_resume_path = args.auto_resume_path
    
    kd_flag = False

    if args.bound == "upperbound":
        flag = "upperbound"
    else:
        if args.com == "when2com":
            if args.warp_flag:
                flag = "when2com_warp"
            else:
                flag = "when2com"
        elif args.com == "v2v":
            flag = "v2v"
        elif args.com == "mean":
            flag = "mean"
        elif args.com == "max":
            flag = "max"
        elif args.com == "sum":
            flag = "sum"
        elif args.com == "agent":
            flag = "agent"
        elif args.com == "cat":
            flag = "cat"
        elif args.com == "disco":
            flag = "disco"
            kd_flag = True
        elif args.com == "ind_mae":
            flag = "mae"
        elif args.com == "joint_mae":
            flag = "joint_mae"
        else:
            flag = "lowerbound"
    config.flag = flag

    num_agent = args.num_agent
    agent_idx_range = range(1, num_agent) if args.no_cross_road else range(num_agent)

    # trainset = V2XSimSeg(
    #     dataset_roots=[args.data + "/agent%d" % i for i in agent_idx_range],
    #     config=config,
    #     split="train",
    #     com=args.com,
    #     bound=args.bound,
    #     kd_flag=kd_flag,
    #     no_cross_road=args.no_cross_road,
    # )
    trainset = MultiTempV2XSimSeg(
        dataset_roots=[args.data + "/agent%d" % i for i in agent_idx_range],
        config=config,
        split="train",
        com=args.com,
        bound=args.bound,
        kd_flag=kd_flag,
        no_cross_road=args.no_cross_road,
        time_stamp = args.time_stamp,
    )
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    print("Training dataset size:", len(trainset))

    logger_root = args.logpath if args.logpath != "" else "logs"
    model_save_path = os.path.join(logger_root, flag)

    if args.no_cross_road:
        model_save_path = os.path.join(model_save_path, "no_cross")
    else:
        model_save_path = os.path.join(model_save_path, "with_cross")
    cross_path = "no_cross" if args.no_cross_road else "with_cross"
    os.makedirs(model_save_path, exist_ok=True)

    # build model
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
        model_completion = multiagent_mae.__dict__[args.mae_model](norm_pix_loss=args.norm_pix_loss, time_stamp=args.time_stamp, mask_method=args.mask)
        # also include individual reconstruction: reconstruct then aggregate
    else:
        raise NotImplementedError("Fusion type: {args.com} not implemented")

    # Model for segmentation
    model = UNet(
        config.in_channels,
        config.num_class,
        num_agent=num_agent,
        compress_level=compress_level,
    )
    # model = nn.DataParallel(model)
    model = model.to(device)
    # model_completion = model_completion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_completion = torch.load(args.resume_completion, map_location='cpu')
    completion_start_epoch = checkpoint_completion["epoch"] + 1
    model_completion.load_state_dict(checkpoint_completion["model_state_dict"])
    model_completion = model_completion.to(device)
    model_completion.train() # but will not actually update weights
    print("completion model loaded from {} at epoch {}".format(args.resume_completion, completion_start_epoch))

    if not args.fine_tune:
        print("do not fine tune the completion model..")
        for param in model_completion.parameters():
            param.requires_grad = False

    # config.com = args.com
    config.com = "" # Seg module do not communicate

    if kd_flag:
        teacher = UNet(
            config.in_channels, config.num_class, num_agent=num_agent, kd_flag=True
        )
        teacher = teacher.to(device)
        seg_module = SegModule(model, teacher, config, optimizer, kd_flag)
        checkpoint_teacher = torch.load(args.resume_teacher)
        start_epoch_teacher = checkpoint_teacher["epoch"]
        seg_module.teacher.load_state_dict(checkpoint_teacher["model_state_dict"])
        print(
            "Load teacher model from {}, at epoch {}".format(
                args.resume_teacher, start_epoch_teacher
            )
        )
        seg_module.teacher.eval()
    else:
        seg_module = SegModule(model, None, config, optimizer, kd_flag)

    if args.resume is None and (
        args.auto_resume_path == ""
        or "seg_epoch_1.pth"
        not in os.listdir(os.path.join(args.auto_resume_path, f"{flag}/{cross_path}"))
    ):
        log_file_name = os.path.join(model_save_path, "log.txt")
        saver = open(log_file_name, "w")
        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()
        print("no trained model is found.")

    else:
        if args.auto_resume_path != "":
            model_save_path = os.path.join(
                args.auto_resume_path, f"{flag}/{cross_path}"
            )
        else:
            model_save_path = args.resume[: args.resume.rfind("/")]

        print(f"model save path: {model_save_path}")

        log_file_name = os.path.join(model_save_path, "log.txt")
        if os.path.exists(log_file_name):
            saver = open(log_file_name, "a")
        else:
            os.makedirs(model_save_path)
            saver = open(log_file_name, "w")

        saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
        saver.flush()

        # Logging the details for this experiment
        saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
        saver.write(args.__repr__() + "\n\n")
        saver.flush()

        if args.auto_resume_path != "":
            list_of_files = glob.glob(f"{model_save_path}/*.pth")
            latest_pth = max(list_of_files, key=os.path.getctime)
            checkpoint = torch.load(latest_pth)
        else:
            checkpoint = torch.load(args.resume)

        seg_module.model.load_state_dict(checkpoint["model_state_dict"])
        seg_module.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        seg_module.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

        print(
            "Load model from {}, at epoch {}".format(
                args.resume if args.resume is not None else args.auto_resume_path,
                start_epoch - 1,
            )
        )

    for epoch in range(start_epoch, num_epochs + 1):
        lr = seg_module.optimizer.param_groups[0]["lr"]
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        running_loss_disp = AverageMeter(
            "Total loss", ":.6f"
        )  # for motion prediction error
        seg_module.model.train()

        t = time.time()
        for idx, sample in enumerate(tqdm(trainloader)):

            if args.com:
                (
                    padded_voxel_points_list,
                    padded_voxel_points_next_list,
                    padded_voxel_points_teacher_list,
                    label_one_hot_list,
                    trans_matrices,
                    target_agent,
                    num_sensor,
                ) = list(zip(*sample))
            else:
                (
                    padded_voxel_points_list,
                    padded_voxel_points_next_list,
                    padded_voxel_points_teacher_list,
                    label_one_hot_list,
                ) = list(zip(*sample))

            if flag == "upperbound":
                padded_voxel_points = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
            else:
                padded_voxel_points = torch.cat(tuple(padded_voxel_points_list), 0)

            label_one_hot = torch.cat(tuple(label_one_hot_list), 0)
            
            num_all_agents = torch.stack(tuple(num_sensor), 1)
            if args.no_cross_road:
                num_all_agents -= 1
            trans_matrices = torch.stack(trans_matrices, 1)

            padded_voxel_points_teacher = torch.cat(tuple(padded_voxel_points_teacher_list), 0)
            # print(padded_voxel_points_teacher.size())
            padded_voxel_points_next = torch.cat(tuple(padded_voxel_points_next_list), 0)
            # print(padded_voxel_points_next.size())
            if padded_voxel_points_next.size(1) >0:
                padded_voxel_points_next = padded_voxel_points_next.permute(0, 1, 4, 2, 3)
            
            completion_data = {
                'bev_seq': padded_voxel_points.squeeze(1).permute(0,3,1,2).to(device).float(),
                'bev_seq_next': padded_voxel_points_next.to(device).float(),
                'bev_teacher': padded_voxel_points_teacher.squeeze(1).permute(0,3,1,2).to(device),
                'num_agent_tensor': num_all_agents.to(device),
                'trans_matrices': trans_matrices,
            }
            # print(num_all_agents)
            # infer completion
            with torch.no_grad():
                _, completed_point_cloud, _, _ = model_completion.inference(completion_data['bev_seq'], 
                                                    completion_data['bev_seq_next'], 
                                                    completion_data['bev_teacher'], 
                                                    completion_data['trans_matrices'], 
                                                    completion_data['num_agent_tensor'],
                                                    batch_size,
                                                    mask_ratio=args.mask_ratio)

            # completed_point_cloud = completed_point_cloud.unsqueeze(1)
            completed_point_cloud = completed_point_cloud.permute(0, 2, 3, 1)
            # print("point cloud size", completed_point_cloud.size())

            data = {}
            # substitute with the completed data
            data["bev_seq"] = completed_point_cloud.to(device)
            # data["bev_seq"] = padded_voxel_points.to(device).float()
            data["labels"] = label_one_hot.to(device)
            if args.com:
                # trans_matrices = torch.stack(trans_matrices, 1)
                # add pose noise
                if pose_noise > 0:
                    apply_pose_noise(pose_noise, trans_matrices)
                target_agent = torch.stack(target_agent, 1)
                num_sensor = torch.stack(num_sensor, 1)
                data["trans_matrices"] = trans_matrices.to(device)
                data["target_agent"] = target_agent

                if args.no_cross_road:
                    num_sensor -= 1

                data["num_sensor"] = num_sensor

            if kd_flag:
                padded_voxel_points_teacher = torch.cat(
                    tuple(padded_voxel_points_teacher_list), 0
                )
                data["bev_seq_teacher"] = padded_voxel_points_teacher.to(device)
                data["kd_weight"] = args.kd_weight

            pred, loss = seg_module.step(data, num_agent, batch_size)

            running_loss_disp.update(loss)
        print("\nEpoch {}".format(epoch))
        print("Running total loss: {}".format(running_loss_disp.avg))
        seg_module.scheduler.step()
        print("{}\t Takes {} s\n".format(running_loss_disp, str(time.time() - t)))

        if need_log:
            saver.write("{}\n".format(running_loss_disp))
            saver.flush()

        # save model
        if need_log:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": seg_module.model.state_dict(),
                "optimizer_state_dict": seg_module.optimizer.state_dict(),
                "scheduler_state_dict": seg_module.scheduler.state_dict(),
                "loss": running_loss_disp.avg,
            }
            print(model_save_path)
            torch.save(
                save_dict, os.path.join(model_save_path, "seg_epoch_" + str(epoch) + ".pth")
            )
    if need_log:
        saver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        default="./dataset/train",
        type=str,
        help="The path to the preprocessed sparse BEV training data",
    )
    parser.add_argument("--bound")
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="The path to the saved model that is loaded to resume training",
    )
    parser.add_argument("--model_only", action="store_true", help="only load model")
    parser.add_argument("--batch", default=1, type=int, help="Batch size")
    parser.add_argument("--warp_flag", action="store_true")
    parser.add_argument(
        "--augmentation", default=False, help="Whether to use data augmentation"
    )
    parser.add_argument("--nepoch", default=100, type=int, help="Number of epochs")
    parser.add_argument("--nworker", default=2, type=int, help="Number of workers")
    parser.add_argument("--lr", default=0.001, type=float, help="Initial learning rate")
    parser.add_argument("--log", action="store_true", help="Whether to log")
    parser.add_argument("--logpath", default="", help="The path to the output log file")
    parser.add_argument("--com", default="", type=str, help="Whether to communicate")
    parser.add_argument(
        "--no_cross_road", action="store_true", help="Do not load data of cross roads"
    )
    parser.add_argument(
        "--num_agent", default=6, type=int, help="The total number of agents"
    )
    parser.add_argument(
        "--resume_teacher",
        default="",
        type=str,
        help="The path to the saved teacher model that is loaded to resume training",
    )
    parser.add_argument("--kd_weight", default=100, type=int, help="KD loss weight")
    parser.add_argument(
        "--auto_resume_path",
        default="",
        type=str,
        help="The path to automatically reload the latest pth",
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
        "--resume_completion",
        default="",
        type=str,
        help="The path to the saved completion model that is loaded to resume training",
    )
    parser.add_argument(
        "--fine_tune",
        default=0,
        type=int,
        help="Fine tune the completion model or not",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")

    args = parser.parse_args()
    print(args)
    config = Config("train")
    main(config, args)